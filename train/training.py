import copy
import logging
from argparse import Namespace
from dataclasses import dataclass

import cupy as cp
import torch
from torch import nn
from torch.multiprocessing import Value, Queue
from torch.utils.data import DataLoader

from security.shamir import split_sharing, recover_sharing, recover_weight_shape


@dataclass
class ModelWrapper:
    model: nn.Module

    def __init__(self, model: nn.Module):
        self.model = model

    def clone_to_device(self, device: torch.device):
        self.model.zero_grad(set_to_none=True)  # 防止梯度跨进程
        copied = ModelWrapper(copy.deepcopy(self.model).to(device))
        return copied


def aggregate_models(models: list[tuple[nn.Module, int]]) -> nn.Module:
    aggregated_param = {k: torch.zeros_like(v) for k, v in models[0][0].state_dict().items()}
    total_weight = sum(weight for _, weight in models)
    # 计算各个模型的加权weight
    for model, weight in models:
        model_state_dict = model.state_dict()
        for key in aggregated_param.keys():
            aggregated_param[key] += model_state_dict[key] * weight / total_weight  # 加权平均
    aggregated_model = copy.deepcopy(models[0][0])
    aggregated_model.load_state_dict(aggregated_param)
    return aggregated_model


def local_train(args: Namespace, model: nn.Module, train_data: DataLoader, global_weights=None, mu=-1.0):
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for _ in range(args.local_ep):
        for _, (data, target) in enumerate(train_data):
            if args.gpu != -1:
                data, target = data.cuda(), target.cuda()
            model.zero_grad()  # 清空累积梯度
            y_logit = model(data)
            loss = torch.nn.CrossEntropyLoss()(y_logit, target)

            if mu != -1.0:
                # 添加FedProx正则化项
                fed_prox_loss = 0.5 * mu * sum(
                    [(model.state_dict()[k] - global_weights[k]).pow(2).sum() for k in global_weights])
                loss += fed_prox_loss

            loss.backward()
            optimizer.step()


def fed_avg(w: list[dict]) -> dict:
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg


def eval_client_loss(args: Namespace, test_data: DataLoader, model: nn.Module):
    losses = 0
    model.eval()

    for idx, (data, target) in enumerate(test_data):
        if args.gpu != -1:
            data, target = data.cuda(), target.cuda()
        y_logit = model(data)
        losses += torch.nn.CrossEntropyLoss()(y_logit, target).item()

    losses /= len(test_data.dataset)
    return losses


def mpc_threading(args: Namespace, current_epoch: int, train_data: list[DataLoader],
                  client_models_p: dict[int, ModelWrapper], communicate_queue: Queue, terminate_signal: Value):
    # 复制传参
    client_models = {i: client.clone_to_device(args.device) for i, client in client_models_p.items()}

    logging.info(f"Rd.{current_epoch + 1} Start.")
    communicate_queue.put(('mpc', 0))

    server_received_split: dict[int, list[cp.ndarray]] = {server_id: [] for server_id in range(args.num_servers)}
    for client_id in range(args.num_users):
        # Step.1 本地训练
        local_train(args, client_models[client_id].model, train_data[client_id])
        # Step.2 分片
        splits = split_sharing(args, client_models[client_id].model)
        # Step.3 发送至服务器
        for server_id, split in splits.items():
            server_received_split[server_id].append(split)
        communicate_queue.put(('mpc', 1))

    communicate_queue.put(('epoch', 1))

    client_received_split: dict[int, dict[int, tuple[cp.ndarray, int]]] = {client_id: {}
                                                                           for client_id in range(args.num_users)}
    for server_id in range(args.num_servers):
        # Step.4 累加收到的分片
        merged = cp.sum(cp.stack(server_received_split[server_id]), axis=0)
        for client_id in range(args.num_users):
            client_received_split[client_id][server_id] = (merged, len(server_received_split[server_id]))
        communicate_queue.put(('mpc', 1))

    communicate_queue.put(('epoch', 1))

    # Step.5 恢复数据
    for client_id in range(args.num_users):
        # 检查与各个服务端通信的客户端数量是否一致
        client_counts = [data[1] for server_id, data in client_received_split[client_id].items()]
        assert len(set(client_counts)) == 1

        recovered = recover_sharing(
            args,
            {server_id: data[0] for server_id, data in client_received_split[client_id].items()}
        )
        recovered_weight = recover_weight_shape(recovered, client_models[client_id].model)
        aggregated_weight = {name: torch.div(param, client_counts[0]) for name, param in recovered_weight.items()}
        client_models[client_id].model.load_state_dict(aggregated_weight)
        communicate_queue.put(('mpc', 1))

    communicate_queue.put(('epoch', 1))
    communicate_queue.put(('mpc_r',
                           {i: model.clone_to_device(torch.device('cpu')) for i, model in client_models.items()}))

    while terminate_signal.value == 0:
        pass

    torch.cuda.empty_cache()


def avg_threading(args: Namespace, train_data: list[DataLoader], avg_net_glob: nn.Module,
                  communicate_queue: Queue, terminate_signal: Value):
    avg_w_locals = []
    communicate_queue.put(('avg', 0))
    for client_id in range(args.num_users):
        now_net_glob = copy.deepcopy(avg_net_glob).to(args.device)
        local_train(args, now_net_glob, train_data[client_id])
        avg_w_locals.append(copy.deepcopy(now_net_glob.state_dict()))
        communicate_queue.put(('avg', 1))

    avg_w_glob = fed_avg(avg_w_locals)
    avg_net_glob.load_state_dict(avg_w_glob)

    communicate_queue.put(('epoch', 1))
    communicate_queue.put(('avg_r', copy.deepcopy(avg_net_glob).to(torch.device('cpu'))))

    while terminate_signal.value == 0:
        pass

    torch.cuda.empty_cache()


def prox_threading(args: Namespace, train_data: list[DataLoader], prox_w_glob_p: dict,
                   prox_net_glob: nn.Module, communicate_queue: Queue, terminate_signal: Value):
    # 复制传参
    prox_w_glob = {key: copy.deepcopy(val).to(args.device) for key, val in prox_w_glob_p.items()}

    prox_w_locals = []
    communicate_queue.put(('prox', 0))
    for client_id in range(args.num_users):
        now_net_glob = copy.deepcopy(prox_net_glob).to(args.device)
        local_train(args, now_net_glob, train_data[client_id], prox_w_glob, 0.01)
        prox_w_locals.append(copy.deepcopy(now_net_glob.state_dict()))
        communicate_queue.put(('prox', 1))

    prox_w_glob = fed_avg(prox_w_locals)
    prox_net_glob.load_state_dict(prox_w_glob)

    communicate_queue.put(('epoch', 1))
    communicate_queue.put(('prox_r', ({key: copy.deepcopy(val).to(torch.device('cpu'))
                                       for key, val in prox_w_glob.items()},
                                      copy.deepcopy(prox_net_glob).to(torch.device('cpu')))))

    while terminate_signal.value == 0:
        pass

    torch.cuda.empty_cache()