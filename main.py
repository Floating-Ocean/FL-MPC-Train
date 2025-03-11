import copy
from argparse import Namespace
from dataclasses import dataclass

import cupy as cp
import enlighten
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.multiprocessing import Value, Process, Queue, set_start_method
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model.cnn import CNN
from security.shamir import split_sharing, recover_sharing, recover_weight_shape
from util.options import args_parser
from util.sampling import split_iid_data
from util.test import test_acc
from util.update import DatasetSplit


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


def split_data(args: Namespace, indices: dict[int, set],
               data: DataLoader, is_train: bool = True) -> list[DataLoader]:
    return [DataLoader(DatasetSplit(data.dataset, idxes),
                       batch_size=args.local_bs if is_train else args.bs,
                       shuffle=is_train)
            for (_, idxes) in indices.items()]


def local_train(args: Namespace, model: nn.Module, train_data: DataLoader, global_weights=None, mu=-1.0) -> float:
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    epoch_loss = []

    for _ in range(args.local_ep):
        batch_loss = []
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
            batch_loss.append(loss.item())
        epoch_loss.append(sum(batch_loss) / len(batch_loss))

    return sum(epoch_loss) / len(epoch_loss)


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

    mpc_loss_locals = []
    print(f"Rd.{current_epoch + 1} Start.")
    communicate_queue.put(('mpc', 0))

    server_received_split: dict[int, list[cp.ndarray]] = {server_id: [] for server_id in range(args.num_servers)}
    for client_id in range(args.num_users):
        # Step.1 本地训练
        current_loss = local_train(args, client_models[client_id].model, train_data[client_id])
        mpc_loss_locals.append(copy.deepcopy(current_loss))
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

    mpc_ave_losses = sum(mpc_loss_locals) / len(mpc_loss_locals)

    communicate_queue.put(('epoch', 1))
    communicate_queue.put(('mpc_r',
                           ({i: model.clone_to_device(torch.device('cpu')) for i, model in client_models.items()},
                            mpc_ave_losses)))

    while terminate_signal.value == 0:
        pass

    torch.cuda.empty_cache()


def avg_threading(args, train_data, avg_net_glob, communicate_queue, terminate_signal):
    avg_w_locals, avg_loss_locals = [], []
    communicate_queue.put(('avg', 0))
    for client_id in range(args.num_users):
        now_net_glob = copy.deepcopy(avg_net_glob).to(args.device)
        current_loss = local_train(args, now_net_glob, train_data[client_id])
        avg_w_locals.append(copy.deepcopy(now_net_glob.state_dict()))
        avg_loss_locals.append(copy.deepcopy(current_loss))
        communicate_queue.put(('avg', 1))

    avg_w_glob = fed_avg(avg_w_locals)
    avg_net_glob.load_state_dict(avg_w_glob)
    avg_ave_losses = sum(avg_loss_locals) / len(avg_loss_locals)

    communicate_queue.put(('epoch', 1))
    communicate_queue.put(('avg_r', (copy.deepcopy(avg_net_glob).to(torch.device('cpu')), avg_ave_losses)))

    while terminate_signal.value == 0:
        pass

    torch.cuda.empty_cache()


def prox_threading(args, train_data, prox_w_glob_p, prox_net_glob, communicate_queue, terminate_signal):
    # 复制传参
    prox_w_glob = {key: copy.deepcopy(val).to(args.device) for key, val in prox_w_glob_p.items()}

    prox_w_locals, prox_loss_locals = [], []
    communicate_queue.put(('prox', 0))
    for client_id in range(args.num_users):
        now_net_glob = copy.deepcopy(prox_net_glob).to(args.device)
        current_loss = local_train(args, now_net_glob, train_data[client_id], prox_w_glob, 0.01)
        prox_w_locals.append(copy.deepcopy(now_net_glob.state_dict()))
        prox_loss_locals.append(copy.deepcopy(current_loss))
        communicate_queue.put(('prox', 1))

    prox_w_glob = fed_avg(prox_w_locals)
    prox_net_glob.load_state_dict(prox_w_glob)
    prox_ave_losses = sum(prox_loss_locals) / len(prox_loss_locals)

    communicate_queue.put(('epoch', 1))
    communicate_queue.put(('prox_r', ({key: copy.deepcopy(val).to(torch.device('cpu'))
                                       for key, val in prox_w_glob.items()},
                                      copy.deepcopy(prox_net_glob).to(torch.device('cpu')), prox_ave_losses)))

    while terminate_signal.value == 0:
        pass

    torch.cuda.empty_cache()


def solve():
    # 命令行参数
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # 加载并划分数据
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        origin_train_data = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        origin_test_data = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        train_dataloader, test_dataloader = DataLoader(origin_train_data), DataLoader(origin_test_data)
        # 为各用户采样数据
        data_indices = split_iid_data(args, train_dataloader)
        train_data = split_data(args, data_indices, train_dataloader)
        test_data = split_data(args, data_indices, test_dataloader, False)
    else:
        exit('Error: unrecognized dataset')

    # 初始模型
    if args.model == 'cnn' and args.dataset == 'mnist':
        net_init = CNN().to(args.device)
    else:
        exit('Error: unrecognized model')

    print(net_init)
    print()
    net_init.train()

    # 对照组 FedAvg 和 FedProx 的初始化
    avg_net_glob, prox_net_glob = copy.deepcopy(net_init), copy.deepcopy(net_init)
    prox_w_glob = prox_net_glob.state_dict()

    client_models: dict[int, ModelWrapper] = {i: ModelWrapper(model=copy.deepcopy(net_init))
                                              for i in range(args.num_users)}

    mpc_loss_trains, avg_loss_trains, prox_loss_trains = [], [], []  # 用于绘图

    with manager.counter(total=args.epochs, desc='训练进度', color="bold_cyan", leave=False) as overall_pbar:
        for current_epoch in range(args.epochs):
            with manager.counter(total=6, desc=f'第{current_epoch + 1}轮训练',
                                 color="bold_yellow", leave=False) as epoch_bar:
                epoch_bar.update()
                overall_pbar.update(incr=0)

                mpc_counter = manager.counter(total=args.num_users * 2 + args.num_servers, desc='Mpc训练',
                                              color="bold_green", leave=False)
                avg_counter = manager.counter(total=args.num_users, desc='Avg训练',
                                              color="bold_green", leave=False)
                prox_counter = manager.counter(total=args.num_users, desc='Prox训练',
                                               color="bold_green", leave=False)

                communicate_queue = Queue()
                terminate_signal = Value('i', 0)

                mpc_thread = Process(target=mpc_threading,
                                     args=(args, current_epoch, train_data,
                                           {i: client.clone_to_device(torch.device('cpu'))
                                            for i, client in client_models.items()}, communicate_queue,
                                           terminate_signal))
                avg_thread = Process(target=avg_threading,
                                     args=(args, train_data,
                                           copy.deepcopy(avg_net_glob).to(torch.device('cpu')),
                                           communicate_queue, terminate_signal))
                prox_thread = Process(target=prox_threading,
                                      args=(args, train_data,
                                            {key: copy.deepcopy(val).to(torch.device('cpu'))
                                             for key, val in prox_w_glob.items()},
                                            copy.deepcopy(prox_net_glob).to(torch.device('cpu')),
                                            communicate_queue, terminate_signal))

                mpc_thread.start()
                avg_thread.start()
                prox_thread.start()

                returned_count = 0
                while returned_count < 3:
                    try:
                        msg = communicate_queue.get()
                        if msg[0] == 'mpc':
                            mpc_counter.update(msg[1])
                        elif msg[0] == 'avg':
                            avg_counter.update(msg[1])
                        elif msg[0] == 'prox':
                            prox_counter.update(msg[1])
                        elif msg[0] == 'epoch':
                            epoch_bar.update(msg[1])
                        elif msg[0] == 'mpc_r':
                            client_models_p, mpc_loss_train = msg[1]
                            client_models = {i: client.clone_to_device(args.device)
                                             for i, client in client_models_p.items()}
                            mpc_loss_trains.append(mpc_loss_train)
                            returned_count += 1
                        elif msg[0] == 'avg_r':
                            avg_net_glob_p, avg_loss_train = msg[1]
                            avg_net_glob = copy.deepcopy(avg_net_glob_p).to(args.device)
                            avg_loss_trains.append(avg_loss_train)
                            returned_count += 1
                        elif msg[0] == 'prox_r':
                            prox_w_glob_p, prox_net_glob_p, prox_loss_train = msg[1]
                            prox_w_glob = {key: copy.deepcopy(val).to(args.device)
                                           for key, val in prox_w_glob_p.items()}
                            prox_net_glob = copy.deepcopy(prox_net_glob_p).to(args.device)
                            prox_loss_trains.append(prox_loss_train)
                            returned_count += 1
                    except KeyboardInterrupt:
                        break

                terminate_signal.value = 1

                mpc_thread.join()
                avg_thread.join()
                prox_thread.join()

                mpc_counter.close()
                avg_counter.close()
                prox_counter.close()

                # 本轮结束，衡量准确度
                print(f"OK, 评估中")
                mpc_acc, mpc_loss = test_acc(args, train_data, client_models[0].model)
                avg_acc, avg_loss = test_acc(args, train_data, avg_net_glob)
                prox_acc, prox_loss = test_acc(args, train_data, prox_net_glob)

                print('\n'.join(["Mpc_acc: {:.2f}%, loss: {:.4f}".format(mpc_acc, mpc_loss),
                                 "Avg_acc: {:.2f}%, loss: {:.4f}".format(avg_acc, avg_loss),
                                 "Prox_acc: {:.2f}%, loss: {:.4f}".format(prox_acc, prox_loss)]))
                print()
                overall_pbar.update()

        print("All OK, 评估中")

        # plot loss curve
        plt.figure()
        plt.plot(range(len(mpc_loss_trains)), mpc_loss_trains)
        plt.plot(range(len(avg_loss_trains)), avg_loss_trains)
        plt.plot(range(len(prox_loss_trains)), prox_loss_trains)
        plt.legend(['MPC', 'AVG', 'PROX'])
        plt.ylabel('train_loss')
        plt.savefig('./save/mpc_avg_prox_{}_{}_{}.png'
                    .format(args.dataset, args.model, args.epochs))

        # 训练结束，测试准度
        client_models[0].model.eval()
        mpc_acc_train, _ = test_acc(args, train_data, client_models[0].model)
        mpc_acc_test, _ = test_acc(args, test_data, client_models[0].model)

        avg_net_glob.eval()
        avg_acc_train, _ = test_acc(args, train_data, avg_net_glob)
        avg_acc_test, _ = test_acc(args, test_data, avg_net_glob)

        prox_net_glob.eval()
        prox_acc_train, _ = test_acc(args, train_data, prox_net_glob)
        prox_acc_test, _ = test_acc(args, test_data, prox_net_glob)

        print(
            "训练集Acc: {:.2f} (MPC), {:.2f} (AVG), {:.2f} (PROX)".format(mpc_acc_train, avg_acc_train, prox_acc_train))
        print("测试集Acc: {:.2f} (MPC), {:.2f} (AVG), {:.2f} (PROX)".format(mpc_acc_test, avg_acc_test, prox_acc_test))


if __name__ == '__main__':
    set_start_method(method='spawn', force=True)
    with enlighten.get_manager() as manager:
        solve()
