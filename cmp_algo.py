import copy

import enlighten
import torch
from matplotlib import pyplot as plt
from torch.multiprocessing import Value, Process, Queue, set_start_method

from training import ModelWrapper, mpc_threading, avg_threading, prox_threading
from util.dataset import available_models, DatasetWrapper
from util.options import args_parser
from util.test import test_acc


def solve():
    # 命令行参数
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # 加载并划分数据
    if args.dataset in available_models:
        dataset: DatasetWrapper = available_models[args.dataset](args)
    else:
        exit('Error: unrecognized dataset')

    # 初始模型
    if args.model == 'cnn':
        net_init = dataset.init_cnn().to(args.device)
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

    mpc_acc_trains, avg_acc_trains, prox_acc_trains = [], [], []  # 用于acc绘图
    mpc_loss_trains, avg_loss_trains, prox_loss_trains = [], [], []  # 用于loss绘图

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
                                     args=(args, current_epoch, dataset.train_data,
                                           {i: client.clone_to_device(torch.device('cpu'))
                                            for i, client in client_models.items()}, communicate_queue,
                                           terminate_signal))
                avg_thread = Process(target=avg_threading,
                                     args=(args, dataset.train_data,
                                           copy.deepcopy(avg_net_glob).to(torch.device('cpu')),
                                           communicate_queue, terminate_signal))
                prox_thread = Process(target=prox_threading,
                                      args=(args, dataset.train_data,
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
                            client_models_p = msg[1]
                            client_models = {i: client.clone_to_device(args.device)
                                             for i, client in client_models_p.items()}
                            returned_count += 1
                        elif msg[0] == 'avg_r':
                            avg_net_glob_p = msg[1]
                            avg_net_glob = copy.deepcopy(avg_net_glob_p).to(args.device)
                            returned_count += 1
                        elif msg[0] == 'prox_r':
                            prox_w_glob_p, prox_net_glob_p = msg[1]
                            prox_w_glob = {key: copy.deepcopy(val).to(args.device)
                                           for key, val in prox_w_glob_p.items()}
                            prox_net_glob = copy.deepcopy(prox_net_glob_p).to(args.device)
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
                mpc_acc, mpc_loss = test_acc(args, dataset.train_data, client_models[0].model)
                avg_acc, avg_loss = test_acc(args, dataset.train_data, avg_net_glob)
                prox_acc, prox_loss = test_acc(args, dataset.train_data, prox_net_glob)

                print('\n'.join(["Mpc_acc: {:.2f}%, loss: {:.4f}".format(mpc_acc, mpc_loss),
                                 "Avg_acc: {:.2f}%, loss: {:.4f}".format(avg_acc, avg_loss),
                                 "Prox_acc: {:.2f}%, loss: {:.4f}".format(prox_acc, prox_loss)]))
                print()
                mpc_acc_trains.append(mpc_acc / 100.0)
                avg_acc_trains.append(avg_acc / 100.0)
                prox_acc_trains.append(prox_acc / 100.0)
                mpc_loss_trains.append(mpc_loss)
                avg_loss_trains.append(avg_loss)
                prox_loss_trains.append(prox_loss)
                overall_pbar.update()

        print("All OK, 评估中")

        # plot acc curve
        plt.figure()
        plt.plot(range(len(mpc_acc_trains)), mpc_acc_trains)
        plt.plot(range(len(avg_acc_trains)), avg_acc_trains)
        plt.plot(range(len(prox_acc_trains)), prox_acc_trains)
        plt.legend(['MPC', 'AVG', 'PROX'])
        plt.ylabel('Train accuracy')
        plt.savefig('./save/acc_mpc_avg_prox_{}_{}_{}.png'
                    .format(args.dataset, args.model, args.epochs))

        # plot loss curve
        plt.figure()
        plt.plot(range(len(mpc_loss_trains)), mpc_loss_trains)
        plt.plot(range(len(avg_loss_trains)), avg_loss_trains)
        plt.plot(range(len(prox_loss_trains)), prox_loss_trains)
        plt.legend(['MPC', 'AVG', 'PROX'])
        plt.ylabel('Train loss')
        plt.savefig('./save/loss_mpc_avg_prox_{}_{}_{}.png'
                    .format(args.dataset, args.model, args.epochs))

        # 训练结束，测试准度
        client_models[0].model.eval()
        mpc_acc_train, _ = test_acc(args, dataset.train_data, client_models[0].model)
        mpc_acc_test, _ = test_acc(args, dataset.test_data, client_models[0].model)

        avg_net_glob.eval()
        avg_acc_train, _ = test_acc(args, dataset.train_data, avg_net_glob)
        avg_acc_test, _ = test_acc(args, dataset.test_data, avg_net_glob)

        prox_net_glob.eval()
        prox_acc_train, _ = test_acc(args, dataset.train_data, prox_net_glob)
        prox_acc_test, _ = test_acc(args, dataset.test_data, prox_net_glob)

        print(
            "训练集Acc: {:.2f} (MPC), {:.2f} (AVG), {:.2f} (PROX)".format(mpc_acc_train, avg_acc_train, prox_acc_train))
        print("测试集Acc: {:.2f} (MPC), {:.2f} (AVG), {:.2f} (PROX)".format(mpc_acc_test, avg_acc_test, prox_acc_test))


if __name__ == '__main__':
    set_start_method(method='spawn', force=True)
    with enlighten.get_manager() as manager:
        solve()
