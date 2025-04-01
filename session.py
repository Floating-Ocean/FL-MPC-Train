import copy
import traceback
from multiprocessing.managers import DictProxy
from uuid import UUID

import torch
from PIL import Image
from matplotlib import pyplot as plt
from torch.multiprocessing import Value, Process, Queue

from .training import ModelWrapper, mpc_threading
from .util.dataset import available_models, DatasetWrapper
from .util.options import get_default_args
from .util.test import test_acc


def open_session(uuid: UUID, epochs: int, dataset_name: str, model_output_folder: str, status_dict: DictProxy):
    """
    新建训练通信，需要在multiprocessing.Process中作为target执行

    Args:
        uuid: 本次训练的唯一标识符
        epochs: 训练轮次
        dataset_name: 训练使用的数据集
        model_output_folder: 训练结束后存储模型的文件夹，导出为 [uuid].pth 文件
        status_dict: 用于通信训练状态

    status_dict包含两个字段：
    status字段为当前状态，候选为 'INITIALIZING', 'FAILED', 'TRAINING', 'EVALUATING', 'FINISHED'；
    data字段为当前状态的数据，INITIALIZING和EVALUATING对应None，FAILED对应一个错误提示字符串，其余对应一个dict.

    """
    status_dict[uuid] = {
        'status': 'INITIALIZING',
        'data': None
    }
    try:
        args = get_default_args()
        args.device = torch.device(
            'cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
        args.epochs = epochs
        args.dataset = dataset_name

        # 加载并划分数据
        if args.dataset in available_models:
            dataset: DatasetWrapper = available_models[args.dataset](args)
        else:
            raise TypeError('Unknown dataset: {}'.format(args.dataset))

        # 初始模型
        if args.model == 'cnn':
            net_init = dataset.init_cnn().to(args.device)
        else:
            raise TypeError('Unknown model: {}'.format(args.model))

        net_init.train()

        client_models: dict[int, ModelWrapper] = {i: ModelWrapper(model=copy.deepcopy(net_init))
                                                  for i in range(args.num_users)}
        training_status = {
            'epoch': 0,
            'total_epoch': epochs,
            'epoch_progress': 0.0,
            'acc_trains': [],
            'loss_trains': []
        }

        for current_epoch in range(args.epochs):
            # 开始本轮训练
            training_status['epoch'] = current_epoch + 1
            training_status['epoch_progress'] = 0.0
            status_dict[uuid] = {
                'status': 'TRAINING',
                'data': training_status
            }

            # 计数器
            epoch_progress_counter, epoch_steps = 0, args.num_users * 2 + args.num_servers

            communicate_queue = Queue()
            terminate_signal = Value('i', 0)

            mpc_thread = Process(target=mpc_threading,
                                 args=(args, current_epoch, dataset.train_data,
                                       {i: client.clone_to_device(torch.device('cpu'))
                                        for i, client in client_models.items()}, communicate_queue,
                                       terminate_signal))
            mpc_thread.start()

            returned_count = 0
            while returned_count < 1:
                try:
                    msg = communicate_queue.get()
                    if msg[0] == 'mpc':
                        epoch_progress_counter += msg[1]
                        training_status['epoch_progress'] = epoch_progress_counter / epoch_steps * 100
                        status_dict[uuid] = {
                            'status': 'TRAINING',
                            'data': training_status
                        }
                    elif msg[0] == 'mpc_r':
                        client_models_p = msg[1]
                        client_models = {i: client.clone_to_device(args.device)
                                         for i, client in client_models_p.items()}
                        returned_count += 1
                except KeyboardInterrupt:
                    break

            terminate_signal.value = 1

            # 本轮结束，衡量准确度
            status_dict[uuid] = {
                'status': 'EVALUATING',
                'data': None
            }
            mpc_thread.join()

            mpc_acc, mpc_loss = test_acc(args, dataset.train_data, client_models[0].model)
            training_status['acc_trains'].append(mpc_acc / 100.0)
            training_status['loss_trains'].append(mpc_loss)

        # plot curve
        plt_path = f"{model_output_folder}/{uuid}_metrics.png"
        plt.figure()
        fig, ax1 = plt.subplots(figsize=(10, 6))
        color_acc = 'tab:blue'
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Accuracy', color=color_acc)
        line_acc, = ax1.plot(training_status['acc_trains'], color=color_acc, marker='o', label='Accuracy')
        ax1.tick_params(axis='y', labelcolor=color_acc)
        ax2 = ax1.twinx()
        color_loss = 'tab:red'
        ax2.set_ylabel('Loss', color=color_loss)
        line_loss, = ax2.plot(training_status['loss_trains'], color=color_loss, linestyle='--', marker='x', label='Loss')
        ax2.tick_params(axis='y', labelcolor=color_loss)
        lines = [line_acc, line_loss]
        labels = [line.get_label() for line in lines]
        ax1.legend(lines, labels, loc='upper left')
        plt.title('Training Metrics')
        fig.tight_layout()
        plt.grid(True, alpha=0.3)
        plt.savefig(plt_path)

        # 训练结束，测试准度
        status_dict[uuid] = {
            'status': 'EVALUATING',
            'data': None
        }
        client_models[0].model.eval()
        mpc_acc_train, _ = test_acc(args, dataset.train_data, client_models[0].model)
        mpc_acc_test, _ = test_acc(args, dataset.test_data, client_models[0].model)

        model_path = f"{model_output_folder}/{uuid}.pth"
        torch.save({
            'dataset': dataset_name,
            'state_dict': client_models[0].model.state_dict()
        }, model_path)

        status_dict[uuid] = {
            'status': 'FINISHED',
            'data': {
                'train_acc': mpc_acc_train,
                'test_acc': mpc_acc_test
            }
        }

    except Exception as e:
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        status_dict[uuid] = {
            'status': 'FAILED',
            'data': error_msg
        }


def check_classify_acc(model_path: str, img_path: str, acc_dict: DictProxy):
    """
    获取模型对给定图片的分类

    Args:
        model_path: 模型所在路径，包含文件名
        img_path: 图片所在路径，包含文件名
        acc_dict: 接受返回值至 result 字段，为包含不同类型的可能性的字典

    """
    args = get_default_args()
    args.device = torch.device(
        'cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    checkpoint: dict = torch.load(model_path, map_location=args.device)
    dataset: DatasetWrapper = available_models[checkpoint['dataset']](args)
    model = dataset.init_cnn().to(args.device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    test_img = Image.open(img_path)
    test_tensor = dataset.transform_test_img(test_img)

    with torch.no_grad():
        outputs = model(test_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]

    class_names = dataset.get_classes()
    acc_dict['result'] = {class_names[i]: round(prob.item(), 4) for i, prob in enumerate(probabilities)}
