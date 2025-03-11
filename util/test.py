from argparse import Namespace

import torch
from torch import nn
from torch.utils.data import DataLoader


def test_acc(args: Namespace, test_data: list[DataLoader], model: nn.Module) -> tuple[float, float]:
    """测试模型准确度"""
    loss, acc = 0, 0

    for m_i in range(args.num_users):
        model.eval()
        current_losses, current_n_c = 0, 0
        for _, (data, target) in enumerate(test_data[m_i]):
            if args.gpu != -1:
                data, target = data.cuda(), target.cuda()
            y_logit = model(data)
            current_losses += torch.nn.CrossEntropyLoss()(y_logit, target).item()
            current_n_c += n_correct(y_logit, target)

        loss += current_losses / len(test_data[m_i].dataset)  # 根据数据集大小进行加权平均
        acc += current_n_c / len(test_data[m_i].dataset)

    loss /= args.num_users
    acc /= args.num_users

    return acc * 100.0, loss


def n_correct(y_logit, y):
    _, predicted = torch.max(y_logit.data, 1)
    correct = (predicted == y).sum().item()
    return correct
