import argparse
import os


def args_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    default_args = get_default_args()
    # federated arguments
    parser.add_argument('--epochs', type=int, default=default_args.epochs,
                        help="rounds of training")
    parser.add_argument('--num_users', type=int, default=default_args.num_users,
                        help="number of users: N")
    parser.add_argument('--num_servers', type=int, default=default_args.num_servers,
                        help="number of server: M")
    parser.add_argument('--num_recover', type=int, default=default_args.num_recover,
                        help="number of server to recover secret: T")
    parser.add_argument('--local_ep', type=int, default=default_args.local_ep,
                        help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=default_args.local_bs,
                        help="local batch size: B")
    parser.add_argument('--bs', type=int, default=default_args.bs,
                        help="test batch size")
    parser.add_argument('--lr', type=float, default=default_args.lr,
                        help="learning rate")
    parser.add_argument('--momentum', type=float, default=default_args.momentum,
                        help="SGD momentum (default: 0.5)")
    parser.add_argument('--scale', type=int, default=default_args.scale,
                        help="scale in shamir")
    parser.add_argument('--prime', type=int, default=default_args.prime,
                        help="modulo in shamir")

    # model arguments
    parser.add_argument('--model', type=str, default=default_args.model,
                        help='model name')

    # other arguments
    parser.add_argument('--dataset', type=str, default=default_args.dataset,
                        help="name of dataset")
    parser.add_argument('--gpu', type=int, default=default_args.gpu,
                        help="GPU ID, -1 for CPU")
    parser.add_argument('--dataset_path', type=str, default=default_args.dataset_path,
                        help="Path to save datasets")
    args = parser.parse_args()
    return args


def get_default_args() -> argparse.Namespace:
    return argparse.Namespace(
        epochs=10,
        num_users=100,
        num_servers=10,
        num_recover=9,
        local_ep=5,
        local_bs=10,
        bs=128,
        lr=0.01,
        momentum=0.5,
        scale=10**6,
        prime=2**31-1,
        model='cnn',
        dataset='mnist',
        gpu=0,
        dataset_path=os.path.join('..', 'data')
    )
