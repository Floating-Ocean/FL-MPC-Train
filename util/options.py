import argparse


def args_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--epochs', type=int, default=10, help="rounds of training")
    parser.add_argument('--num_users', type=int, default=100, help="number of users: N")
    parser.add_argument('--num_servers', type=int, default=10, help="number of server: M")
    parser.add_argument('--num_recover', type=int, default=9, help="number of server to recover secret: T")
    parser.add_argument('--local_ep', type=int, default=5, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=10, help="local batch size: B")
    parser.add_argument('--bs', type=int, default=128, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")
    parser.add_argument('--scale', type=int, default=10**6, help="scale in shamir")
    parser.add_argument('--prime', type=int, default=2**31-1, help="modulo in shamir")

    # model arguments
    parser.add_argument('--model', type=str, default='mlp', help='model name')

    # other arguments
    parser.add_argument('--dataset', type=str, default='mnist', help="name of dataset")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    args = parser.parse_args()
    return args
