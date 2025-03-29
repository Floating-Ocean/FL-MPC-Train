import abc
from argparse import Namespace

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets

from model.cnn import CNN
from util.sampling import split_iid_data


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxes):
        self.dataset = dataset
        self.idxes = list(idxes)

    def __len__(self):
        return len(self.idxes)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxes[item]]
        return image, label


def split_data(args: Namespace, indices: dict[int, set],
               data: DataLoader, is_train: bool = True) -> list[DataLoader]:
    return [DataLoader(DatasetSplit(data.dataset, idxes),
                       batch_size=args.local_bs if is_train else args.bs,
                       shuffle=is_train)
            for (_, idxes) in indices.items()]


class DatasetWrapper(abc.ABC):
    train_data: list[DataLoader]
    test_data: list[DataLoader]
    _args: Namespace
    _train_dataloader: DataLoader
    _test_dataloader: DataLoader

    def __init__(self, args: Namespace):
        super().__init__()
        self._args = args

    def _init_data(self):
        train_data_indices = split_iid_data(self._args, self._train_dataloader)
        test_data_indices = split_iid_data(self._args, self._test_dataloader)
        self.train_data = split_data(self._args, train_data_indices, self._train_dataloader)
        self.test_data = split_data(self._args, test_data_indices, self._test_dataloader, False)


    @abc.abstractmethod
    def init_cnn(self) -> CNN:
        pass


class Mnist(DatasetWrapper):

    def __init__(self, args: Namespace):
        super().__init__(args)
        trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        origin_train_data = datasets.MNIST('../data/', train=True, download=True, transform=trans)
        origin_test_data = datasets.MNIST('../data/', train=False, download=True, transform=trans)
        self._train_dataloader, self._test_dataloader = DataLoader(origin_train_data), DataLoader(origin_test_data)
        self._init_data()


    def init_cnn(self):
        return CNN(
            in_channels=1,
            num_classes=10
        )


class EMnist(DatasetWrapper):

    def __init__(self, args: Namespace):
        super().__init__(args)
        trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1736,), (0.3317,))])
        origin_train_data = datasets.EMNIST('../data/', split='byclass', train=True, download=True, transform=trans)
        origin_test_data = datasets.EMNIST('../data/', split='byclass', train=False, download=True, transform=trans)
        # EMNIST需要做图像转置（原始数据是转置存储的）
        origin_train_data.data = origin_train_data.data.permute(0, 2, 1)
        origin_test_data.data = origin_test_data.data.permute(0, 2, 1)
        self._train_dataloader, self._test_dataloader = DataLoader(origin_train_data), DataLoader(origin_test_data)
        self._init_data()


    def init_cnn(self):
        return CNN(
            in_channels=1,
            num_classes=62
        )


class QMnist(DatasetWrapper):

    def __init__(self, args: Namespace):
        super().__init__(args)
        trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        origin_train_data = datasets.QMNIST('../data/', train=True, download=True, transform=trans)
        origin_test_data = datasets.QMNIST('../data/', train=False, download=True, transform=trans)
        self._train_dataloader, self._test_dataloader = DataLoader(origin_train_data), DataLoader(origin_test_data)
        self._init_data()


    def init_cnn(self):
        return CNN(
            in_channels=1,
            num_classes=10
        )


class KMnist(DatasetWrapper):

    def __init__(self, args: Namespace):
        super().__init__(args)
        trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1904,), (0.3475,))])
        origin_train_data = datasets.KMNIST('../data/', train=True, download=True, transform=trans)
        origin_test_data = datasets.KMNIST('../data/', train=False, download=True, transform=trans)
        self._train_dataloader, self._test_dataloader = DataLoader(origin_train_data), DataLoader(origin_test_data)
        self._init_data()


    def init_cnn(self):
        return CNN(
            in_channels=1,
            num_classes=10
        )


class FashionMnist(DatasetWrapper):

    def __init__(self, args: Namespace):
        super().__init__(args)
        trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))])
        origin_train_data = datasets.FashionMNIST('../data/', train=True, download=True, transform=trans)
        origin_test_data = datasets.FashionMNIST('../data/', train=False, download=True, transform=trans)
        self._train_dataloader, self._test_dataloader = DataLoader(origin_train_data), DataLoader(origin_test_data)
        self._init_data()


    def init_cnn(self):
        return CNN(
            in_channels=1,
            num_classes=10
        )


class Cifar10(DatasetWrapper):

    def __init__(self, args: Namespace):
        super().__init__(args)
        train_trans = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
        test_trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
        origin_train_data = datasets.CIFAR10('../data/CIFAR10/', train=True, download=True, transform=train_trans)
        origin_test_data = datasets.CIFAR10('../data/CIFAR10/', train=False, download=True, transform=test_trans)
        self._train_dataloader, self._test_dataloader = DataLoader(origin_train_data), DataLoader(origin_test_data)
        self._init_data()


    def init_cnn(self):
        return CNN(
            in_channels=3,
            num_classes=10,
            img_size=32
        )


class Cifar100(DatasetWrapper):

    def __init__(self, args: Namespace):
        super().__init__(args)
        train_trans = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        test_trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        origin_train_data = datasets.CIFAR100('../data/CIFAR100/', train=True, download=True, transform=train_trans)
        origin_test_data = datasets.CIFAR100('../data/CIFAR100/', train=False, download=True, transform=test_trans)
        self._train_dataloader, self._test_dataloader = DataLoader(origin_train_data), DataLoader(origin_test_data)
        self._init_data()


    def init_cnn(self):
        return CNN(
            in_channels=3,
            num_classes=100,
            img_size=32
        )


class STL10(DatasetWrapper):

    def __init__(self, args: Namespace):
        super().__init__(args)
        trans = transforms.Compose([
            transforms.Resize(96),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        origin_train_data = datasets.STL10('../data/STL10/', split='train', download=True, transform=trans)
        origin_test_data = datasets.STL10('../data/STL10/', split='test', download=True, transform=trans)
        self._train_dataloader, self._test_dataloader = DataLoader(origin_train_data), DataLoader(origin_test_data)
        self._init_data()


    def init_cnn(self):
        return CNN(
            in_channels=3,
            num_classes=10,
            img_size=96
        )


class SVHN(DatasetWrapper):

    def __init__(self, args: Namespace):
        super().__init__(args)
        trans = transforms.Compose([
            transforms.Resize(96),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        origin_train_data = datasets.SVHN('../data/SVHN/', split='train', download=True, transform=trans)
        origin_test_data = datasets.SVHN('../data/SVHN/', split='test', download=True, transform=trans)
        self._train_dataloader, self._test_dataloader = DataLoader(origin_train_data), DataLoader(origin_test_data)
        self._init_data()


    def init_cnn(self):
        return CNN(
            in_channels=3,
            num_classes=10,
            img_size=32
        )


class CelebA(DatasetWrapper):

    def __init__(self, args: Namespace):
        super().__init__(args)
        trans = transforms.Compose([
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
        ])
        origin_train_data = datasets.CelebA('../data/CELEBA/', split='train', download=True, transform=trans)
        origin_test_data = datasets.CelebA('../data/CELEBA/', split='test', download=True, transform=trans)
        self._train_dataloader, self._test_dataloader = DataLoader(origin_train_data), DataLoader(origin_test_data)
        self._init_data()


    def init_cnn(self):
        return CNN(
            in_channels=3,
            num_classes=40,
            img_size=64
        )


available_models = {
    'mnist': Mnist,
    'emnist': EMnist,
    'qmnist': QMnist,
    'kmnist': KMnist,
    'fashionmnist': FashionMnist,
    'cifar10': Cifar10,
    'cifar100': Cifar100,
    'stl10': STL10,
    'svhn': SVHN,
    'celeba': CelebA,
}
