import abc
import os
from argparse import Namespace

from PIL.Image import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from torchvision.transforms import Compose

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
    _classes: list[str]
    _args: Namespace
    _train_dataloader: DataLoader
    _test_dataloader: DataLoader
    _test_transformer: Compose

    def __init__(self, args: Namespace):
        super().__init__()
        self._args = args

    def _init_data(self):
        train_data_indices = split_iid_data(self._args, self._train_dataloader)
        test_data_indices = split_iid_data(self._args, self._test_dataloader)
        self.train_data = split_data(self._args, train_data_indices, self._train_dataloader)
        self.test_data = split_data(self._args, test_data_indices, self._test_dataloader, False)
        
    def get_classes(self):
        return self._classes

    def init_cnn(self) -> CNN:
        return CNN(**self.cnn_kwargs())

    @abc.abstractmethod
    def cnn_kwargs(self) -> dict:
        pass

    def transform_test_img(self, img: Image) -> Tensor:
        return self._test_transformer(img).unsqueeze(0).to(self._args.device)


class Mnist(DatasetWrapper):

    def __init__(self, args: Namespace):
        super().__init__(args)
        trans = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize(28),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        origin_train_data = datasets.MNIST('../data/', train=True, download=True, transform=trans)
        origin_test_data = datasets.MNIST('../data/', train=False, download=True, transform=trans)
        self._train_dataloader, self._test_dataloader = DataLoader(origin_train_data), DataLoader(origin_test_data)
        self._classes = origin_train_data.classes
        self._test_transformer = trans
        self._init_data()


    def cnn_kwargs(self) -> dict:
        return {
            'in_channels': 1,
            'num_classes': 10,
            'img_size': 28
        }


class EMnist(DatasetWrapper):

    def __init__(self, args: Namespace):
        super().__init__(args)
        trans = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize(28),
            transforms.ToTensor(),
            transforms.Normalize((0.1736,), (0.3317,))
        ])
        origin_train_data = datasets.EMNIST('../data/', split='byclass', train=True, download=True, transform=trans)
        origin_test_data = datasets.EMNIST('../data/', split='byclass', train=False, download=True, transform=trans)
        # EMNIST需要做图像转置（原始数据是转置存储的）
        origin_train_data.data = origin_train_data.data.permute(0, 2, 1)
        origin_test_data.data = origin_test_data.data.permute(0, 2, 1)
        self._train_dataloader, self._test_dataloader = DataLoader(origin_train_data), DataLoader(origin_test_data)
        self._classes = origin_train_data.classes
        self._test_transformer = trans
        self._init_data()


    def cnn_kwargs(self) -> dict:
        return {
            'in_channels': 1,
            'num_classes': 62,
            'img_size': 28
        }


class QMnist(DatasetWrapper):

    def __init__(self, args: Namespace):
        super().__init__(args)
        trans = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize(28),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        origin_train_data = datasets.QMNIST('../data/', train=True, download=True, transform=trans)
        origin_test_data = datasets.QMNIST('../data/', train=False, download=True, transform=trans)
        self._train_dataloader, self._test_dataloader = DataLoader(origin_train_data), DataLoader(origin_test_data)
        self._classes = origin_train_data.classes
        self._test_transformer = trans
        self._init_data()


    def cnn_kwargs(self) -> dict:
        return {
            'in_channels': 1,
            'num_classes': 10,
            'img_size': 28
        }


class KMnist(DatasetWrapper):

    def __init__(self, args: Namespace):
        super().__init__(args)
        trans = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize(28),
            transforms.ToTensor(),
            transforms.Normalize((0.1904,), (0.3475,))
        ])
        origin_train_data = datasets.KMNIST('../data/', train=True, download=True, transform=trans)
        origin_test_data = datasets.KMNIST('../data/', train=False, download=True, transform=trans)
        self._train_dataloader, self._test_dataloader = DataLoader(origin_train_data), DataLoader(origin_test_data)
        self._classes = origin_train_data.classes
        self._test_transformer = trans
        self._init_data()


    def cnn_kwargs(self) -> dict:
        return {
            'in_channels': 1,
            'num_classes': 10,
            'img_size': 28
        }


class FashionMnist(DatasetWrapper):

    def __init__(self, args: Namespace):
        super().__init__(args)
        trans = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize(28),
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))
        ])
        origin_train_data = datasets.FashionMNIST('../data/', train=True, download=True, transform=trans)
        origin_test_data = datasets.FashionMNIST('../data/', train=False, download=True, transform=trans)
        self._train_dataloader, self._test_dataloader = DataLoader(origin_train_data), DataLoader(origin_test_data)
        self._classes = origin_train_data.classes
        self._test_transformer = trans
        self._init_data()


    def cnn_kwargs(self) -> dict:
        return {
            'in_channels': 1,
            'num_classes': 10,
            'img_size': 28
        }


class Cifar10(DatasetWrapper):

    def __init__(self, args: Namespace):
        super().__init__(args)
        train_trans = transforms.Compose([
            transforms.Lambda(lambda img: img.convert("RGB")),
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
        self._classes = origin_train_data.classes
        self._test_transformer = test_trans
        self._init_data()


    def cnn_kwargs(self) -> dict:
        return {
            'in_channels': 3,
            'num_classes': 10,
            'img_size': 32
        }


class Cifar100(DatasetWrapper):

    def __init__(self, args: Namespace):
        super().__init__(args)
        train_trans = transforms.Compose([
            transforms.Lambda(lambda img: img.convert("RGB")),
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
        self._classes = origin_train_data.classes
        self._test_transformer = test_trans
        self._init_data()


    def cnn_kwargs(self) -> dict:
        return {
            'in_channels': 3,
            'num_classes': 100,
            'img_size': 32
        }


class STL10(DatasetWrapper):

    def __init__(self, args: Namespace):
        super().__init__(args)
        trans = transforms.Compose([
            transforms.Lambda(lambda img: img.convert("RGB")),
            transforms.Resize(96),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        origin_train_data = datasets.STL10('../data/STL10/', split='train', download=True, transform=trans)
        origin_test_data = datasets.STL10('../data/STL10/', split='test', download=True, transform=trans)
        self._train_dataloader, self._test_dataloader = DataLoader(origin_train_data), DataLoader(origin_test_data)
        self._classes = origin_train_data.classes
        self._test_transformer = trans
        self._init_data()


    def cnn_kwargs(self) -> dict:
        return {
            'in_channels': 3,
            'num_classes': 10,
            'img_size': 96
        }


class SVHN(DatasetWrapper):

    def __init__(self, args: Namespace):
        super().__init__(args)
        trans = transforms.Compose([
            transforms.Lambda(lambda img: img.convert("RGB")),
            transforms.Resize(96),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        origin_train_data = datasets.SVHN('../data/SVHN/', split='train', download=True, transform=trans)
        origin_test_data = datasets.SVHN('../data/SVHN/', split='test', download=True, transform=trans)
        self._train_dataloader, self._test_dataloader = DataLoader(origin_train_data), DataLoader(origin_test_data)
        self._classes = ['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four', '5 - five', 
                        '6 - six', '7 - seven', '8 - eight', '9 - nine']
        self._test_transformer = trans
        self._init_data()


    def cnn_kwargs(self) -> dict:
        return {
            'in_channels': 3,
            'num_classes': 10,
            'img_size': 32
        }


class CelebA(DatasetWrapper):

    def __init__(self, args: Namespace):
        super().__init__(args)
        trans = transforms.Compose([
            transforms.Lambda(lambda img: img.convert("RGB")),
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
        ])
        origin_train_data = datasets.CelebA('../data/CELEBA/', split='train', download=True, transform=trans)
        origin_test_data = datasets.CelebA('../data/CELEBA/', split='test', download=True, transform=trans)
        self._train_dataloader, self._test_dataloader = DataLoader(origin_train_data), DataLoader(origin_test_data)
        self._classes = self._get_celeba_attribute_names()
        self._test_transformer = trans
        self._init_data()

    @classmethod
    def _get_celeba_attribute_names(cls) -> list[str]:
        attr_file = os.path.join('../data/CELEBA/celeba/list_attr_celeba.txt')

        # 读取属性文件（第一行是标题行）
        with open(attr_file, 'r') as f:
            header = f.readline().strip()
            attribute_names = header.split()

        return attribute_names


    def cnn_kwargs(self) -> dict:
        return {
            'in_channels': 3,
            'num_classes': 40,
            'img_size': 64
        }


available_models = {
    'mnist': {
        'referer': Mnist,
        'intro': (
            "MNIST 数据集包含 60,000 张 28x28 的灰度手写数字图像，是计算机视觉的基准数据集。"
            "广泛用于图像分类算法的入门训练和验证。"
        )
    },
    'emnist': {
        'referer': EMnist,
        'intro': (
            "EMNIST 扩展了 MNIST，包含手写字母和数字的混合数据，提供更复杂的分类场景。"
            "支持包括 ByMerge 和 ByClass 在内的多种数据划分方式。"
        )
    },
    'qmnist': {
        'referer': QMnist,
        'intro': (
            "QMNIST 是 MNIST 的升级版本，保留了原始数据格式同时提供额外元信息。"
            "包含附加的笔画轨迹数据和测试集扩展，适合研究鲁棒性。"
        )
    },
    'kmnist': {
        'referer': KMnist,
        'intro': (
            "Kuzushiji-MNIST 包含古典日文平假名字符的现代变体图像。"
            "专为研究古籍数字化和跨文化字符识别设计。"
        )
    },
    'fashionmnist': {
        'referer': FashionMnist,
        'intro': (
            "Fashion-MNIST 用 10 类服装灰度图像替代原始数字，难度更高。"
            "常用于测试复杂特征提取模型的性能。"
        )
    },
    'cifar10': {
        'referer': Cifar10,
        'intro': (
            "CIFAR-10 包含 60,000 张 32x32 彩色图像，涵盖 10 个常见物体类别。"
            "适合测试轻量级模型的图像分类能力。"
        )
    },
    'cifar100': {
        'referer': Cifar100,
        'intro': (
            "CIFAR-100 在 CIFAR-10 基础上细化到 100 个精细类别，挑战细粒度分类。"
            "包含 20 个超类层级结构，支持层次化学习任务。"
        )
    },
    'stl10': {
        'referer': STL10,
        'intro': (
            "STL-10 提供 96x96 高分辨率图像，专为半监督学习设计。"
            "包含少量标注数据和大量无标签数据，模拟真实场景。"
        )
    },
    'svhn': {
        'referer': SVHN,
        'intro': (
            "Street View House Numbers 数据集来自谷歌街景门牌号照片。"
            "包含裁剪的数字区域图像和额外干扰数字，测试现实场景识别。"
        )
    },
    'celeba': {
        'referer': CelebA,
        'intro': (
            "CelebFaces Attributes 数据集包含 20 万张名人面部图像。"
            "提供 40 种属性标签，支持多任务学习和人脸特征分析。"
        )
    }
}
