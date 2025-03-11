import unittest
from argparse import Namespace

import torch
from torch import Tensor

from security.shamir import _shamir_split, recover_sharing
from util.options import args_parser


class Shamir(unittest.TestCase):
    def test_shamir(self):
        args = args_parser()
        p = torch.randn(10).cuda()
        print(p)
        splits = _shamir_split(args, p)
        print(splits)
        recovered = recover_sharing(args, splits)
        print(recovered)


if __name__ == '__main__':
    unittest.main()
