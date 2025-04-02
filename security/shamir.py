from argparse import Namespace

import cupy as cp
import torch
from torch import nn, Tensor


def flatten_weight(grad: dict[str, Tensor]) -> Tensor:
    """展平为向量"""
    grad_list = [param_grad.view(-1) for _, param_grad in grad.items()]
    return torch.cat(grad_list)


def recover_weight_shape(flatten: Tensor, refer_model: nn.Module) -> dict[str, Tensor]:
    """恢复维度信息"""
    weight: dict[str, Tensor] = {}
    pointer = 0
    for name, param in refer_model.named_parameters():
        num_elements = param.numel()
        shape = param.shape  # 当前层的原始形状
        layer_weight = flatten[pointer: pointer + num_elements].view(shape)  # 提取对应段的权重并重塑
        weight[name] = layer_weight.clone()
        pointer += num_elements
    return weight


def split_sharing(args: Namespace, model: nn.Module) -> dict[int, cp.ndarray]:
    """生成 Shamir 分片，返回 {接收方ID: 分片}"""
    float_secret = flatten_weight(model.state_dict())
    return _shamir_split(args, float_secret)


def _shamir_split(args: Namespace, float_secret: Tensor) -> dict[int, cp.ndarray]:
    secret = (cp.from_dlpack((float_secret * args.scale).__dlpack__()).astype(cp.int64) % args.prime)
    model_dim = secret.shape[0]
    coefficients = cp.zeros((args.num_recover, model_dim), dtype=cp.int64)
    coefficients[0] = secret % args.prime
    coefficients[1:] = cp.random.randint(0, args.prime, (args.num_recover - 1, model_dim))
    # 矩阵乘法加速计算
    x = cp.arange(1, args.num_servers + 1).reshape(-1, 1)
    powers = cp.arange(args.num_recover).reshape(1, -1)
    x_powers = cp.power(x, powers)
    shares = (x_powers @ coefficients) % args.prime
    return {client_id: shares[client_id] for client_id in range(args.num_servers)}


def recover_sharing(args: Namespace, shares: dict[int, cp.ndarray]) -> Tensor:
    """
    从至少 t 个服务器的聚合分片重构全局模型，使用向量化方法优化计算。
    对恢复后的模 p 数值做了中心化处理，即如果值大于 args.prime // 2，则认为它在原本表示中为负数。
    """
    t = args.num_recover

    share_keys = list(shares.keys())[:t]
    x = cp.array([k + 1 for k in share_keys], dtype=cp.int64)
    # 堆叠 t 个分片，形成形状 (t, model_dim) 的矩阵
    y = cp.vstack([shares[k] for k in share_keys])

    # 预计算每个 x_i 对应的拉格朗日系数 lg_i，
    # 这里 lg_i(0) = ∏_{j!=i} (-x_j/(x_i - x_j)) mod prime
    lg = cp.empty(t, dtype=cp.int64)
    for i in range(t):
        mask = cp.arange(t) != i
        numerator = cp.prod(-x[mask]) % args.prime
        denominator = cp.prod(x[i] - x[mask]) % args.prime
        inv_denominator = pow(int(denominator), -1, args.prime)
        lg[i] = (numerator * inv_denominator) % args.prime

    # 利用向量化计算 f(0) = ∑ lg[i] * y[i] mod prime
    secret_int = cp.mod(cp.sum(lg[:, cp.newaxis] * y, axis=0), args.prime)

    # 中心化：假设原秘密绝对值小于 prime/2，那么若 secret_int > prime/2
    # 则应认为它是负数表示，即 secret_int - prime
    half_prime = args.prime // 2
    secret_signed = cp.where(secret_int > half_prime, secret_int - args.prime, secret_int)

    float_secret = secret_signed.astype(cp.float64) / args.scale
    return torch.from_dlpack(float_secret.__dlpack__())
