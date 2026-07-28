"""Native 6D LitePT variant.

This fork keeps the LitePT encoder/decoder shape, but removes the 3D sparse
convolution path. The model's primary coordinates are 6D phase-space voxel
coordinates, not auxiliary features.
"""

from collections import OrderedDict

import flash_attn
import torch
import torch.nn as nn
import torch_scatter
from torch.utils.checkpoint import checkpoint
from addict import Dict
from timm.layers import DropPath

from .serialization import encode


@torch.no_grad()
def offset2bincount(offset):
    return torch.diff(
        offset,
        prepend=torch.tensor([0], device=offset.device, dtype=torch.long),
    )


@torch.no_grad()
def offset2batch(offset):
    bincount = offset2bincount(offset)
    return torch.arange(
        len(bincount), device=offset.device, dtype=torch.long
    ).repeat_interleave(bincount)


@torch.no_grad()
def batch2offset(batch):
    return torch.cumsum(batch.bincount(), dim=0).long()


class Point(Dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if "batch" not in self.keys() and "offset" in self.keys():
            self["batch"] = offset2batch(self.offset)
        elif "offset" not in self.keys() and "batch" in self.keys():
            self["offset"] = batch2offset(self.batch)

    @property
    def coord_dim(self):
        if "grid_coord" in self.keys():
            return int(self.grid_coord.shape[1])
        if "coord" in self.keys():
            return int(self.coord.shape[1])
        raise KeyError("Point requires coord or grid_coord")

    def serialization(self, order=("z",), depth=None, shuffle_orders=False):
        self["order"] = order
        assert "batch" in self.keys()
        if "grid_coord" not in self.keys():
            assert {"grid_size", "coord"}.issubset(self.keys())
            self["grid_coord"] = torch.div(
                self.coord - self.coord.min(0)[0],
                self.grid_size,
                rounding_mode="trunc",
            ).int()
        if torch.any(self.grid_coord < 0):
            raise ValueError("grid_coord must be non-negative")

        ndim = int(self.grid_coord.shape[1])
        if depth is None:
            max_coord = int(self.grid_coord.max().item()) if self.grid_coord.numel() else 0
            depth = max(1, int(max_coord + 1).bit_length())
        batch_bits = max(1, len(self.offset).bit_length())
        if depth * ndim + batch_bits > 63:
            raise ValueError(
                f"Serialization budget exceeded: depth={depth}, ndim={ndim}, "
                f"batch_bits={batch_bits}"
            )
        self["serialized_depth"] = int(depth)
        self["serialized_ndim"] = ndim

        orders = (order,) if isinstance(order, str) else tuple(order)
        code = torch.stack([
            encode(self.grid_coord, self.batch, depth=depth, order=order_)
            for order_ in orders
        ])
        sorted_order = torch.argsort(code, dim=1)
        inverse = torch.zeros_like(sorted_order).scatter_(
            dim=1,
            index=sorted_order,
            src=torch.arange(0, code.shape[1], device=code.device).repeat(
                code.shape[0], 1
            ),
        )
        if shuffle_orders:
            perm = torch.randperm(code.shape[0], device=code.device)
            code = code[perm]
            sorted_order = sorted_order[perm]
            inverse = inverse[perm]
        self["serialized_code"] = code
        self["serialized_order"] = sorted_order
        self["serialized_inverse"] = inverse

    def sparsify(self, *_, **__):
        self["native_sparse"] = True
        return self

    @torch.no_grad()
    def get_padding_and_inverse(self, patch_size):
        pad_key = "pad"
        unpad_key = "unpad"
        cu_key = "cu_seqlens_key"
        if pad_key in self.keys() and unpad_key in self.keys() and cu_key in self.keys():
            return self[pad_key], self[unpad_key], self[cu_key]

        offset = self.offset
        bincount = offset2bincount(offset)
        bincount_pad = (
            torch.div(
                bincount + patch_size - 1,
                patch_size,
                rounding_mode="trunc",
            )
            * patch_size
        )
        mask_pad = bincount > patch_size
        bincount_pad = (~mask_pad) * bincount + mask_pad * bincount_pad
        _offset = nn.functional.pad(offset, (1, 0))
        _offset_pad = nn.functional.pad(torch.cumsum(bincount_pad, dim=0), (1, 0))
        pad = torch.arange(_offset_pad[-1], device=offset.device)
        unpad = torch.arange(_offset[-1], device=offset.device)
        cu_seqlens = []
        for i in range(len(offset)):
            unpad[_offset[i] : _offset[i + 1]] += _offset_pad[i] - _offset[i]
            if bincount[i] != bincount_pad[i]:
                pad[
                    _offset_pad[i + 1]
                    - patch_size
                    + (bincount[i] % patch_size) : _offset_pad[i + 1]
                ] = pad[
                    _offset_pad[i + 1]
                    - 2 * patch_size
                    + (bincount[i] % patch_size) : _offset_pad[i + 1]
                    - patch_size
                ]
            pad[_offset_pad[i] : _offset_pad[i + 1]] -= _offset_pad[i] - _offset[i]
            cu_seqlens.append(
                torch.arange(
                    _offset_pad[i],
                    _offset_pad[i + 1],
                    step=patch_size,
                    dtype=torch.int32,
                    device=offset.device,
                )
            )
        self[pad_key] = pad
        self[unpad_key] = unpad
        self[cu_key] = nn.functional.pad(
            torch.concat(cu_seqlens), (0, 1), value=_offset_pad[-1]
        )
        return self[pad_key], self[unpad_key], self[cu_key]


class PointModule(nn.Module):
    pass


def _point_with_replaced_feat(point, feat):
    copied = Point()
    for key, value in point.items():
        copied[key] = value
    copied["feat"] = feat
    return copied


class PointSequential(PointModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)
        for name, module in kwargs.items():
            if name in self._modules:
                raise ValueError("name exists")
            self.add_module(name, module)

    def __len__(self):
        return len(self._modules)

    def add(self, module, name=None):
        if name is None:
            name = str(len(self._modules))
        self.add_module(name, module)

    def forward(self, input):
        for module in self._modules.values():
            if isinstance(module, PointModule):
                input = module(input)
            elif isinstance(input, Point):
                input.feat = module(input.feat)
            else:
                input = module(input)
        return input


class SerializedAttention(PointModule):
    def __init__(
        self,
        channels,
        num_heads,
        patch_size,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        order_index=0,
    ):
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError("channels must be divisible by num_heads")
        self.channels = channels
        self.num_heads = num_heads
        self.patch_size = int(patch_size)
        self.scale = qk_scale or (channels // num_heads) ** -0.5
        self.attn_drop = float(attn_drop)
        self.order_index = int(order_index)
        self.qkv = nn.Linear(channels, channels * 3, bias=qkv_bias)
        self.proj = nn.Linear(channels, channels)
        self.proj_drop = nn.Dropout(proj_drop)

    def _fallback_attention(self, qkv, cu_seqlens):
        H = self.num_heads
        C = self.channels
        out = torch.empty((qkv.shape[0], C), device=qkv.device, dtype=qkv.dtype)
        for start, end in zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist()):
            block = qkv[start:end]
            q, k, v = block[:, 0], block[:, 1], block[:, 2]
            q = q.transpose(0, 1)
            k = k.transpose(0, 1)
            v = v.transpose(0, 1)
            attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            attn = torch.softmax(attn, dim=-1)
            y = torch.matmul(attn, v).transpose(0, 1).reshape(-1, C)
            out[start:end] = y
        return out

    def forward(self, point):
        H = self.num_heads
        C = self.channels
        pad, unpad, cu_seqlens = point.get_padding_and_inverse(self.patch_size)
        order_bank = point.serialized_order
        inv_bank = point.serialized_inverse
        order_idx = self.order_index % order_bank.shape[0]
        order = order_bank[order_idx][pad]
        inverse = unpad[inv_bank[order_idx]]

        qkv_raw = self.qkv(point.feat)[order]
        qkv = qkv_raw.reshape(-1, 3, H, C // H)
        use_flash = qkv.is_cuda and qkv.dtype in (torch.float16, torch.bfloat16)
        if use_flash:
            feat = flash_attn.flash_attn_varlen_qkvpacked_func(
                qkv,
                cu_seqlens,
                max_seqlen=self.patch_size,
                dropout_p=self.attn_drop if self.training else 0.0,
                softmax_scale=self.scale,
            ).reshape(-1, C)
        elif qkv.is_cuda:
            qkv_half = qkv.to(torch.float16)
            feat = flash_attn.flash_attn_varlen_qkvpacked_func(
                qkv_half,
                cu_seqlens,
                max_seqlen=self.patch_size,
                dropout_p=self.attn_drop if self.training else 0.0,
                softmax_scale=self.scale,
            ).reshape(-1, C).to(qkv_raw.dtype)
        else:
            feat = self._fallback_attention(qkv, cu_seqlens)
        feat = feat[inverse]
        point.feat = self.proj_drop(self.proj(feat))
        return point


class GridPooling(PointModule):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=2,
        norm_layer=None,
        act_layer=None,
        reduce="max",
        shuffle_orders=True,
        traceable=True,
        re_serialization=False,
        serialization_order=("z",),
    ):
        super().__init__()
        if reduce not in ("sum", "mean", "min", "max"):
            raise ValueError("reduce must be sum, mean, min, or max")
        self.stride = int(stride)
        self.reduce = reduce
        self.shuffle_orders = shuffle_orders
        self.traceable = traceable
        self.re_serialization = re_serialization
        self.serialization_order = serialization_order
        self.proj = nn.Linear(in_channels, out_channels)
        self.norm = PointSequential(norm_layer(out_channels)) if norm_layer is not None else None
        self.act = PointSequential(act_layer()) if act_layer is not None else None

    def forward(self, point):
        grid_coord = point.grid_coord
        grid_coord = torch.div(grid_coord, self.stride, rounding_mode="trunc")
        key = torch.cat([point.batch.view(-1, 1), grid_coord.long()], dim=1)
        unique_key, cluster, counts = torch.unique(
            key,
            sorted=True,
            return_inverse=True,
            return_counts=True,
            dim=0,
        )
        _, indices = torch.sort(cluster)
        idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])
        head_indices = indices[idx_ptr[:-1]]
        point_dict = Dict(
            feat=torch_scatter.segment_csr(
                self.proj(point.feat)[indices], idx_ptr, reduce=self.reduce
            ),
            coord=torch_scatter.segment_csr(
                point.coord[indices], idx_ptr, reduce="mean"
            ),
            grid_coord=unique_key[:, 1:].int(),
            batch=unique_key[:, 0].long(),
        )
        if "origin_coord" in point.keys():
            point_dict["origin_coord"] = torch_scatter.segment_csr(
                point.origin_coord[indices], idx_ptr, reduce="mean"
            )
        if "grid_size" in point.keys():
            point_dict["grid_size"] = point.grid_size * self.stride
        if self.traceable:
            point_dict["pooling_inverse"] = cluster
            point_dict["pooling_parent"] = point
        point = Point(point_dict)
        if self.norm is not None:
            point = self.norm(point)
        if self.act is not None:
            point = self.act(point)
        if self.re_serialization:
            point.serialization(
                order=self.serialization_order,
                shuffle_orders=self.shuffle_orders,
            )
        return point


class GridUnpooling(PointModule):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        norm_layer=None,
        act_layer=None,
        traceable=False,
    ):
        super().__init__()
        self.proj = PointSequential(nn.Linear(in_channels, out_channels))
        self.proj_skip = PointSequential(nn.Linear(skip_channels, out_channels))
        if norm_layer is not None:
            self.proj.add(norm_layer(out_channels))
            self.proj_skip.add(norm_layer(out_channels))
        if act_layer is not None:
            self.proj.add(act_layer())
            self.proj_skip.add(act_layer())
        self.traceable = traceable

    def forward(self, point):
        parent = point.pop("pooling_parent")
        inverse = point.pooling_inverse
        feat = point.feat
        parent = self.proj_skip(parent)
        parent.feat = parent.feat + self.proj(point).feat[inverse]
        if self.traceable:
            point.feat = feat
            parent["unpooling_parent"] = point
            parent["unpooling_inverse"] = inverse
        return parent


class Embedding(PointModule):
    def __init__(self, in_channels, embed_channels, norm_layer=None, act_layer=None, **_):
        super().__init__()
        self.stem = PointSequential(nn.Linear(in_channels, embed_channels, bias=False))
        if norm_layer is not None:
            self.stem.add(norm_layer(embed_channels), name="norm")
        if act_layer is not None:
            self.stem.add(act_layer(), name="act")

    def forward(self, point):
        return self.stem(point)


class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels=None, out_channels=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_channels, out_channels)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


class Block(PointModule):
    def __init__(
        self,
        channels,
        num_heads,
        patch_size=512,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        pre_norm=True,
        order_index=0,
        enable_conv=False,
        enable_attn=True,
        activation_checkpointing=False,
        **_,
    ):
        super().__init__()
        self.pre_norm = pre_norm
        self.enable_conv = bool(enable_conv)
        self.enable_attn = bool(enable_attn)
        self.activation_checkpointing = bool(activation_checkpointing)
        if self.enable_conv:
            self.conv = PointSequential(
                nn.Linear(channels, channels),
                norm_layer(channels),
            )
        else:
            self.norm0 = PointSequential(norm_layer(channels))
        if self.enable_attn:
            self.norm1 = PointSequential(norm_layer(channels))
            self.attn = SerializedAttention(
                channels=channels,
                patch_size=patch_size,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                order_index=order_index,
            )
            self.norm2 = PointSequential(norm_layer(channels))
            self.mlp = PointSequential(
                MLP(
                    in_channels=channels,
                    hidden_channels=int(channels * mlp_ratio),
                    out_channels=channels,
                    act_layer=act_layer,
                    drop=proj_drop,
                )
            )
            self.drop_path = PointSequential(
                DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
            )

    def _forward_impl(self, point):
        if self.enable_conv:
            shortcut = point.feat
            point = self.conv(point)
            point.feat = shortcut + point.feat
        else:
            point = self.norm0(point)
        if self.enable_attn:
            shortcut = point.feat
            if self.pre_norm:
                point = self.norm1(point)
            point = self.drop_path(self.attn(point))
            point.feat = shortcut + point.feat
            if not self.pre_norm:
                point = self.norm1(point)

            shortcut = point.feat
            if self.pre_norm:
                point = self.norm2(point)
            point = self.drop_path(self.mlp(point))
            point.feat = shortcut + point.feat
            if not self.pre_norm:
                point = self.norm2(point)
        return point

    def _forward_checkpointed(self, point):
        if self.enable_attn:
            point.get_padding_and_inverse(self.attn.patch_size)

        def run(feat):
            copied = _point_with_replaced_feat(point, feat)
            return self._forward_impl(copied).feat

        point.feat = checkpoint(run, point.feat, use_reentrant=False)
        return point

    def forward(self, point):
        if (
            self.activation_checkpointing
            and self.training
            and torch.is_grad_enabled()
            and point.feat.requires_grad
        ):
            return self._forward_checkpointed(point)
        return self._forward_impl(point)


class LitePT(PointModule):
    def __init__(
        self,
        in_channels=4,
        order=("z", "z-rev", "z-evenodd", "z-oddeven"),
        stride=(2, 2, 2),
        enc_depths=(2, 2, 2, 2),
        enc_channels=(48, 96, 192, 384),
        enc_num_head=(3, 6, 12, 16),
        enc_patch_size=(512, 512, 512, 512),
        enc_conv=(False, False, False, False),
        enc_attn=(True, True, True, True),
        enc_rope_freq=None,
        dec_depths=(0, 0, 0),
        dec_channels=(64, 96, 192),
        dec_num_head=(4, 6, 12),
        dec_patch_size=(512, 512, 512),
        dec_conv=(False, False, False),
        dec_attn=(False, False, False),
        dec_rope_freq=None,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.1,
        pre_norm=True,
        shuffle_orders=True,
        enc_mode=False,
        coord_dim=6,
        activation_checkpointing=False,
        **_,
    ):
        super().__init__()
        self.num_stages = len(enc_depths)
        self.order = (order,) if isinstance(order, str) else tuple(order)
        self.enc_mode = enc_mode
        self.shuffle_orders = shuffle_orders
        self.coord_dim = int(coord_dim)
        self.activation_checkpointing = bool(activation_checkpointing)
        self.enc_attn = tuple(bool(x) for x in enc_attn)

        assert self.num_stages == len(stride) + 1
        assert self.num_stages == len(enc_channels)
        assert self.num_stages == len(enc_num_head)
        assert self.num_stages == len(enc_patch_size)
        assert self.enc_mode or self.num_stages == len(dec_depths) + 1
        assert self.enc_mode or self.num_stages == len(dec_channels) + 1

        bn_layer = lambda channels: nn.BatchNorm1d(channels, eps=1e-3, momentum=0.01)
        ln_layer = nn.LayerNorm
        act_layer = nn.GELU

        self.embedding = Embedding(
            in_channels=in_channels,
            embed_channels=enc_channels[0],
            norm_layer=bn_layer,
            act_layer=act_layer,
        )

        enc_drop_path = [x.item() for x in torch.linspace(0, drop_path, sum(enc_depths))]
        self.enc = PointSequential()
        for s in range(self.num_stages):
            enc_drop_path_ = enc_drop_path[
                sum(enc_depths[:s]) : sum(enc_depths[: s + 1])
            ]
            enc = PointSequential()
            if s > 0:
                enc.add(
                    GridPooling(
                        in_channels=enc_channels[s - 1],
                        out_channels=enc_channels[s],
                        stride=stride[s - 1],
                        norm_layer=bn_layer,
                        act_layer=act_layer,
                        re_serialization=self.enc_attn[s],
                        serialization_order=self.order,
                    ),
                    name="down",
                )
            for i in range(enc_depths[s]):
                enc.add(
                    Block(
                        channels=enc_channels[s],
                        num_heads=enc_num_head[s],
                        patch_size=enc_patch_size[s],
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        attn_drop=attn_drop,
                        proj_drop=proj_drop,
                        drop_path=enc_drop_path_[i],
                        norm_layer=ln_layer,
                        act_layer=act_layer,
                        pre_norm=pre_norm,
                        order_index=i % len(self.order),
                        enable_conv=enc_conv[s],
                        enable_attn=enc_attn[s],
                        activation_checkpointing=self.activation_checkpointing,
                    ),
                    name=f"block{i}",
                )
            self.enc.add(module=enc, name=f"enc{s}")

        if not self.enc_mode:
            dec_drop_path = [x.item() for x in torch.linspace(0, drop_path, sum(dec_depths))]
            self.dec = PointSequential()
            dec_channels_full = list(dec_channels) + [enc_channels[-1]]
            for s in reversed(range(self.num_stages - 1)):
                dec_drop_path_ = dec_drop_path[
                    sum(dec_depths[:s]) : sum(dec_depths[: s + 1])
                ]
                dec_drop_path_.reverse()
                dec = PointSequential()
                dec.add(
                    GridUnpooling(
                        in_channels=dec_channels_full[s + 1],
                        skip_channels=enc_channels[s],
                        out_channels=dec_channels_full[s],
                        norm_layer=bn_layer,
                        act_layer=act_layer,
                    ),
                    name="up",
                )
                for i in range(dec_depths[s]):
                    dec.add(
                        Block(
                            channels=dec_channels_full[s],
                            num_heads=dec_num_head[s],
                            patch_size=dec_patch_size[s],
                            mlp_ratio=mlp_ratio,
                            qkv_bias=qkv_bias,
                            qk_scale=qk_scale,
                            attn_drop=attn_drop,
                            proj_drop=proj_drop,
                            drop_path=dec_drop_path_[i],
                            norm_layer=ln_layer,
                            act_layer=act_layer,
                            pre_norm=pre_norm,
                            order_index=i % len(self.order),
                            enable_conv=dec_conv[s],
                            enable_attn=dec_attn[s],
                            activation_checkpointing=self.activation_checkpointing,
                        ),
                        name=f"block{i}",
                    )
                self.dec.add(module=dec, name=f"dec{s}")

    def forward(self, data_dict):
        point = Point(data_dict)
        if point.coord_dim != self.coord_dim:
            raise ValueError(
                f"Native LitePT expected {self.coord_dim}D coordinates, "
                f"got {point.coord_dim}D"
            )
        point.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
        point = self.embedding(point)
        point = self.enc(point)
        if not self.enc_mode:
            point = self.dec(point)
        return point
