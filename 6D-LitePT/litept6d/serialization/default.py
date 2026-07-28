import torch


def _permute_dims(grid_coord, order):
    ndim = int(grid_coord.shape[1])
    if order in ("z", "z6"):
        perm = tuple(range(ndim))
    elif order in ("z-rev", "z6-rev"):
        perm = tuple(reversed(range(ndim)))
    elif order in ("z-evenodd", "z6-evenodd"):
        perm = tuple(list(range(0, ndim, 2)) + list(range(1, ndim, 2)))
    elif order in ("z-oddeven", "z6-oddeven"):
        perm = tuple(list(range(1, ndim, 2)) + list(range(0, ndim, 2)))
    else:
        raise ValueError(f"Unsupported native serialization order {order!r}")
    return grid_coord[:, perm]


@torch.inference_mode()
def encode(grid_coord, batch=None, depth=None, order="z"):
    grid_coord = _permute_dims(grid_coord.long(), order)
    if grid_coord.ndim != 2:
        raise ValueError("grid_coord must be [N, D]")
    if torch.any(grid_coord < 0):
        raise ValueError("grid_coord must be non-negative before serialization")
    ndim = int(grid_coord.shape[1])
    if depth is None:
        max_coord = int(grid_coord.max().item()) if grid_coord.numel() else 0
        depth = max(1, int(max_coord + 1).bit_length())
    if depth * ndim > 62:
        raise ValueError(
            f"Cannot encode {ndim}D coords at depth={depth}; "
            f"needs {depth * ndim} position bits"
        )

    code = torch.zeros(grid_coord.shape[0], device=grid_coord.device, dtype=torch.long)
    for bit in range(int(depth)):
        for dim in range(ndim):
            code |= ((grid_coord[:, dim] >> bit) & 1) << (bit * ndim + dim)

    if batch is not None:
        batch = batch.long()
        code = batch << (int(depth) * ndim) | code
    return code
