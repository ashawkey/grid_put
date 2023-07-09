import torch
import torch.nn.functional as F

def stride_from_shape(shape):
    stride = [1]
    for x in reversed(shape[1:]):
        stride.append(stride[-1] * x) 
    return list(reversed(stride))


def scatter_add_nd(input, indices, values):
    # input: [..., C], D dimension + C channel
    # indices: [N, D], long
    # values: [N, C]

    D = indices.shape[-1]
    C = input.shape[-1]
    size = input.shape[:-1]
    stride = stride_from_shape(size)

    assert len(size) == D

    input = input.view(-1, C)  # [HW, C]
    flatten_indices = (indices * torch.tensor(stride, dtype=torch.long, device=indices.device)).sum(-1)  # [N]

    input.scatter_add_(0, flatten_indices.unsqueeze(1).repeat(1, C), values)

    return input.view(*size, C)


def nearest_grid_put_2d(H, W, coords, values):
    # coords: [N, 2], float in [-1, 1]
    # values: [N, C]

    C = values.shape[-1]

    indices = (coords * 0.5 + 0.5) * torch.tensor(
        [H - 1, W - 1], dtype=torch.float32, device=coords.device
    )
    indices = indices.round().long()  # [N, 2]

    result = torch.zeros(H, W, C, device=values.device, dtype=values.dtype)  # [H, W, C]
    count = torch.zeros(H, W, 1, device=values.device, dtype=values.dtype)  # [H, W, 1]
    ones = torch.ones_like(values[..., :1])  # [N, 1]
    result = scatter_add_nd(result, indices, values)
    count = scatter_add_nd(count, indices, ones)

    mask = (count.squeeze(-1) > 0)
    result[mask] = result[mask] / count[mask].repeat(1, C)

    return result


def bilinear_grid_put_2d(H, W, coords, values, return_count=False):
    # coords: [N, 2], float in [-1, 1]
    # values: [N, C]

    C = values.shape[-1]

    indices = (coords * 0.5 + 0.5) * torch.tensor(
        [H - 1, W - 1], dtype=torch.float32, device=coords.device
    )
    indices_00 = indices.floor().long()  # [N, 2]
    indices_00[:, 0].clamp_(max=H - 2)
    indices_00[:, 1].clamp_(max=W - 2)
    indices_01 = indices_00 + torch.tensor(
        [0, 1], dtype=torch.long, device=indices.device
    )
    indices_10 = indices_00 + torch.tensor(
        [1, 0], dtype=torch.long, device=indices.device
    )
    indices_11 = indices_00 + torch.tensor(
        [1, 1], dtype=torch.long, device=indices.device
    )

    h = indices[..., 0] - indices_00[..., 0].float()
    w = indices[..., 1] - indices_00[..., 1].float()
    w_00 = (1 - h) * (1 - w)
    w_01 = (1 - h) * w
    w_10 = h * (1 - w)
    w_11 = h * w

    result = torch.zeros(H, W, C, device=values.device, dtype=values.dtype)  # [H, W, C]
    count = torch.zeros(H, W, 1, device=values.device, dtype=values.dtype)  # [H, W, 1]
    ones = torch.ones_like(values[..., :1])  # [N, 1]
    result = scatter_add_nd(result, indices_00, values * w_00.unsqueeze(1))
    count = scatter_add_nd(count, indices_00, ones * w_00.unsqueeze(1))
    result = scatter_add_nd(result, indices_01, values * w_01.unsqueeze(1))
    count = scatter_add_nd(count, indices_01, ones * w_01.unsqueeze(1))
    result = scatter_add_nd(result, indices_10, values * w_10.unsqueeze(1))
    count = scatter_add_nd(count, indices_10, ones * w_10.unsqueeze(1))
    result = scatter_add_nd(result, indices_11, values * w_11.unsqueeze(1))
    count = scatter_add_nd(count, indices_11, ones * w_11.unsqueeze(1))

    if return_count:
        return result, count

    mask = (count.squeeze(-1) > 0)
    result[mask] = result[mask] / count[mask].repeat(1, C)

    return result


def mipmap_bilinear_grid_put_2d(H, W, coords, values, min_resolution=32):
    # coords: [N, 2], float in [-1, 1]
    # values: [N, C]

    C = values.shape[-1]

    result = torch.zeros(H, W, C, device=values.device, dtype=values.dtype)  # [H, W, C]
    count = torch.zeros(H, W, 1, device=values.device, dtype=values.dtype)  # [H, W, 1]

    cur_H, cur_W = H, W
    
    while min(cur_H, cur_W) > min_resolution:

        # try to fill the holes
        mask = (count.squeeze(-1) == 0)
        if not mask.any():
            break

        cur_result, cur_count = bilinear_grid_put_2d(cur_H, cur_W, coords, values, return_count=True)
        result[mask] = result[mask] + F.interpolate(cur_result.permute(2,0,1).unsqueeze(0).contiguous(), (H, W), mode='bilinear', align_corners=False).squeeze(0).permute(1,2,0).contiguous()[mask]
        count[mask] = count[mask] + F.interpolate(cur_count.view(1, 1, cur_H, cur_W), (H, W), mode='bilinear', align_corners=False).view(H, W, 1)[mask]
        cur_H //= 2
        cur_W //= 2
    
    mask = (count.squeeze(-1) > 0)
    result[mask] = result[mask] / count[mask].repeat(1, C)

    return result


def grid_put(shape, coords, values, mode='bilinear-mipmap', min_resolution=32):
    # shape: [D], list/tuple
    # coords: [N, D], float in [-1, 1]
    # values: [N, C]

    D = len(shape)
    assert D == 2, f'only support D == 2, but got D == {D}'

    if mode == 'nearest':
        out = nearest_grid_put_2d(*shape, coords, values)
    elif mode == 'bilinear':
        out = bilinear_grid_put_2d(*shape, coords, values)
    elif mode == 'bilinear-mipmap':
        out = mipmap_bilinear_grid_put_2d(*shape, coords, values, min_resolution=min_resolution)
    else:
        raise NotImplementedError(f"got mode {mode}")
    
    return out
