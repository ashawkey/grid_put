import math
import torch
import torch.nn.functional as F
from grid_put import grid_put

import kiui
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='random')
parser.add_argument('--ratio', type=float, default=1.0)
args = parser.parse_args()

source = kiui.read_image("assets/img.png", mode="tensor").cuda()  # [H, W, C]

H, W, C = source.shape

if args.mode == 'random':
    # random coords
    coords = torch.rand(int(H * W * args.ratio), 2, device="cuda") * 2 - 1  # [N, 2] in [-1, 1]
else:
    # grid coords
    coords = torch.stack(torch.meshgrid(torch.linspace(-1, 1, steps=int(H * math.sqrt(args.ratio))), torch.linspace(-1, 1, steps=int(W * math.sqrt(args.ratio))), indexing='ij'), dim=-1).view(-1, 2).cuda()

# grid sample
values = (
    F.grid_sample(
        source.permute(2, 0, 1).unsqueeze(0).contiguous(),
        coords.view(1, 1, -1, 2)[..., [1, 0]],
        mode="bilinear",
        align_corners=False,
    )
    .view(C, -1)
    .permute(1, 0)
    .contiguous()
)

# grid put
for mode in ['nearest', 'bilinear', 'bilinear-mipmap']:
    out = grid_put((H, W), coords, values, mode=mode)
    kiui.write_image(f"assets/out_{mode}_{args.ratio}_{args.mode}.png", out)