# Grid Put

An operation trying to do the opposite of `F.grid_sample()`.


### Install

Assume `torch` already installed.

```bash
pip install git+https://github.com/ashawkey/grid_put

# or locally
git clone https://github.com/ashawkey/grid_put
cd grid_put
pip install .
```

### Usage

```python
from grid_put import grid_put

H, W # target grid shape
coords # [N, 2], float grid coords in [-1, 1]
values # [N, C], values to put into grid

# mode: nearest, bilinear, bilinear-mipmap (default)
out = grid_put((H, W), coords, values, mode='bilinear-mipmap') # [H, W, C]
```

### Examples

```bash
# extra dependency: pip install kiui
python test.py --mode <random|grid> --ratio <float>
```

|mode-ratio | nearest | bilinear | bilinear-mipmap |
|:-:|:-:|:-:|:-:|
|grid-10%|![](assets/out_nearest_0.1_grid.png)  |  ![](assets/out_bilinear_0.1_grid.png) | ![](assets/out_bilinear-mipmap_0.1_grid.png) |
|grid-90%|![](assets/out_nearest_0.9_grid.png)  |  ![](assets/out_bilinear_0.9_grid.png) | ![](assets/out_bilinear-mipmap_0.9_grid.png) |
|random-10%|![](assets/out_nearest_0.1_random.png)  |  ![](assets/out_bilinear_0.1_random.png) | ![](assets/out_bilinear-mipmap_0.1_random.png) |
|random-90%|![](assets/out_nearest_0.9_random.png)  |  ![](assets/out_bilinear_0.9_random.png) | ![](assets/out_bilinear-mipmap_0.9_random.png) |