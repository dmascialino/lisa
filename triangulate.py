import torch
import torch.cuda
import matplotlib.pyplot as plt
import scipy as sp
import cv2
from rich.console import Console
import einops
import numpy as np

# TODO
# fix render triangles to support 2-d meshgrid and subsampling
# add CLI
# add test to learn ONE triangle as target
# add weights and bias https://wandb.ai/
# upload this to github :)
# run experiments and see...

# setting device on GPU if available, else CPU
print(torch.cuda.is_available(), torch.cuda.get_device_name(0))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print()

#Additional Info when using cuda
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')


DEVICE = "cuda"


def blend_triangles(multi_canvas):
    # multi_canvas.shape == num_triangle, (sample), colors
    shape = list(multi_canvas.shape)
    canvas = torch.zeros(shape[1:-1] + [3])
    canvas = canvas.to(DEVICE)
    for i in range(shape[0]):
        triangle_canvas = multi_canvas[i]
        alpha = triangle_canvas[:,:,3:4]
        triangle_canvas = triangle_canvas[:,:,:3]
        canvas = canvas * (1 - alpha) + triangle_canvas[:,:,:3] * alpha
    return canvas



def random_triangles(num_triangles, x_size, y_size):
    positions = torch.rand(num_triangles, 6)
    position_limits = torch.tensor([x_size, y_size, x_size, y_size, x_size, y_size])
    position_limits = torch.tile(position_limits, (num_triangles,)).reshape(num_triangles, 6)
    positions = positions * position_limits
    positions = positions.to(DEVICE)
    positions = positions.to(torch.float64)

    colors = torch.rand(num_triangles, 4)
    color_limits = torch.tensor([1, 1, 1, 1])
    color_limits = torch.tile(color_limits, (num_triangles,)).reshape(num_triangles, 4)
    colors = colors * color_limits
    colors = colors.to(DEVICE)
    colors = colors.to(torch.float64)


    return torch.tensor(positions, requires_grad=True), torch.tensor(colors, requires_grad=True)


def build_triangles(positions, colors):
    return torch.cat((positions, colors), dim=1)


def render_multicanvas(triangles, x_grid, y_grid):
    triangles_canvas = triangles[:, -4:]
    t = triangles[:, :6]
    xg = einops.rearrange(x_grid, "... -> ... 1")
    yg = einops.rearrange(y_grid, "... -> ... 1")
    inside = inside_triangle(xg, yg, t[:, 0], t[:, 1], t[:, 2], t[:, 3], t[:, 4], t[:, 5])
    # [num_t, color] * [sample, num_t] -> [sample, num_t, color]
    multi_canvas = triangles_canvas * einops.rearrange(inside, "... -> ... 1")

    return einops.rearrange(multi_canvas, "... t c -> t ... c")


def render_triangles(triangles, x_grid, y_grid):
    multi_canvas = render_multicanvas(triangles, x_grid, y_grid)
    canvas = blend_triangles(multi_canvas)

    return canvas


def triangle_area(ax, ay, bx, by, cx, cy):
    abx = bx - ax
    aby = by - ay
    acx = cx - ax
    acy = cy - ay
    cp = abx * acy - aby * acx
    return torch.abs(cp) / 2


def inside_triangle(px, py, ax, ay, bx, by, cx, cy):
    orig = triangle_area(ax, ay, bx, by, cx, cy)
    p1 = triangle_area(px, py, bx, by, cx, cy)
    p2 = triangle_area(ax, ay, px, py, cx, cy)
    p3 = triangle_area(ax, ay, bx, by, px, py)

    area_diff = (p1 + p2 + p3) - orig
    return torch.special.expit((1 - area_diff) * 5)


def generate_meshgrid(x_size, y_size):
  cx, cy = torch.meshgrid(torch.arange(0, x_size), torch.arange(0, y_size), indexing="ij")
  return cx.to(DEVICE), cy.to(DEVICE)


def sample_meshgrid(x_size, y_size, points):
  cx = torch.randint(0, x_size, (points, 1))
  cy = torch.randint(0, y_size, (points, 1))
  return cx.to(DEVICE), cy.to(DEVICE)


def show(render):
  plt.imshow(render.cpu().detach().numpy())


def save(render, name=None):
  if name is None:
      name = "output.png"
  plt.imsave(name, np.clip(render.cpu().detach().numpy(), 0, 1))


def test_one_triangle():
    triangles = torch.tensor([
        [0, 0, 100, 0, 0, 100, 1, 1, 1, 1],
    ]).to(DEVICE)
    x_grid, y_grid = generate_meshgrid(100, 100)
    generated = render_multicanvas(triangles, x_grid, y_grid)[0,:,:,:3].detach().cpu().numpy()
    target = plt.imread("white_triangle.png")[:, :, :3]
    assert np.allclose(generated, target, rtol=0.01)


def test_one_triangle_blend():
    triangles = torch.tensor([
        [0, 0, 100, 0, 0, 100, 1, 1, 1, 1],
        [0, 0, 50, 0, 0, 50, 0, 1, 0, 1]
    ]).to(DEVICE)
    x_grid, y_grid = generate_meshgrid(100, 100)
    generated = render_triangles(triangles, x_grid, y_grid)[:,:,:3].detach().cpu().numpy()
    target = plt.imread("green_white_triangle.png")[:, :, :3]
    assert np.allclose(generated, target, rtol=0.01, atol=0.01)


def learn(scale, num_triangles, img):
    width = int(img.shape[1] * scale)
    height = int(img.shape[0] * scale)
    dim = (width, height)

    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

    if img.dtype == np.uint8:
        img = img / 255.

    target = torch.Tensor(img).to(DEVICE)

    x_size, y_size = target.shape[:2]

    x_grid_full, y_grid_full = generate_meshgrid(x_size, y_size)
    positions, colors = random_triangles(num_triangles, x_size, y_size)

    optimizer = torch.optim.SGD([
        {'params': positions, 'lr': 0.01},
        {'params': colors, 'lr': 0.00001}
    ])


    console = Console()
    save(target[x_grid_full, y_grid_full] , "target1.png")

    try:
        with console.status("Booting") as status:
            for t in range(2000000):
                # Forward pass: Compute predicted y by passing x to the model
                triangles = build_triangles(positions, colors)

                x_grid, y_grid = sample_meshgrid(x_size, y_size, 200)
                x_grid = x_grid_full
                y_grid = y_grid_full
                candidate = render_triangles(triangles, x_grid, y_grid)

                # Compute and print loss
                loss = torch.sum((target[x_grid, y_grid] - candidate) ** 2)
                loss_value = loss.item()
                # Zero gradients, perform a backward pass, and update the weights.
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if t % 100 == 0:
                    candidate = render_triangles(triangles, x_grid_full, y_grid_full)
                    save(candidate)
                    print(loss_value)
                    status.update(f"Iteration: {t} Loss: {loss_value}")
                    optimizer.zero_grad()

    except KeyboardInterrupt:
        save(candidate)
        save(target[x_grid, y_grid] , "target.png")


import fire

class Trainer(object):
    """A simple calculator class."""

    def learn(self, scale, num_triangles, target_image=None):
        if target_image is None:
            target = sp.misc.face()
        else:
            target = plt.imread(target_image)[:, :, :3]
        learn(scale, num_triangles, target)


if __name__ == '__main__':
  fire.Fire(Trainer)