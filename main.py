import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.misc

# TODO: usar tensor board (o simil)


def loss(input, target):
    return np.sum((input - target) ** 2)


def generate_multi_canvas(num_triangles, target):
    return np.zeros([num_triangles] + list(target.shape))


def triangle_area(ax, ay, bx, by, cx, cy):
    # print(f"==== {bx!r} {ax!r}")
    abx = bx - ax
    aby = by - ay
    acx = cx - ax
    acy = cy - ay
    cp = abx * acy - aby * acx
    return np.abs(cp) / 2


def inside_triangle(px, py, ax, ay, bx, by, cx, cy):
    orig = triangle_area(ax, ay, bx, by, cx, cy)
    p1 = triangle_area(px, py, bx, by, cx, cy)
    p2 = triangle_area(ax, ay, px, py, cx, cy)
    p3 = triangle_area(ax, ay, bx, by, px, py)
    epsilon = 1e-8
    return np.abs((orig - (p1 + p2 + p3))) < epsilon


def render_triangles(triangles, target):
    x_size, y_size = target.shape[:2]
    cx, cy, ct = np.meshgrid(np.arange(0, x_size), np.arange(0, y_size), np.arange(triangles.shape[0]), indexing="ij")
    triangles_canvas = triangles[:, -4:].T
    t = triangles[:, :6]
    inside = inside_triangle(cx, cy, t[:, 0], t[:, 1], t[:, 2], t[:, 3], t[:, 4], t[:, 5]).reshape(x_size, y_size, 1,
                                                                                                   -1) #FIXME: FEO!!!
    rendered = triangles_canvas * inside
    return rendered


def blend_triangles(multi_canvas):
    xz, yz, colors, triangles = multi_canvas.shape
    canvas = np.zeros((xz, yz, 3))
    for i in range(triangles):
        triangle_canvas = multi_canvas[:,:,:,i]
        alpha = triangle_canvas[:,:,3].reshape(xz,yz,1)
        canvas = canvas * (1 - alpha) + triangle_canvas[:,:,:3] * alpha
    return canvas


def generate_triangles(num_triangles, x_size, y_size):
    triangles = np.random.rand(num_triangles, 10)
    triangle_limits = np.array([x_size, y_size, x_size, y_size, x_size, y_size, 1, 1, 1, 1])
    triangles_limits = np.tile(triangle_limits, num_triangles).reshape(num_triangles, 10)
    return triangles * triangles_limits


def render(triangles, target):
    rend = render_triangles(triangles, target)
    blend = blend_triangles(rend)
    return blend


def mutate(triangles, num_mutations):
    ps, cs = 30.0, .01
    scale = np.array([ps] * 6 + [cs] * 4)
    for i in range(num_mutations):
        yield triangles + np.random.randn(*triangles.shape) * scale


def run():
    # [(ax,ay,bx,by,cx,cy,r,g,b,a)]

    target = sp.misc.face()
    x_size, y_size = target.shape[:2]
    NUM_TRIANGLES = 10

    triangles = generate_triangles(NUM_TRIANGLES, x_size, y_size)
    current = triangles

    while True:
        best = None
        best_error = float("inf")

        for mutation in mutate(current, 100):
            candidate = render(mutation, target)
            error = loss(candidate, target)
            if error < best_error:
                best_error = error
                best = mutation

        current = best
        print(best_error)


if __name__ == "__main__":
    run()