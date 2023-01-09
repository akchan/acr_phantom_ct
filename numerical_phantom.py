#!/usr/bin/env python
# coding: UTF-8


'''
Reference
=========

ACR phantom (Gamma 464)

https://accreditationsupport.acr.org/support/solutions/articles/11000053945-phantom-overview-ct-revised-11-9-2022-
'''


import numpy as np
import skimage


def gen_module2(nx=512, fov=300):
    img = gen_module2_seed(nx, fov).astype(np.float32)

    img[img == 0] = -1000.0
    img[img >= 2] = 96.0
    img[img == 1] = 90.0

    return img


def gen_module2_seed(nx=512, fov=300.0):
    '''
    Generate a seed image for low contrast resolution (module2)
    '''
    r1 = 100.0
    cylinder_list = [
        # [diameter, n, theta, val, r]
        [4,  4,  45, 60, 2.0],
        [6,  4,   0, 60, 3.0],
        [8,  4, 300, 60, 4.0],
        [10, 4, 230, 60, 5.0],
        [12, 4, 150, 55, 6.0],
        [25, 1,  90, 60, 7.0],
    ]

    img = np.zeros([nx, nx])
    _, ny = img.shape

    center = [nx//2, ny//2]
    body_diameter = nx * 2.0 * r1 / fov
    rr, cc = skimage.draw.disk(center, body_diameter/2.0)

    img[rr, cc] = 1.0

    for diameter, n, theta, r, val in cylinder_list:
        r = float(r)
        img = draw_dots(img, diameter / fov * nx, n, theta+180.0, r / fov * nx, val)

    return img


def draw_dots(img, diameter, n, theta, r, val=1.0):
    nx, ny = img.shape
    rad = theta / 180.0 * np.pi

    px_left = - 2*diameter*(n - 1)/2
    py_left = -r

    for i in range(n):
        px_tmp = px_left + i * 2 * diameter
        py_tmp = py_left

        px = px_tmp * np.cos(rad) - py_tmp * np.sin(rad) + nx//2
        py = px_tmp * np.sin(rad) + py_tmp * np.cos(rad) + ny//2

        center = [px, py]

        rr, cc = skimage.draw.disk(center, diameter/2.0)

        img[rr, cc] = val

    return img


def main():
    import plotly.express as px
    
    px.imshow(gen_module2(), color_continuous_scale='jet', zmin=0.0).show()


if __name__ == '__main__':
    main()
