import numpy as np
from matplotlib import pyplot as plt


def image(data):
    plt.figure(figsize=(10, 10))
    plt.imshow(data)
    plt.colorbar()
    plt.show()


def plot(x, y):
    fig, axes = plt.figure(figsize=(10, 10))
    plt.plot(x, y, marker='o', color='red')
    axes.grid(True)
    plt.show()


if __name__ == '__main__':
    n1 = 262144
    n2 = 13
    nx = 512
    nz = 512
    f_vel = "/var/tmp/tccs/lfd/twod/vel7_t.rsf@"
    f_coe = "/var/tmp/tccs/lfd/twod/G_10.rsf@"
    vel = np.fromfile(f_vel, dtype=np.float32).reshape((1, n1))
    coeff = np.fromfile(f_coe, dtype=np.float32).reshape((13, n1))
    plot(vel[0], coeff[0])
