import numpy as np
import torch
from matplotlib import pyplot as plt


def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape).astype(np.float32), k=1)
    return torch.from_numpy(1 - subsequent_mask).float()


if __name__ == '__main__':
    mask = subsequent_mask(10)
    plt.figure(figsize=(12, 8))
    plt.imshow(mask[0], aspect='equal', cmap='viridis')
    plt.colorbar(label='mask value')
    plt.title("mask")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
