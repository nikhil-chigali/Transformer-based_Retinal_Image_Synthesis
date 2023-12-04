import matplotlib.pyplot as plt

import torchvision.transforms as transforms
from torchvision.utils import make_grid


def visualize_batch(images):
    # num_images = images.size(0)
    images = images * 0.5 + 0.5
    inv_transform = transforms.ToPILImage()
    grid = make_grid(images)
    grid = inv_transform(grid)
    plt.imshow(grid)
    plt.xticks([])
    plt.yticks([])
    plt.show()
    # num_cols = 4
    # num_rows = (num_images - 1) // num_cols + 1
    # fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 10))

    # axes = axes.flatten()
    # for i, ax in enumerate(axes):
    #     img = images[i]
    #     img = inv_transform(img)
    #     ax.imshow(img)
    #     ax.axis("off")
    # plt.tight_layout()
    # plt.show()
