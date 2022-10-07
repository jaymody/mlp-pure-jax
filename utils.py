def read_mnist_csv(csv_filepath):
    import numpy as np

    with open(csv_filepath, "r") as fi:
        images = []
        labels = []
        for line in fi:
            nums = list(map(int, line.split(",")))
            labels.append(nums[0])
            images.append(nums[1:])

    return np.array(images), np.array(labels)


def set_torch_seed(seed):
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
