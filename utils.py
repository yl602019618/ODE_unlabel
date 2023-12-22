import numpy as np
import torch
import random
import os
# Linear, damped harmonic oscillator
def linear_damped_SHO(t, x):
    return [-0.1 * x[0] + 2 * x[1], -2 * x[0] - 0.1 * x[1]]





def set_seed_everywhere(seed):
    """
    Set seed for all randomness sources
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)