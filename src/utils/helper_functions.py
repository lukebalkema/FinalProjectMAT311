import random
import numpy as np

def set_random_seed(seed=123):
    random.seed(seed)
    np.random.seed(seed)
