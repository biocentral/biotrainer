# https://discuss.pytorch.org/t/reproducibility-with-all-the-bells-and-whistles/81097
def seed_all(seed: int = 42):
    import random
    random.seed(seed)

    try:
        import numpy as np
        np.random.seed(seed)  # Also seeds sklearn
    except ImportError:
        pass

    try:
        import torch
        torch.manual_seed(seed)

        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
