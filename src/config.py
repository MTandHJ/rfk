







# Here are some basic settings.
# It could be overwritten if you want to specify
# some configs. However, please check the correspoding
# codes in loadopts.py.



import torchvision.transforms as T
import random
import logging
from PIL import ImageFilter
from .dict2obj import Config



class _GaussBlur:

    def __init__(self, sigma=(.1, 2.)):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x



ROOT = "../data" # the path saving the data
SAVED_FILENAME = "paras.pt" # the filename of saved model paramters
INFO_PATH = "./infos/{method}/{dataset}-{model}/{description}"
LOG_PATH = "./logs/{method}/{dataset}-{model}/{description}-{time}"
TIMEFMT = "%m%d%H"

# logger
LOGGER = Config(
    name='RFK', filename='log.txt', level=logging.INFO,
    formatter=Config(
        filehandler=logging.Formatter('%(asctime)s:\t%(message)s'),
        consolehandler=logging.Formatter('%(message)s')
    )
)

# for visualization
BASE_FTYPES = ('max', 'min', 'mean', 'norm1', 'norm2', 'norminf')
FTYPES = BASE_FTYPES
STATS_FILENAME = "model.stats"

TRANSFORMS = {
    "mnist": {
        'default': T.ToTensor()
    },
    "fashionmnist": {
        'default': T.ToTensor()
    },
    "cifar10": {
        'default': T.Compose((
            T.Pad(4, padding_mode='reflect'),
            T.RandomCrop(32),
            T.RandomHorizontalFlip(),
            T.ToTensor()
        )),
        'simclr': T.Compose((
            T.RandomResizedCrop(32, scale=(0.2, 1.0)),
            T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.RandomApply([_GaussBlur()], p=0.5),
            T.RandomHorizontalFlip(),
            T.ToTensor()
        ))
    },
    "cifar100": {
        'default': T.Compose((
            T.Pad(4, padding_mode='reflect'),
            T.RandomCrop(32),
            T.RandomHorizontalFlip(),
            T.ToTensor()
        )),
        'simclr': T.Compose((
            T.RandomResizedCrop(32, scale=(0.2, 1.0)),
            T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.RandomApply([_GaussBlur()], p=0.5),
            T.RandomHorizontalFlip(),
            T.ToTensor()
        ))
    }
}

VALIDER = {
    "mnist": (Config(attack_type="pgd-linf", stepsize=0.033333, steps=100), 0.3),
    "cifar10": (Config(attack_type="pgd-linf", stepsize=0.25, steps=10), 8/255),
    "cifar100": (Config(attack_type="pgd-linf", stepsize=0.25, steps=10), 8/255)
}

# env settings
NUM_WORKERS = 3
PIN_MEMORY = True

# basic properties of inputs
BOUNDS = (0, 1) # for fb.attacks.Attack
PREPROCESSING = None # for fb.attacks.Attack
MEANS = {
    "mnist": None,
    "fashionmnist": None,
    "cifar10": [0.4914, 0.4824, 0.4467],
    "cifar100": [0.5071, 0.4867, 0.4408]
}

STDS = {
    "mnist": None,
    "fashionmnist": None,
    "cifar10": [0.2471, 0.2435, 0.2617],
    "cifar100": [0.2675, 0.2565, 0.2761]
}

# the settings of optimizers of which lr could be pointed
# additionally.
OPTIMS = {
    "sgd": Config(lr=0.1, momentum=0.9, weight_decay=0.0005, nesterov=False, prefix="SGD:"),
    "adam": Config(lr=0.01, betas=(0.9, 0.999), weight_decay=0., prefix="Adam:")
}


# the learning schedular can be added here
LEARNING_POLICY = {
   "default": (
        "MultiStepLR",
        Config(
            milestones=[50, 75],
            gamma=0.1,
            prefix="Default leaning policy will be applied:"
        )
    ),
    "null": (
        "StepLR",
        Config(
            step_size=9999999999999,
            gamma=1,
            prefix="Null leaning policy will be applied:"
        )
    ),
    "STD": (
        "MultiStepLR",
        Config(
            milestones=[82, 123],
            gamma=0.1,
            prefix="STD leaning policy will be applied:"
        )
    ),
    "STD-wrn": (
        "MultiStepLR",
        Config(
            milestones=[60, 120, 160],
            gamma=0.2,
            prefix="STD-wrn leaning policy will be applied:"
        )
    ),
    "AT":(
        "MultiStepLR",
        Config(
            milestones=[102, 154],
            gamma=0.1,
            prefix="AT learning policy, an official config:"
        )
    ),
    "TRADES":(
        "MultiStepLR",
        Config(
            milestones=[75, 90, 100],
            gamma=0.1,
            prefix="TRADES learning policy, an official config:"
        )
    ),
    "TRADES-M":(
        "MultiStepLR",
        Config(
            milestones=[55, 75, 90],
            gamma=0.1,
            prefix="TRADES learning policy, an official config for MNIST:"
        )
    ),
    "cosine":(   
        "CosineAnnealingLR",   
        Config(          
            T_max=200,
            eta_min=0.,
            last_epoch=-1,
            prefix="cosine learning policy: T_max == epochs - 1:"
        )
    )
}






