




from typing import Optional, Any, Union
import torch
import torch.nn as nn
import numpy as np
import random
import os
import sys
from .config import SAVED_FILENAME



class AverageMeter:

    def __init__(self, name: str, fmt: str = ".5f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self) -> None:
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0

    def update(self, val: float, n: int = 1, mode: str = "mean") -> None:
        self.val = val
        self.count += n
        if mode == "mean":
            self.sum += val * n
        elif mode == "sum":
            self.sum += val
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} Avg:{avg:{fmt}}"
        return fmtstr.format(**self.__dict__)


class ProgressMeter:
    def __init__(self, *meters: AverageMeter, prefix: str = ""):
        self.meters = list(meters)
        self.prefix = prefix

    def display(self, *, epoch: int = 8888) -> None:
        entries = [self.prefix + f"[Epoch: {epoch:<4d}]"]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def add(self, *meters: AverageMeter) -> None:
        self.meters += list(meters)

    def step(self) -> None:
        for meter in self.meters:
            meter.reset()

def gpu(*models: nn.Module) -> torch.device:

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for model in models:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        else:
            model.to(device)
    return device

def mkdirs(*paths: str) -> None:
    for path in paths:
        try:
            os.makedirs(path)
        except FileExistsError:
            pass

def readme(path: str, opts: "parser", mode: str = "w") -> None:
    """
    opts: the argparse
    """
    import time
    time_ = time.strftime("%Y-%m-%d-%H:%M:%S")
    filename = path + "/README.md"
    s = "- {0[0]}:  {0[1]}\n"
    info = "\n## {0}".format(time_)
    for item in opts._get_kwargs():
        info += s.format(item)
    with open(filename, mode, encoding="utf8") as fh:
        fh.write(info)

# load model's parameters
def load(
    model: nn.Module, 
    path: str, 
    device: torch.device,
    strict: bool = True, 
    except_key: Optional[str] = None
) -> None:

    filename = os.path.join(path, SAVED_FILENAME)
    if str(device) =="cpu":
        state_dict = torch.load(filename, map_location="cpu")
        
    else:
        state_dict = torch.load(filename)
    if except_key is not None:
        except_keys = list(filter(lambda key: except_key in key, state_dict.keys()))
        for key in except_keys:
            del state_dict[key]
    model.load_state_dict(state_dict, strict=strict)
    model.eval()

# save the checkpoint
def save_checkpoint(
    path: str, 
    model: nn.Module, 
    optimizer: torch.optim.optimizer.Optimizer, 
    lr_scheduler: "learning rate policy",
    epoch: int
) -> None:
    path = path + "/model-optim-lr_sch-epoch.tar"
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "epoch": epoch
        },
        path
    )

# load the checkpoint
def load_checkpoint(
    path: str, 
    model: nn.Module, 
    optimizer: torch.optim.optimizer.Optimizer, 
    lr_scheduler: "learning rate policy"
) -> int:
    path = path + "/model-optim-lr_sch-epoch.tar"
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    epoch = checkpoint['epoch']
    return epoch

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# caculate the lp distance along the dim you need,
# dim could be tuple or list containing multi dims.
def distance_lp(
    x: torch.Tensor, 
    y: torch.Tensor, 
    p: Union[int, float, str], 
    dim: Optional[int] = None
) -> torch.Tensor:
    return torch.norm(x-y, p, dim=dim)