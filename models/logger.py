

from typing import Tuple, List, Optional, Iterable, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import os
from collections import defaultdict
from functools import partial

from .base import AdversarialDefensiveModel
from .layerops import MarkLayer
from src.dict2obj import Config
from src.utils import export_pickle, import_pickle




class BaseLogger:

    LAYERS = (MarkLayer, nn.AdaptiveAvgPool2d, nn.Linear)
    FTYPES = ('max', 'min', 'mean', 'norm2', 'norm1', 'norminf')

    def __init__(
        self, module: nn.Module, name: str
    ) -> None:

        self.module = module
        self.name = name
        self.reset() 
        self.ftypes = []
        for func_type in self.FTYPES:
            self.ftypes.append(
                getattr(self, func_type)
            )

    def reset(self):
        self.logging: bool = False # Logger will log infos if True ...
        self.nora: bool = True # Model takes natural samples if True, adversarial samples otherwise ... 
        self.logger = Config(
            nat=defaultdict(list), 
            adv=defaultdict(list),
        )

    @staticmethod
    def dims(x: np.ndarray, except_: Union[int, Iterable] = 0) -> Tuple:
        if isinstance(except_, int):
            except_ = [except_]
        except_ = set(except_)
        ndim = x.ndim
        dims = set(range(0, ndim))
        dims = dims - except_
        return tuple(dims)

    @staticmethod
    def _normp(x: np.ndarray, p: Union[int, float] = 2) -> List:
        x = x.reshape(x.shape[0], -1)
        return np.linalg.norm(x, ord=p, axis=1).tolist()

    def max(self, logger: defaultdict, inputs: np.ndarray, outputs: np.ndarray) -> None:
        dims = self.dims(outputs, except_=0)
        values = outputs.max(axis=dims).tolist()
        logger['max'] += values

    def min(self, logger: defaultdict, inputs: np.ndarray, outputs: np.ndarray) -> None:
        dims = self.dims(outputs, except_=0)
        values = outputs.min(axis=dims).tolist()
        logger['min'] += values

    def mean(self, logger: defaultdict, inputs: np.ndarray, outputs: np.ndarray) -> None:
        dims = self.dims(outputs, except_=0)
        values = outputs.mean(axis=dims).tolist()
        logger['mean'] += values
    
    def norm2(self, logger: defaultdict, inputs: np.ndarray, outputs: np.ndarray) -> None:
        values = self._normp(outputs, p=2)
        logger['norm2'] += values

    def norm1(self, logger: defaultdict, inputs: np.ndarray, outputs: np.ndarray) -> None:
        values = self._normp(outputs, p=1)
        logger['norm1'] += values

    def norminf(self, logger: defaultdict, inputs: np.ndarray, outputs: np.ndarray) -> None:
        values = self._normp(outputs, p=float('inf'))
        logger['norminf'] += values

    def step(self, inputs: np.ndarray, outputs: np.ndarray) -> None:
        logger = self.logger.nat if self.nora else self.logger.adv
        for func in self.ftypes:
            func(logger, inputs, outputs)

    @torch.no_grad()
    def log(self, inputs: Tuple, outputs: torch.Tensor) -> int:
        if not self.logging:
            return 0
        inputs = inputs[0].clone().detach().cpu().numpy()
        outputs = outputs.clone().detach().cpu().numpy()
        self.step(inputs, outputs)
        return 1

class DiffLogger(BaseLogger):

    # LAYERS = (nn.AdaptiveAvgPool2d, nn.Linear)
    # FTYPES = ('max', 'min', 'mean', 'norm2', 'norm1', 'norminf')

    def __init__(
        self, module: nn.Module, name: str
    ) -> None:
        super(DiffLogger, self).__init__(module, name)

        self.logger = Config(
            diff=defaultdict(list)
        )
        self.flag_nat: bool = False
        self.flag_adv: bool = False

    def reset(self):
        self.logging: bool = False # Logger will log infos if True ...
        self.nora: bool = True # Model takes natural samples if True, adversarial samples otherwise ... 
        self.logger = Config(
            diff=defaultdict(list), 
        )
        self.flag_nat: bool = False
        self.flag_adv: bool = False

    def step(self, inputs: np.ndarray, outputs: np.ndarray) -> None:
        logger= self.logger.diff
        for func in self.ftypes:
            func(logger, inputs, outputs)

    @torch.no_grad()
    def log(self, inputs: Tuple, outputs: torch.Tensor) -> int:
        if not self.logging:
            return 0
        inputs = inputs[0].clone().detach().cpu().numpy()
        outputs = outputs.clone().detach().cpu().numpy()
        if self.nora:
            assert not self.flag_nat, \
                "Model takes as input natural samples again, but adversarial samples expected ..."
            self.temp_nat = (inputs, outputs)
            self.flag_nat = True
        else:
            assert not self.flag_adv, \
                "Model takes as input adversarial samples again, but natural samples expected ..."
            self.temp_adv = (inputs, outputs)
            self.flag_adv = True
        
        if self.flag_nat and self.flag_adv:
            self.step(
                self.temp_nat[0] - self.temp_adv[0],
                self.temp_nat[1] - self.temp_adv[1]
            )
            self.flag_nat = False
            self.flag_adv = False
        
        return 1


class Loggers:

    def __init__(self, model: AdversarialDefensiveModel) -> None:
        self.loggers = []
        self.model = model
        self.register()
    
    def record(self, mode: bool = True):
        for logger in self.loggers:
            logger.logging = mode

    def nora(self, mode: bool = True):
        for logger in self.loggers:
            logger.nora = mode

    def reset(self):
        for logger in self.loggers:
            logger.reset()

    def register(self):
        def hook(module, inputs, outputs, logger):
            logger.log(inputs, outputs)

        for name, m in self.model.named_modules():
            if isinstance(m, BaseLogger.LAYERS):
                logger = BaseLogger(m, name)
                m.register_forward_hook(
                    hook=partial(hook, logger=logger)
                )
                self.loggers.append(logger)
            if isinstance(m, DiffLogger.LAYERS):
                logger = DiffLogger(m, name)
                m.register_forward_hook(
                    hook=partial(hook, logger=logger)
                )
                self.loggers.append(logger)
    @property 
    def infos_dict(self):
        infos = defaultdict(dict)
        for logger in self.loggers:
            for key, value in logger.logger.items():
                infos[logger.name][key] = value
        return infos

    def save(self, log_path: str, filename = "model.stats", *, T: int = 8888):
        file_ = os.path.join(log_path, filename)
        try:
            data = import_pickle(file_)
        except:
            data = dict()
        data[T] = self.infos_dict
        export_pickle(
            data,
            file_
        )
            
class BlankLoggers(Loggers):

    def __init__(self, model=None) -> None:
        super(BlankLoggers, self).__init__(nn.Identity())

    def save(self, log_path: str, filename = "model.stats", *, T: int = 8888):
        pass
