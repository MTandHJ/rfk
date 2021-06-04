

from typing import Dict
import numpy as np

from freeplot.base import FreePlot
from models.logger import BaseLogger, DiffLogger


class BaseVisualizer:

    KINDS = ('nat', 'adv', 'diff')

    def __init__(self, data: Dict) -> None:
        self.data = data
        self.epochs = list(self.data.keys())
        self.layers = list(self.data[self.epochs[0]].keys())
        assert BaseLogger.FTYPES == DiffLogger.FTYPES
        self.ftypes = BaseLogger.FTYPES
        self.shape = (len(self.KINDS), len(self.epochs))
        self.figsize = (len(self.epochs) * 2.3, len(self.KINDS) * 2)

    def violinplot(self, ftype: str = 'norm2'):
        fp = FreePlot(
            shape=self.shape,
            figsize=self.figsize,
            sharey=False,
            dpi=200
        )
        
        cur_index = 0
        flag = True
        for kind in self.KINDS:
            fp[cur_index].set(ylabel=kind)
            for epoch in self.epochs:
                data = [np.array(self.data[epoch][layer][kind][ftype]) for layer in self.layers]
                fp.violinplot(x=self.layers, y=data, index=cur_index)
                if flag:
                    fp[cur_index].set(title=epoch)
                cur_index += 1
            flag = False
        return fp
        