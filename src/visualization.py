

from freeplot.utils import axis
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

    def violinplot(self, ftype: str = 'norm2', show_layers: bool = False):
        fp = FreePlot(
            shape=self.shape,
            figsize=self.figsize,
            sharey=False,
            dpi=200
        )

        x = self.layers if show_layers else None
        
        flag = True
        for i, kind in enumerate(self.KINDS):
            for j, epoch in enumerate(self.epochs):
                data = [np.array(self.data[epoch][layer][kind][ftype]) for layer in self.layers]
                fp.violinplot(x=x, y=data, index=(i, j))
                if flag:
                    fp[i, j].set(title=epoch)
            fp.set_label(kind, index=(i, 0), axis='y')
            flag = False
        return fp
        