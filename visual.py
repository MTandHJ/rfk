#!/usr/bin/env python


from functools import partial
from typing import Tuple
import argparse
import os


METHOD = "Visualization"
FMT = "{description}"


parser = argparse.ArgumentParser()
parser.add_argument("log_path", type=str)
parser.add_argument("--ftype", type=str, choices=('max', 'min', 'mean', 'norm1', 'norm2', 'norminf', 'all'), default='all')
parser.add_argument("-m", "--description", type=str, default="vvv")
opts = parser.parse_args()
opts.description = FMT.format(**opts.__dict__)




def load_cfg() -> 'Config':
    from src.dict2obj import Config
    from src.utils import import_pickle
    from src.visualization import BaseVisualizer

    cfg = Config()
    
    # load data
    file_ = os.path.join(opts.log_path, "model.stats")
    data = import_pickle(file_=file_)
    cfg['visualizer'] = BaseVisualizer(data)
    if opts.ftype == "all":
        cfg['ftypes'] = ('max', 'min', 'mean', 'norm1', 'norm2', 'norminf')
    else:
        cfg['ftypes'] = (opts.ftype,)

    return cfg

def main(visualizer, ftypes):
    import matplotlib.pyplot as plt
    for ftype in ftypes:
        visualizer.violinplot(ftype=ftype)
        try:
            os.mkdir(opts.log_path)
        except FileExistsError:
            pass
        plt.savefig(
            os.path.join(opts.log_path, "_".join((opts.description, ftype))),
            bbox_inches = 'tight'
        )
   

if __name__ == "__main__":
    from src.utils import mkdirs, readme
    cfg = load_cfg()
    main(**cfg)






