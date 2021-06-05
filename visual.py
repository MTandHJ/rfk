#!/usr/bin/env python


import argparse
import os
from  src.config import FTYPES, STATS_FILENAME

METHOD = "Visualization"
FMT = "{description}"


parser = argparse.ArgumentParser()
parser.add_argument("log_path", type=str)
parser.add_argument("--ftype", type=str, choices=FTYPES, default='all')
parser.add_argument("-m", "--description", type=str, default="vvv")
opts = parser.parse_args()
opts.description = FMT.format(**opts.__dict__)




def load_cfg() -> 'Config':
    from src.dict2obj import Config
    from src.utils import import_pickle
    from src.visualization import BaseVisualizer

    cfg = Config()
    
    # load data
    file_ = os.path.join(opts.log_path, STATS_FILENAME)
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
        path = os.path.join(opts.log_path, opts.description)
        try:
            os.mkdir(path)
        except FileExistsError:
            pass
        plt.tight_layout()
        plt.savefig(
            os.path.join(path, ftype),
            bbox_inches = 'tight'
        )
   

if __name__ == "__main__":
    from src.utils import mkdirs, readme
    cfg = load_cfg()
    main(**cfg)






