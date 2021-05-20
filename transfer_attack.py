#!/usr/bin/env python


"""
Transfer Attack: utilize the source_model to attack 
the target model...
"""

from typing import Tuple
import torch
import argparse
from src.loadopts import *

METHOD = "Transfer"
FMT = "{description}={attack}-{epsilon:.4f}-{stepsize}-{steps}"

parser = argparse.ArgumentParser()
parser.add_argument("source_model", type=str)
parser.add_argument("source_path", type=str)
parser.add_argument("target_model", type=str)
parser.add_argument("target_path", type=str)
parser.add_argument("dataset", type=str)

# adversarial settings
parser.add_argument("--attack", type=str, default="pgd-linf")
parser.add_argument("--epsilon", type=float, default=8/255)
parser.add_argument("--stepsize", type=float, default=0.1, 
                    help="pgd:rel_stepsize, cwl2:step_size, deepfool:overshoot, bb:lr")
parser.add_argument("--steps", type=int, default=20)

# basic settings
parser.add_argument("-b", "--batch_size", type=int, default=128)
parser.add_argument("--transform", type=str, default='default', 
                help="the data augmentation which will be applied in training mode.")
parser.add_argument("--progress", action="store_false", default=True, 
                help="show the progress if true")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("-m", "--description", type=str, default="attack")
opts = parser.parse_args()
opts.description = FMT.format(**opts.__dict__)


def load_cfg() -> Tuple[Config, str]:
    from src.dict2obj import Config
    from src.base import FBDefense, AdversaryForValid
    from src.utils import gpu, load, set_seed

    cfg = Config()
    set_seed(opts.seed)

    # load the source_model
    source_model = load_model(opts.source_model)(num_classes=get_num_classes(opts.dataset))
    device = gpu(source_model)
    load(
        model=source_model, 
        path=opts.source_path,
        device=device
    )

    # load the target_model
    target_model = load_model(opts.target_model)(num_classes=get_num_classes(opts.dataset))
    device = gpu(target_model)
    load(
        model=target_model, 
        filename=opts.target_path + "/paras.pt", 
        device=device
    )

    # load the testset
    testset = load_dataset(
        dataset_type=opts.dataset,
        transform=opts.transform,
        train=False
    )
    cfg['testloader'] = load_dataloader(
        dataset=testset, 
        batch_size=opts.batch_size, 
        train=False,
        show_progress=opts.progress
    )
    normalizer = load_normalizer(dataset_type=opts.dataset)

    # generate the log path
    mix_model = opts.source_model + "---" + opts.target_model
    _, log_path = generate_path(
        method=METHOD, dataset_type=opts.dataset,
        model=mix_model, description=opts.description
    )

    # set the attacker
    attack, bounds, preprocessing = load_attacks(
        attack_type=opts.attack, dataset_type=opts.dataset, 
        stepsize=opts.stepsize, steps=opts.steps
    )

    cfg['attacker'] = AdversaryForValid(
        model=source_model, attacker=attack, device=device,
        bounds=bounds, preprocessing=preprocessing, epsilon=opts.epsilon
    )

    # set the defender ...
    cfg['defender'] = FBDefense(
        model=target_model, device=device,
        bounds=bounds, preprocessing=preprocessing
    )

    return cfg, log_path

def main(defender, attacker, testloader):
    from src.criteria import TransferClassification
    from src.utils import distance_lp
    running_success = 0.
    running_distance_linf = 0.
    running_distance_l2 = 0.
    for inputs, labels in testloader:
        inputs = inputs.to(attacker.device)
        labels = labels.to(attacker.device)

        criterion = TransferClassification(defender, labels)
        _, clipped, is_adv = attacker(inputs, criterion)
        inputs_ = inputs[is_adv]
        clipped_ = clipped[is_adv]

        dim_ = list(range(1, inputs.dim()))
        running_distance_linf += distance_lp(inputs_, clipped_, p=float("inf"), dim=dim_).sum().item()
        running_distance_l2 += distance_lp(inputs_, clipped_, p=2, dim=dim_).sum().item()
        running_success += is_adv.sum().item()

    running_distance_linf /= running_success
    running_distance_l2 /= running_success
    running_success /= len(testloader.dataset)

    results = "Success: {0:.5f}, Linf: {1:.5f}, L2: {2:.5f}".format(
        running_success, running_distance_linf, running_distance_l2
    )
    head = "-".join(map(str, (opts.attack, opts.epsilon, opts.stepsize, opts.steps)))
    writter.add_text(head, results)
    print(results)




if __name__ == "__main__":
    from torch.utils.tensorboard import SummaryWriter
    from src.utils import mkdirs, readme
    cfg, log_path = load_cfg()
    mkdirs(log_path)
    readme(log_path, opts, mode="a")
    writter = SummaryWriter(log_dir=log_path, filename_suffix=METHOD)

    main(**cfg)

    writter.close()
































