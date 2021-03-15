#!/usr/bin/env python



import torch
import torch.nn as nn
import argparse
from src.loadopts import *



METHOD = "AT"
SAVE_FREQ = 20
FMT = "{description}={learning_policy}-{optimizer}-{lr}" \
        "={attack}-{epsilon:.4f}-{stepsize}-{steps}" \
        "={batch_size}={transform}"

parser = argparse.ArgumentParser()
parser.add_argument("model", type=str)
parser.add_argument("dataset", type=str)

# adversarial training settings
parser.add_argument("--attack", type=str, default="pgd-linf")
parser.add_argument("--epsilon", type=float, default=8/255)
parser.add_argument("--stepsize", type=float, default=0.25, 
                    help="pgd:rel_stepsize, cwl2:step_size, deepfool:overshoot, bb:lr")
parser.add_argument("--steps", type=int, default=10)

# basic settings
parser.add_argument("--loss", type=str, default="cross_entropy")
parser.add_argument("--optimizer", type=str, choices=("sgd", "adam"), default="sgd")
parser.add_argument("-mom", "--momentum", type=float, default=0.9,
                help="the momentum used for SGD")
parser.add_argument("-beta1", "--beta1", type=float, default=0.9,
                help="the first beta argument for Adam")
parser.add_argument("-beta2", "--beta2", type=float, default=0.999,
                help="the second beta argument for Adam")
parser.add_argument("-wd", "--weight_decay", type=float, default=5e-4,
                help="weight decay")
parser.add_argument("-lr", "--lr", "--LR", "--learning_rate", type=float, default=0.1)
parser.add_argument("-lp", "--learning_policy", type=str, default="AT", 
                help="learning rate schedule defined in config.py")
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("-b", "--batch_size", type=int, default=128)
parser.add_argument("--transform", type=str, default='default', 
                help="the data augmentation which will be applied during training.")
parser.add_argument("--resume", action="store_true", default=False)
parser.add_argument("--progress", action="store_true", default=False, 
                help="show the progress if true")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("-m", "--description", type=str, default="AT")
opts = parser.parse_args()
opts.description = FMT.format(**opts.__dict__)



def load_cfg():
    from src.dict2obj import Config
    from src.base import Coach, AdversaryForTrain
    from src.utils import gpu, set_seed, load_checkpoint

    cfg = Config()
    set_seed(opts.seed)

    # the model and other settings for training
    model = load_model(opts.model)(num_classes=get_num_classes(opts.dataset))
    device = gpu(model)

    # load the dataset
    trainset = load_dataset(
        dataset_type=opts.dataset, 
        transform=opts.transform, 
        train=True
    )
    cfg['trainloader'] = load_dataloader(
        dataset=trainset, 
        batch_size=opts.batch_size, 
        train=True,
        show_progress=opts.progress
    )
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
    
    # load the optimizer and learning_policy
    optimizer = load_optimizer(
        model=model, optim_type=opts.optimizer, lr=opts.lr,
        momentum=opts.momentum, betas=(opts.beta1, opts.beta2),
        weight_decay=opts.weight_decay
    )
    learning_policy = load_learning_policy(
        optimizer=optimizer, 
        learning_policy_type=opts.learning_policy,
        T_max=opts.epochs
    )

    # generate the path for logging information and saving parameters
    cfg['info_path'], log_path = generate_path(
        method=METHOD, dataset_type=opts.dataset, 
        model=opts.model, description=opts.description
    )
    if opts.resume:
        cfg['start_epoch'] = load_checkpoint(
            path=cfg.info_path, model=model, 
            optimizer=optimizer, lr_scheduler=learning_policy
        )
    else:
        cfg['start_epoch'] = 0

    cfg['coach'] = Coach(
        model=model, device=device, 
        loss_func=load_loss_func(opts.loss), 
        normalizer=normalizer, optimizer=optimizer, 
        learning_policy=learning_policy
    )

    # set the attack
    attack, bounds, preprocessing = load_attacks(
        attack_type=opts.attack, dataset_type=opts.dataset, 
        stepsize=opts.stepsize, steps=opts.steps
    )

    cfg['attacker'] = AdversaryForTrain(
        model=model, attacker=attack, device=device, 
        bounds=bounds, preprocessing=preprocessing, epsilon=opts.epsilon
    )

    cfg['valider'] = load_valider(
        model=model, device=device, dataset_type=opts.dataset
    )

    return cfg, log_path


def main(
    coach, attacker, valider, 
    trainloader, testloader, 
    start_epoch, info_path
):  
    from src.utils import save_checkpoint
    for epoch in range(start_epoch, opts.epochs):

        if epoch % SAVE_FREQ == 0:
            save_checkpoint(info_path, coach.model, coach.optimizer, coach.learning_policy, epoch)

            train_accuracy, train_success = valider.evaluate(trainloader)
            valid_accuracy, valid_success = valider.evaluate(testloader)
            print(f"Train >>> [TA: {train_accuracy:.5f}]    [RA: {1 - train_success:.5f}]")
            print(f"Test. >>> [TA: {valid_accuracy:.5f}]    [RA: {1 - valid_success:.5f}]")
            writter.add_scalars("Accuracy", {"train":train_accuracy, "valid":valid_accuracy}, epoch)
            writter.add_scalars("Success", {"train":train_success, "valid":valid_success}, epoch)

        running_loss = coach.adv_train(trainloader, attacker, epoch=epoch)
        writter.add_scalar("Loss", running_loss, epoch)

    train_accuracy, train_success = valider.evaluate(trainloader)
    valid_accuracy, valid_success = valider.evaluate(testloader)
    print(f"Train >>> [TA: {train_accuracy:.5f}]    [RA: {1 - train_success:.5f}]")
    print(f"Test. >>> [TA: {valid_accuracy:.5f}]    [RA: {1 - valid_success:.5f}]")
    writter.add_scalars("Accuracy", {"train":train_accuracy, "valid":valid_accuracy}, epoch)
    writter.add_scalars("Success", {"train":train_success, "valid":valid_success}, epoch)


if __name__ ==  "__main__":
    from torch.utils.tensorboard import SummaryWriter
    from src.utils import mkdirs, readme
    cfg, log_path = load_cfg()
    mkdirs(cfg.info_path, log_path)
    readme(cfg.info_path, opts)
    readme(log_path, opts, mode="a")
    writter = SummaryWriter(log_dir=log_path, filename_suffix=METHOD)

    main(**cfg)

    cfg['coach'].save(cfg.info_path)
    writter.close()


