


from typing import Callable, Any, Union, Optional, List, Tuple, Dict, Iterable, cast
import torch
import torch.nn as nn
import foolbox as fb
import os

from models.base import AdversarialDefensiveModule
from .dict2obj import Config
from .utils import AverageMeter, ProgressMeter, TrackMeter, MultiImageMeter, timemeter, getLogger
from .loss_zoo import cross_entropy, kl_divergence, lploss, mse_loss
from .config import SAVED_FILENAME, PRE_BESTNAT, PRE_BESTROB, \
                        BOUNDS, PREPROCESSING, DEVICE


def enter_attack_exit(func) -> Callable:
    def wrapper(attacker: "Adversary", *args, **kwargs):
        attacker.model.attack(True)
        results = func(attacker, *args, **kwargs)
        attacker.model.attack(False)
        return results
    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    return wrapper


class Coach:
    
    def __init__(
        self, model: AdversarialDefensiveModule,
        loss_func: Callable, 
        optimizer: torch.optim.Optimizer, 
        learning_policy: "learning rate policy",
        device: torch.device = DEVICE
    ):
        self.model = model
        self.device = device
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.learning_policy = learning_policy

        self.meter = Config(
            loss=AverageMeter("Loss"),
            acc=AverageMeter("Acc", fmt=".3%")
        )
        self.meter.progress = ProgressMeter(*self.meter.values())

        self.logger = Config(
            major=TrackMeter("Major Loss"),
            minor=TrackMeter("Minor Loss")
            # other ...
        )
        self.logger.plotter = MultiImageMeter(*self.logger.values(), title="summary")

        self._best_nat = 0.
        self._best_rob = 0.
        self.steps = 0

    def summary(self, log_path: str):
        self.logger.plotter.plot()
        self.logger.plotter.save(log_path)

    def save_best_nat(self, acc_nat: float, path: str, prefix: str = PRE_BESTNAT):
        if acc_nat > self._best_nat:
            self._best_nat = acc_nat
            self.save(path, '_'.join((prefix, SAVED_FILENAME)))
            return 1
        else:
            return 0
    
    def save_best_rob(self, acc_rob: float, path: str, prefix: str = PRE_BESTROB):
        if acc_rob > self._best_rob:
            self._best_rob = acc_rob
            self.save(path, '_'.join((prefix, SAVED_FILENAME)))
            return 1
        else:
            return 0

    def check_best(
        self, acc_nat: float, acc_rob: float,
        path: str, epoch: int = 8888
    ):
        logger = getLogger()
        if self.save_best_nat(acc_nat, path):
            logger.debug(f"[Coach] Saving the best nat ({acc_nat:.3%}) model at epoch [{epoch}]")
        if self.save_best_rob(acc_rob, path):
            logger.debug(f"[Coach] Saving the best rob ({acc_rob:.3%}) model at epoch [{epoch}]")
        
    def save(self, path: str, filename: str = SAVED_FILENAME) -> None:
        torch.save(self.model.state_dict(), os.path.join(path, filename))

    @timemeter("Train/Epoch")
    def train(
        self, 
        trainloader: Iterable[Tuple[torch.Tensor, torch.Tensor]], 
        *, epoch: int = 8888
    ) -> float:

        self.meter.progress.step() # reset the meter
        self.model.train()
        for inputs, labels in trainloader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            self.model.train() # make sure in training mode
            outs = self.model(inputs)
            loss = self.loss_func(outs, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            accuracy_count = (outs.argmax(-1) == labels).sum().item()
            self.meter.loss.update(loss.item(), inputs.size(0), mode="mean")
            self.meter.acc.update(accuracy_count, inputs.size(0), mode="sum")

            # for summary
            self.logger.major(data=loss.item(), T=self.steps)

            self.steps += 1

        self.meter.progress.display(epoch=epoch) 
        self.learning_policy.step() # update the learning rate
        return self.meter.loss.avg

    @timemeter("AdvTraining/Epoch")
    def adv_train(
        self, 
        trainloader: Iterable[Tuple[torch.Tensor, torch.Tensor]], 
        attacker: "Adversary", 
        *, epoch: int = 8888
    ) -> float:
    
        self.meter.progress.step() # reset the meter
        for inputs, labels in trainloader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            clipped = attacker(inputs, labels)
            
            self.model.train()
            outs = self.model(clipped)
            loss = self.loss_func(outs, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            accuracy_count = (outs.argmax(-1) == labels).sum().item()
            self.meter.loss.update(loss.item(), inputs.size(0), mode="mean")
            self.meter.acc.update(accuracy_count, inputs.size(0), mode="sum")

            # for summary
            self.logger.major(data=loss.item(), T=self.steps)

            self.steps += 1

        self.meter.progress.display(epoch=epoch)
        self.learning_policy.step() # update the learning rate
        return self.meter.loss.avg

    @timemeter("ALP/Epoch")
    def alp(
        self,
        trainloader: Iterable[Tuple[torch.Tensor, torch.Tensor]],
        attacker: "Adversary", 
        *, leverage: float = .5, epoch: int = 8888
    ) -> float:

        self.meter.progress.step()
        for inputs, labels in trainloader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            clipped = attacker(inputs, labels)

            self.model.train()
            logits_nat = self.model(inputs)
            logits_adv = self.model(clipped)
            loss_nat = self.loss_func(logits_nat, labels)
            loss_adv = self.loss_func(logits_adv, labels)
            loss_reg = mse_loss(logits_adv, logits_nat)
            loss = loss_nat + loss_adv + leverage * loss_reg

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            acc_count = (logits_adv.argmax(-1) == labels).sum().item()
            self.meter.loss.update(loss.item(), inputs.size(0), mode="mean")
            self.meter.acc.update(acc_count, inputs.size(0), mode="sum")

            # for summary
            self.logger.major(data=loss_nat.item(), T=self.steps)
            self.logger.minor(data=loss_adv.item(), T=self.steps)

            self.steps += 1

        self.meter.progress.display(epoch=epoch)
        self.learning_policy.step()

        return self.meter.loss.avg

    @timemeter("TRADES/Epoch")
    def trades(
        self, 
        trainloader: Iterable[Tuple[torch.Tensor, torch.Tensor]],
        attacker: "Adversary", 
        *, leverage: float = 6., epoch: int = 8888
    ) -> float:

        self.meter.progress.step()
        for inputs, labels in trainloader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            with torch.no_grad():
                self.model.eval()
                logits = self.model(inputs).detach()
            inputs_adv = attacker(inputs, logits)
            
            self.model.train()
            logits_nat = self.model(inputs)
            logits_adv = self.model(inputs_adv)
            loss_nat = cross_entropy(logits_nat, labels)
            loss_adv = kl_divergence(logits_adv, logits_nat)
            loss = loss_nat + leverage * loss_adv

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            acc_count = (logits_adv.argmax(-1) == labels).sum().item()
            self.meter.loss.update(loss.item(), inputs.size(0), mode="mean")
            self.meter.acc.update(acc_count, inputs.size(0), mode="sum")

            # for summary
            self.logger.major(data=loss_nat.item(), T=self.steps)
            self.logger.minor(data=loss_adv.item(), T=self.steps)

            self.steps += 1

        self.meter.progress.display(epoch=epoch)
        self.learning_policy.step()

        return self.meter.loss.avg


class Adversary:
    
    def __init__(
        self, model: AdversarialDefensiveModule, 
        attacker: Callable, device: torch.device = DEVICE,
    ) -> None:

        model.eval()
        self.model = model
        self.attacker = attacker 
        self.device = device

    def attack(self, inputs: torch.Tensor, targets: Union[Iterable, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError

    def __call__(self, inputs: torch.Tensor, targets: Union[Iterable, torch.Tensor]) -> torch.Tensor:
        return self.attack(inputs, targets)

class AdversaryForTrain(Adversary):

    @enter_attack_exit
    def attack(self, inputs: torch.Tensor, targets: Union[Iterable, torch.Tensor]) -> torch.Tensor:
        self.model.eval() # some methods require training mode
        return self.attacker(self.model, inputs, targets)

class AdversaryForValid(Adversary): 

    def attack(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        self.model.eval() # make sure in inference mode
        return self.attacker(self.model, inputs, targets)

    @torch.no_grad()
    def accuracy(self, inputs: torch.Tensor, labels: torch.Tensor) -> int:
        self.model.eval() # make sure in evaluation mode ...
        predictions = self.model(inputs).argmax(dim=-1)
        accuracy = (predictions == labels)
        return cast(int, accuracy.sum().item())

    def evaluate(
        self, 
        dataloader: Iterable[Tuple[torch.Tensor, torch.Tensor]], 
        *, defending: bool = True
    ) -> Tuple[float, float]:

        datasize = len(dataloader.dataset) # type: ignore
        acc_nat = 0
        acc_adv = 0
        self.model.defend(defending) # enter 'defending' mode
        self.model.eval()
        for inputs, labels in dataloader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            inputs_adv = self.attack(inputs, labels)
            acc_nat += self.accuracy(inputs, labels)
            acc_adv += self.accuracy(inputs_adv, labels)
        return acc_nat / datasize, acc_adv / datasize


class FBAdversary(Adversary):
    """
    FBAdversary is mainly based on foolbox, especially pytorchmodel.
    model: Make sure that the model's output is the logits or the attack is adapted.
    attacker: the attack implemented by foolbox or a similar one
    device: ...
    bounds: typically [0, 1]
    preprocessing: including mean, std, which is similar to normalizer
    criterion: typically given the labels and consequently it is Misclassification, 
            other critera could be given to carry target attack or black attack.
    """
    def __init__(
        self, model: AdversarialDefensiveModule, 
        attacker: Callable, epsilon: Union[None, float, List[float]],
        device: torch.device = DEVICE,
        bounds: Tuple[float, float] = BOUNDS, 
        preprocessing: Optional[Dict] = PREPROCESSING
    ) -> None:
        super(FBAdversary, self).__init__(
            model=model, attacker=attacker, device=device
        )
        self.model.eval()
        self.fmodel = fb.PyTorchModel(
            model,
            bounds=bounds,
            preprocessing=preprocessing,
            device=device
        )
        self.device = device
        self.epsilon = epsilon
        self.attacker = attacker 

    def attack(
        self, 
        inputs: torch.Tensor, 
        criterion: Any, 
        epsilon: Union[None, float, List[float]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        if epsilon is None:
            epsilon = self.epsilon
        self.model.eval() # make sure in evaluation mode ...
        return self.attacker(self.fmodel, inputs, criterion, epsilons=epsilon)

    def __call__(
        self, 
        inputs: torch.Tensor, criterion: Any,
        epsilon: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        return self.attack(inputs, criterion, epsilon)


class FBDefense:
    def __init__(
        self, 
        model: nn.Module, 
        device: torch.device = DEVICE, 
        bounds: Tuple[float, float] = BOUNDS, 
        preprocessing: Optional[Dict] = PREPROCESSING
    ) -> None:
        self.rmodel = fb.PyTorchModel(
            model,
            bounds=bounds,
            preprocessing=preprocessing,
            device=device            
        )

        self.model = model
    
    def train(self, mode: bool = True) -> None:
        self.model.train(mode=mode)

    def eval(self) -> None:
        self.train(mode=False)

    def query(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.rmodel(inputs)

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.query(inputs)

