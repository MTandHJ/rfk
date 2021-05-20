
import torch.nn as nn
from .base import AdversarialDefensiveModel


class Sequential(nn.Sequential, AdversarialDefensiveModel): ...
class ModuleList(nn.ModuleList, AdversarialDefensiveModel): ...

class TriggerBN1d(AdversarialDefensiveModel):
    
    def __init__(self, num_features):
        super(TriggerBN1d, self).__init__()
        self.bn_clean = nn.BatchNorm1d(num_features)
        self.bn_adv = nn.BatchNorm1d(num_features)

        nn.init.constant_(self.bn_adv.weight, 1.)
        nn.init.constant_(self.bn_adv.bias, 0.)
        nn.init.constant_(self.bn_clean.weight, 1.)
        nn.init.constant_(self.bn_clean.bias, 0.)

    def forward(self, x):
        if self.defending:
            return self.bn_adv(x)
        else:
            return self.bn_clean(x)


class TriggerBN2d(AdversarialDefensiveModel):
    
    def __init__(self, num_features):
        super(TriggerBN2d, self).__init__()
        self.bn_clean = nn.BatchNorm2d(num_features)
        self.bn_adv = nn.BatchNorm2d(num_features)

        nn.init.constant_(self.bn_adv.weight, 1.)
        nn.init.constant_(self.bn_adv.bias, 0.)
        nn.init.constant_(self.bn_clean.weight, 1.)
        nn.init.constant_(self.bn_clean.bias, 0.)

    def forward(self, x):
        if self.defending:
            return self.bn_adv(x)
        else:
            return self.bn_clean(x)
