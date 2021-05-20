


import torch.nn as nn
import abc

class ADType(abc.ABC): ...

class AdversarialDefensiveModel(ADType, nn.Module):
    """
    Define some basic properties.
    """
    def __init__(self) -> None:
        super(AdversarialDefensiveModel, self).__init__()
        # Some model's outputs for training(evaluating) 
        # and attacking are different.
        self.attacking: bool = False
        self.defending: bool = True

        
    def attack(self, mode: bool = True) -> None:
        # enter attacking mode
        # for adversary only
        self.attacking = mode
        for module in self.children():
            if isinstance(module, ADType):
                module.attack(mode)

    def defend(self, mode: bool = True) -> None:
        # enter defense mode
        # for some special techniques
        self.defending = mode
        for module in self.children():
            if isinstance(module, ADType):
                module.defend(mode)



if __name__ == "__main__":
    
    model = AdversarialDefensiveModel()
    model.child1 = AdversarialDefensiveModel()
    model.child2 = AdversarialDefensiveModel()

    print(model.attack)
    model.attack()
    for m in model.children():
        print(m.attacking)

    model.defense()
    for m in model.children():
        print(m.attacking)

