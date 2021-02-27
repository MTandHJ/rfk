


## Usage


### Training


    python STD.py resnet32 cifar10
    python AT.py resnet32 cifar10 -lp=AT --epochs=200
    python TRADES.py resnet32 cifar10 -lp=TRADES --epochs=76


### Evaluation


Set the saved path as SP.

    python white_box_attack.py resnet32 cifar10 SP --attack=pgd-linf --epsilon_min=0 --epsilon_max=1 --epsilon_times=20
    python transfer_attack.py resnet32 SP1 resnet32 SP2 cifar10
    python auto_attack.py resnet32 cifar10 SP --norm=Linf --version=standard
