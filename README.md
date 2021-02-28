


## Usage


### Training


    python STD.py resnet32 cifar10 -lp=STD --epochs=164 -wd=0.0002
    python AT.py resnet32 cifar10 -lp=AT --epochs=200 -wd=0.0002
    python TRADES.py resnet32 cifar10 -lp=TRADES --epochs=76 -wd=0.0002



Early stopping against over-fitting:

```
python AT.py resnet32 cifar10 -lp=default --epochs=110 -wd=0.0005
python TRADES.py resnet32 cifar10 -lp=default --epochs=110 -wd=0.0005
```



### Evaluation


Set the saved path as SP.

    python white_box_attack.py resnet32 cifar10 SP --attack=pgd-linf --epsilon_min=0 --epsilon_max=1 --epsilon_times=20
    python transfer_attack.py resnet32 SP1 resnet32 SP2 cifar10
    python auto_attack.py resnet32 cifar10 SP --norm=Linf --version=standard





## settings



Note that the first two events stepsize and steps are the actual options for the evaluation while the next events is the real meaning of stepsize.



### CIFAR

#### $\ell_{\infty}(\epsilon=8/255)$



$\epsilon=16/255$ is also a usual choice.

|                  | PGD-10 | PGD-20 | PGD-40 | PGD-1000 |  AA  | DeepFool |  BBA  | FGSM |
| :--------------: | :----: | :----: | :----: | :------: | :--: | :------: | :---: | :--: |
|     stepsize     |  0.25  |  0.1   |  0.1   |   0.1    |  -   |   0.02   | 0.001 |  -   |
|      steps       |   10   |   20   |   40   |   1000   |  -   |    50    | 1000  |  -   |
|   rel_stepsize   |  0.25  |  0.1   |  0.1   |   0.1    |  -   |    -     |   -   |  -   |
|   abs_stepsize   | 2/255  | 0.0031 | 0.0031 |  0.0031  |  -   |    -     |   -   |  -   |
| initial_stepsize |   -    |   -    |   -    |    -     |  -   |    -     |   -   |  -   |
|    overshoot     |   -    |   -    |   -    |    -     |  -   |   0.02   |   -   |  -   |
|        lr        |   -    |   -    |   -    |    -     |  -   |    -     | 0.001 |  -   |



#### $\ell_2 (\epsilon=0.5)$



|                  | PGD-50 |  AA  | DeepFool |  BBA  | C&W  |
| :--------------: | :----: | :--: | :------: | :---: | :--: |
|     stepsize     |  0.1   |  -   |   0.02   | 0.001 | 0.01 |
|      steps       |   50   |  -   |    50    | 1000  | 1000 |
|   rel_stepsize   |  0.1   |  -   |    -     |   -   |  -   |
|   abs_stepsize   |  0.05  |  -   |    -     |   -   |  -   |
| initial_stepsize |   -    |  -   |    -     |   -   |  -   |
|    overshoot     |   -    |  -   |   0.02   |   -   |  -   |
|        lr        |   -    |  -   |    -     | 0.001 | 0.01 |



#### $\ell_1 (\epsilon=12)$



|                  | PGD-50 | Sparse |  BBA  |
| :--------------: | :----: | :----: | :---: |
|     stepsize     |  0.05  |  0.05  | 0.001 |
|      steps       |   50   |   50   | 1000  |
|   rel_stepsize   |  0.05  |  0.05  |   -   |
|   abs_stepsize   |  0.6   |  0.6   |   -   |
| initial_stepsize |   -    |   -    |   -   |
|    overshoot     |   -    |   -    |   -   |
|        lr        |   -    |   -    | 0.001 |



### MNIST

### $\ell_{\infty} (\epsilon=0.3)$



|                  |  PGD-50  | PGD-100  |  AA  | DeepFool |  BBA  | FGSM |
| :--------------: | :------: | :------: | :--: | :------: | :---: | :--: |
|     stepsize     | 0.033333 | 0.033333 |  -   |   0.02   | 0.001 |  -   |
|      steps       |    50    |   100    |  -   |    50    | 1000  |  -   |
|   rel_stepsize   | 0.033333 | 0.033333 |  -   |    -     |   -   |  -   |
|   abs_stepsize   |   0.01   |   0.01   |  -   |    -     |   -   |  -   |
| initial_stepsize |    -     |    -     |  -   |    -     |   -   |  -   |
|    overshoot     |    -     |    -     |  -   |   0.02   |   -   |  -   |
|        lr        |    -     |    -     |  -   |    -     | 0.001 |  -   |



#### $\ell_2 (\epsilon=2)$



|                  | PGD-100 |  AA  | DeepFool |  BBA  | C&W  |
| :--------------: | :-----: | :--: | :------: | :---: | :--: |
|     stepsize     |  0.05   |  -   |   0.02   | 0.001 | 0.01 |
|      steps       |   100   |  -   |    50    | 1000  | 1000 |
|   rel_stepsize   |  0.05   |  -   |    -     |   -   |  -   |
|   abs_stepsize   |   0.1   |  -   |    -     |   -   |  -   |
| initial_stepsize |    -    |  -   |    -     |   -   |  -   |
|    overshoot     |    -    |  -   |   0.02   |   -   |  -   |
|        lr        |    -    |  -   |    -     | 0.001 | 0.01 |



#### $\ell_1 (\epsilon=10)$



|                  | PGD-50 | Sparse |  BBA  |
| :--------------: | :----: | :----: | :---: |
|     stepsize     |  0.05  |  0.05  | 0.001 |
|      steps       |   50   |   50   | 1000  |
|   rel_stepsize   |  0.05  |  0.05  |   -   |
|   abs_stepsize   |  0.5   |  0.5   |   -   |
| initial_stepsize |   -    |   -    |   -   |
|    overshoot     |   -    |   -    |   -   |
|        lr        |   -    |   -    | 0.001 |



## results





### $\ell_{\infty}$

| $\epsilon$ |    -     |     0      | 8/255  | 16/255 | 8/255  | 16/255 | 8/255  | 16/255 | 8/255 | 16/255 |  8/255   |  16/255  | 8/255 | 16/255 | 8/255 | 16/255 |
| :--------: | :------: | :--------: | :----: | :----: | :----: | :----: | :----: | :----: | :---: | :----: | :------: | :------: | :---: | :----: | :---: | :----: |
|   Method   |   Net    |   TA(%)    | PGD-10 | PGD-10 | PGD-20 | PGD-20 | PGD-40 | PGD-40 |  AA   |   AA   | DeepFool | DeepFool |  BBA  |  BBA   | FGSM  |  FGSM  |
|    STD     | ResNet32 | **93.270** | 0.000  | 0.000  | 0.000  | 0.000  | 0.000  | 0.000  | 0.000 | 0.000  |  0.019   |  0.000   |       |        |       |        |
|     AT     | ResNet32 |   79.420   | 48.300 | 18.410 | 48.440 | 19.280 | 47.460 | 15.500 |       |        |          |          |       |        |       |        |
|   TRADES   | ResNet32 |            |        |        |        |        |        |        |       |        |          |          |       |        |       |        |

