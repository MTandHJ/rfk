

## Usage

### Training

#### CIFAR-10


    python STD.py resnet32 cifar10 -lp=STD --epochs=164 -wd=0.0002
    python AT.py resnet32 cifar10 -lp=AT --epochs=200 -wd=0.0002
    python TRADES.py resnet32 cifar10 -lp=TRADES --epochs=76 -wd=0.0002



Early stopping against over-fitting:

```
python AT.py resnet32 cifar10 -lp=default --epochs=110 -wd=0.0005
python TRADES.py resnet32 cifar10 -lp=default --epochs=110 -wd=0.0005
```



#### MNIST



```
python STD.py mnist mnist -lp=null --epochs=50 -lr=0.1
python AT.py mnist mnist -lp=null --epochs=84 -lr=0.0001 --optim=adam --epsilon=0.3 --steps=40 --stepsize=0.0333333
python TRADES.py mnist mnist -lp=TRADES-M --epochs=100 -lr=0.01 --epsilon=0.3 --steps=40 --stepsize=0.0333333
```



### Evaluation


Set the saved path as SP.

    python white_box_attack.py resnet32 cifar10 SP --attack=pgd-linf --epsilon_min=0 --epsilon_max=1 --epsilon_times=20
    python transfer_attack.py resnet32 SP1 resnet32 SP2 cifar10
    python auto_attack.py resnet32 cifar10 SP --norm=Linf --version=standard





## Settings



Note that the first two events stepsize and steps are the actual options for the evaluation while the next events is the real meaning of stepsize.



### CIFAR

#### $\ell_{\infty}(\epsilon=8/255)$



$\epsilon=16/255$ is also a usual choice.

|                  | PGD-10 | PGD-20 | PGD-40 |  AA  | DeepFool |  BBA  | FGSM |
| :--------------: | :----: | :----: | :----: | :--: | :------: | :---: | :--: |
|     stepsize     |  0.25  |  0.1   |  0.1   |  -   |   0.02   | 0.001 |  -   |
|      steps       |   10   |   20   |   40   |  -   |    50    | 1000  |  -   |
|   rel_stepsize   |  0.25  |  0.1   |  0.1   |  -   |    -     |   -   |  -   |
|   abs_stepsize   | 2/255  | 0.0031 | 0.0031 |  -   |    -     |   -   |  -   |
| initial_stepsize |   -    |   -    |   -    |  -   |    -     |   -   |  -   |
|    overshoot     |   -    |   -    |   -    |  -   |   0.02   |   -   |  -   |
|        lr        |   -    |   -    |   -    |  -   |    -     | 0.001 |  -   |



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



|                  | PGD-50 | Sparse |  BBA  | DeepFool |
| :--------------: | :----: | :----: | :---: | :------: |
|     stepsize     |  0.05  |  0.05  | 0.001 |   0.02   |
|      steps       |   50   |   50   | 1000  |    50    |
|   rel_stepsize   |  0.05  |  0.05  |   -   |    -     |
|   abs_stepsize   |  0.5   |  0.5   |   -   |    -     |
| initial_stepsize |   -    |   -    |   -   |    -     |
|    overshoot     |   -    |   -    |   -   |   0.02   |
|        lr        |   -    |   -    | 0.001 |    -     |



## Results



### CIFAR-10

#### $\ell_{\infty}$

| $\epsilon$ |    -     |    -    |     0      | 8/255  | 16/255 | 8/255  | 16/255 | 8/255  | 16/255 | 8/255  | 16/255 |  8/255   |  16/255  | 8/255 | 16/255 | 8/255  | 16/255 |
| :--------: | :------: | :-----: | :--------: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :------: | :------: | :---: | :----: | :----: | :----: |
|   Method   |   Net    |   LP    |   TA(%)    | PGD-10 | PGD-10 | PGD-20 | PGD-20 | PGD-40 | PGD-40 |   AA   |   AA   | DeepFool | DeepFool |  BBA  |  BBA   |  FGSM  |  FGSM  |
|    STD     | ResNet32 |   STD   | **93.270** | 0.000  | 0.000  | 0.000  | 0.000  | 0.000  | 0.000  | 0.000  | 0.000  |  0.019   |  0.000   |   -   |   -    | 21.800 | 14.150 |
|     AT     | ResNet32 |   AT    |   79.420   | 48.300 | 18.410 | 48.440 | 19.280 | 47.460 | 15.500 | 42.990 | 10.920 |  48.700  |  25.500  |   -   |   -    | 53.390 | 35.140 |
|   TRADES   | ResNet32 | TRADES  |   74.470   |        |        | 46.120 |        |        |        |        |        |          |          |   -   |   -    |        |        |
|    STD     | ResNet18 |   STD   |            |        |        |        |        |        |        |        |        |          |          |   -   |   -    |        |        |
|     AT     | ResNet18 |   AT    |   84.780   | 44.660 | 16.649 | 45.450 | 17.530 | 43.210 | 13.670 | 41.400 | 8.490  |  50.230  |  26.670  |   -   |   -    | 53.400 | 35.060 |
|   TRADES   | ResNet18 | TREADES |   81.110   | 51.490 | 24.250 | 51.840 | 25.070 | 50.770 | 22.090 |        |        |  52.280  |  32.580  |   -   |   -    | 55.880 | 38.210 |
|    STD     |  cifar   |   STD   |   91.560   | 0.000  | 0.000  | 0.000  | 0.000  | 0.000  | 0.000  | 0.000  | 0.000  |  0.000   |  0.000   |   -   |   -    | 9.800  | 8.000  |
|     AT     |   ciar   |   AT    |   76.260   | 45.380 | 16.400 | 45.800 | 17.220 | 44.510 | 14.020 | 39.680 | 9.450  |  45.110  |  21.310  |   -   |   -    | 50.620 | 30.500 |
|   TRADES   |  cifar   | TRADES  |   72.960   | 43.200 | 18.670 | 43.410 | 19.050 | 42.650 | 17.420 | 37.600 | 12.210 |  41.720  |  20.510  |   -   |   -    | 46.350 | 26.830 |



#### $\ell_2$



| $\epsilon$ |    -     |    -    |   0    |  0.5   | 0.5  |  0.5  |   0.5    |
| :--------: | :------: | :----: | :----: | :--: | :---: | :------: | :------: |
|   Method   |   Net    |   LP   | TA(%)  | PGD-50 |  Sparse  |  C&W  | DeepFool |
|    STD     | ResNet32 | STD | 93.270 |        |      | 0.000 |          |
|     AT     | ResNet32 | AT | 79.420 | 56.700 | 53.340 |   54.670    |  58.480  |
|   TRADES   | ResNet32 | TRADES | 74.470 |        |      |       |          |
| STD | ResNet18 | STD | | | | | |
| AT | ResNet18 | AT | 84.780 | | | 54.900 | |
| TRADES | ResNet18 | TRADES | 81.110 | | | | |
| STD | cifar | STD | 91.560 | | | | |
| AT | cifar | AT | 76.260 | | | | |
| TRADES | cifar | TRADES | 72.960 | | | | |



#### $\ell_1$

| $\epsilon$ |    -     |   -    |   0    |   12   |   12   |    12    |
| :--------: | :------: | :----: | :----: | :----: | :----: | :------: |
|   Method   |   Net    |   LP   | TA(%)  | PGD-50 | Sparse | DeepFool |
|    STD     | ResNet32 |  STD   | 93.270 |        |        |          |
|     AT     | ResNet32 |   AT   | 79.420 |        |        |          |
|   TRADES   | ResNet32 | TRADES | 74.470 |        |        |          |
|    STD     | ResNet18 |  STD   |        |        |        |          |
|     AT     | ResNet18 |   AT   | 84.780 |        |        |          |
|   TRADES   | ResNet18 | TRADES | 81.110 |        |        |          |
|    STD     |  cifar   |  STD   | 91.560 |        |        |          |
|     AT     |  cifar   |   AT   | 76.260 |        |        |          |
|   TRADES   |  cifar   | TRADES | 72.960 |        |        |          |



### MNIST



#### $\ell_{\infty}$



#### $\ell_2$



#### $\ell_1$