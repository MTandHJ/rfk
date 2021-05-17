


The corresponding trained models could be downloaded [here](https://zenodo.org/record/4669899#.YG2pec_isdU).

## Usage



```
┌── data # the path of data
│	├── mnist
│	└──cifar10
└── rfk
	├── freeplot # for saving image
	├── infos # for saving trained model
	├── logs # logging the curve of loss, accuracy, robutness
	├── models # Architectures
	├── src
		├── attacks.py # 
		├── base.py # Coach, arranging the training procdure
		├── config.py # You can specify the ROOT as the path of training data.
		├── criteria.py # useful criteria of foolbox
		├── dict2obj.py #
		├── loadopts.py # for loading basic configs
		├── loss_zoo.py # The implementations of loss function ...
		└── utils.py # other usful tools
	├── AT.py # adversarial training, Madry A.
	├── auto_attack.py # Croce F.
	├── requirements.txt # requiresments of packages
	├── STD.py # standing training
	├── TRADES.py # TRADES, Zhang H.
	├── transfer_attack.py #
	└── white_box_attack.py # the white-box attacks due to foolbox
```



### Training

#### CIFAR-10


    python STD.py resnet32 cifar10 -lp=STD --epochs=164 -wd=0.0002
    python AT.py resnet32 cifar10 -lp=AT --epochs=200 -wd=0.0002
    python TRADES.py resnet32 cifar10 -lp=TRADES --epochs=76 -wd=0.0002

In particular,

    python STD.py wrn_28_10 cifar10 -lp=STD-wrn --epochs=200 -wd=5e-4

when applying Wide ResNet.

Early stopping against over-fitting:

```
python AT.py resnet32 cifar10 -lp=default --epochs=110 -wd=0.0005
python TRADES.py resnet32 cifar10 -lp=default --epochs=110 -wd=0.0005
```



#### MNIST

```
python STD.py mnist mnist -lp=null --epochs=50 -lr=0.1
python AT.py mnist mnist -lp=null --epochs=84 -lr=0.0001 --optimizer=adam --epsilon=0.3 --steps=40 --stepsize=0.0333333
python TRADES.py mnist mnist -lp=TRADES-M --epochs=100 -lr=0.01 --epsilon=0.3 --steps=40 --stepsize=0.0333333 --leverage=1
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

#### $\ell_{\infty} (\epsilon=0.3)$



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



|                  | PGD-50 | SLIDE |  BBA  |
| :--------------: | :----: | :---: | :---: |
|     stepsize     |  0.05  | 0.05  | 0.001 |
|      steps       |   50   |  50   | 1000  |
|   rel_stepsize   |  0.05  | 0.05  |   -   |
|   abs_stepsize   |  0.5   |  0.5  |   -   |
| initial_stepsize |   -    |   -   |   -   |
|    overshoot     |   -    |   -   |   -   |
|        lr        |   -    |   -   | 0.001 |



## Results



### CIFAR-10

#### $\ell_{\infty}$



| $\epsilon$ |     -     |    -    |   0    | 8/255  | 16/255 | 8/255  | 16/255 | 8/255  | 16/255 | 8/255  | 16/255 |  8/255   |  16/255  | 8/255 | 16/255 | 8/255  | 16/255 |
| :--------: | :-------: | :-----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :------: | :------: | :---: | :----: | :----: | :----: |
|   Method   |    Net    |   LP    | TA(%)  | PGD-10 | PGD-10 | PGD-20 | PGD-20 | PGD-40 | PGD-40 |   AA   |   AA   | DeepFool | DeepFool |  BBA  |  BBA   |  FGSM  |  FGSM  |
|    STD     | ResNet32  |   STD   | 93.270 | 0.000  | 0.000  | 0.000  | 0.000  | 0.000  | 0.000  | 0.000  | 0.000  |  0.019   |  0.000   |   -   |   -    | 21.800 | 14.150 |
|     AT     | ResNet32  |   AT    | 79.420 | 48.300 | 18.410 | 48.440 | 19.280 | 47.460 | 15.500 | 42.990 | 10.920 |  48.700  |  25.500  |   -   |   -    | 53.390 | 35.140 |
|   TRADES   | ResNet32  | TRADES  | 74.040 | 45.590 | 20.420 | 45.900 | 21.060 | 45.110 | 19.000 | 40.780 | 14.490 |  44.730  |  23.800  |   -   |   -    | 48.740 | 29.520 |
|    STD     | ResNet18  |   STD   | 95.280 | 0.030  | 0.000  | 0.010  | 0.000  | 0.000  | 0.000  | 0.000  | 0.000  |  2.030   |  0.030   |   -   |   -    | 37.960 | 24.560 |
|     AT     | ResNet18  |   AT    | 84.780 | 44.660 | 16.649 | 45.450 | 17.530 | 43.210 | 13.670 | 41.400 | 8.490  |  50.230  |  26.670  |   -   |   -    | 53.400 | 35.060 |
|   TRADES   | ResNet18  | TREADES | 81.030 | 51.400 | 23.510 | 51.720 | 24.290 | 50.660 | 21.440 | 47.200 | 16.630 |  52.250  |  31.320  |   -   |   -    | 55.880 | 36.770 |
|    STD     | WRN_28_10 | STD-wrn | 96.070 | 0.230  | 0.080  | 0.070  | 0.030  | 0.020  | 0.000  | 0.000  | 0.000  |  9.630   |  1.220   |   -   |   -    | 48.950 | 32.820 |
|     AT     | WRN_28_10 |   AT    | 86.400 | 47.370 | 22.270 | 48.100 | 23.600 | 48.570 | 17.910 | 44.270 | 9.920  |  53.990  |  36.850  |   -   |   -    | 56.380 | 42.010 |
|   TRADES   | WRN_28_10 | TRADES  | 83.910 | 54.160 | 24.720 | 54.580 | 25.810 | 53.270 | 21.810 | 50.090 | 17.780 |  55.430  |  34.090  |   -   |   -    | 58.940 | 39.680 |
|    STD     |   cifar   |   STD   | 91.560 | 0.000  | 0.000  | 0.000  | 0.000  | 0.000  | 0.000  | 0.000  | 0.000  |  0.000   |  0.000   |   -   |   -    | 9.800  | 8.000  |
|     AT     |   ciar    |   AT    | 76.260 | 45.380 | 16.400 | 45.800 | 17.220 | 44.510 | 14.020 | 39.680 | 9.450  |  45.110  |  21.310  |   -   |   -    | 50.620 | 30.500 |
|   TRADES   |   cifar   | TRADES  | 72.510 | 42.040 | 17.610 | 42.220 | 18.180 | 41.610 | 16.490 | 36.890 | 11.600 |  40.610  |  20.190  |   -   |   -    | 45.220 | 25.660 |



#### $\ell_2$



| $\epsilon$ |    -     |    -    |   0    |  0.5   | 0.5  |  0.5  |   0.5    |   0.5   |
| :--------: | :------: | :----: | :----: | :--: | :---: | :------: | :------: | :------: |
|   Method   |   Net    |   LP   | TA(%)  | PGD-50 |  AA  |  C&W  | DeepFool | BBA |
|    STD     | ResNet32 | STD | 93.270 | 0.010 | 0.000 | 0.000 | 0.660 | - |
|     AT     | ResNet32 | AT | 79.420 | 56.700 | 53.310 |   54.670    |  58.480  |  -  |
|   TRADES   | ResNet32 | TRADES | 74.040 | 54.290 | 51.000 | 51.320 |  54.850  | - |
| STD | ResNet18 | STD | 95.280 | 0.340 | 0.000 | 0.030 | 13.860 | - |
| AT | ResNet18 | AT | 84.780 | 54.970 | 53.800 | 54.920 | 60.530 | - |
| TRADES | ResNet18 | TRADES | 81.030 | 59.900 | 56.850 | 57.420 |  62.040  | - |
| STD | WRN_28_10 | STD-wrn | 96.070 | 0.430 | 0.000 | 0.040 | 33.680 | - |
| AT | WRN_28_10 | AT | 86.400 | 53.950 | 52.680 | 53.600 | 62.600 | - |
| TRADES | WRN_28_10 | TRADES | 83.910 | 59.660 | 56.660 | 57.340 | 63.470 | - |
| STD | cifar | STD | 91.560 | 0.060 | 0.000 | 0.030 | 1.140 | - |
| AT | cifar | AT | 76.260 | 58.180 | 54.850 | 55.750 | 57.780 | - |
| TRADES | cifar | TRADES | 72.510 | 54.720 | 51.360 | 51.580 |  53.740  | - |



#### $\ell_1$



| $\epsilon$ |     -     |    -    |   0    |   12   |   12   |
| :--------: | :-------: | :-----: | :----: | :----: | :----: |
|   Method   |    Net    |   LP    | TA(%)  | PGD-50 | SLIDE  |
|    STD     | ResNet32  |   STD   | 93.270 | 0.620  | 0.690  |
|     AT     | ResNet32  |   AT    | 79.420 | 57.250 | 23.080 |
|   TRADES   | ResNet32  | TRADES  | 70.040 | 54.950 | 25.240 |
|    STD     | ResNet18  |   STD   | 95.280 | 6.600  | 3.210  |
|     AT     | ResNet18  |   AT    | 84.780 | 55.380 | 21.360 |
|   TRADES   | ResNet18  | TRADES  | 81.030 | 59.320 | 27.510 |
|    STD     | WRN_28_10 | STD-wrn | 96.070 | 7.890  | 6.520  |
|     AT     | WRN_28_10 |   AT    | 86.400 | 52.060 | 20.650 |
|   TRADES   | WRN_28_10 | TRADES  | 83.910 | 56.740 | 23.110 |
|    STD     |   cifar   |   STD   | 91.560 | 1.890  | 0.080  |
|     AT     |   cifar   |   AT    | 76.260 | 61.110 | 26.350 |
|   TRADES   |   cifar   | TRADES  | 72.510 | 57.610 | 26.560 |



### MNIST



#### $\ell_{\infty}$



| $\epsilon$ |   -   |    -     |   0    |  0.3   |   0.3   |  0.3   |   0.3    | 0.3  |  0.3   |
| :--------: | :---: | :------: | :----: | :----: | :-----: | :----: | :------: | :--: | :----: |
|   Method   |  Net  |    LP    | TA(%)  | PGD-50 | PGD-100 |   AA   | DeepFool | BBA  |  FGSM  |
|    STD     | mnist |   null   | 99.735 | 0.000  |  0.000  | 0.000  |  0.060   |  -   | 10.990 |
|     AT     | mnist |   null   | 99.400 | 95.860 | 94.990  | 92.590 |  96.460  |  -   | 97.220 |
|   TRADES   | mnist | TRADES-M | 99.450 | 96.150 | 95.240  | 39.950 |  95.860  |  -   | 96.610 |



#### $\ell_2$




| $\epsilon$ |   -   |    -     |   0    |    2    |   2    |    2     |  2   |   2    |
| :--------: | :---: | :------: | :----: | :-----: | :----: | :------: | :--: | :----: |
|   Method   |  Net  |    LP    | TA(%)  | PGD-100 |   AA   | DeepFool | BBA  |  C&W   |
|    STD     | mnist |   null   | 99.735 |  0.450  | 0.000  |  21.950  |  -   | 0.180  |
|     AT     | mnist |   null   | 99.400 | 93.080  | 6.690  |  96.170  |  -   | 62.090 |
|   TRADES   | mnist | TRADES-M | 99.450 | 95.490  | 10.520 |  96.390  |  -   | 84.240 |



#### $\ell_1$



| $\epsilon$ |   -   |    -     |   0    |   10   |   10   |  10  |
| :--------: | :---: | :------: | :----: | :----: | :----: | :--: |
|   Method   |  Net  |    LP    | TA(%)  | PGD-50 | SLIDE  | BBA  |
|    STD     | mnist |   null   | 99.735 | 82.520 | 3.090  |  -   |
|     AT     | mnist |   null   | 99.400 | 96.380 | 78.560 |  -   |
|   TRADES   | mnist | TRADES-M | 99.450 | 96.410 | 94.460 |  -   |

