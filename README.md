# FRAGAN
A Feature Refinement and Adaptive Generative Adversarial Network for Thermal Infrared Image Colorization          


## Prerequisites
- python 3.7
- torch 1.13.1
- torchvision 0.14.1
- dominate
- visdom

## Trian
```
python train.py --dataroot [dataset root] --name [experiment_name] --phase train --which_epoch latest
```

## Test
```
python test.py --dataroot [dataset root] --name [experiment_name] --phase test --which_epoch latest
```


## Colorization results
### KAIST dataset
![KAIST](img/KAIST.png)

### FLIR dataset
![FLIR](img/FLIR.png)

## Acknowledgments
This code heavily borrowes from [MUGAN](https://github.com/HangyingLiao/MUGAN).
