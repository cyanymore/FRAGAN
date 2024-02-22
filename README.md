# FRAGAN
A Feature Refinement and Adaptive Generative Adversarial Network for Thermal Infrared Image Colorization          


## Prerequisites
- python 3.7
- torch 1.13.1
- torchvision 0.14.1
- dominate
- visdom

## Pretrianed
We provide the pretrained model: [Google](https://drive.google.com/drive/folders/1KiL0p8NZdV0YyyOdt4Z2TCOy_jz_i0w3)

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


## Note
We only uploaded the paired strategy, the remaining strategies can be found in [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) and [CUT](https://github.com/taesungp/contrastive-unpaired-translation).

## Acknowledgments
This code heavily borrowes from [MUGAN](https://github.com/HangyingLiao/MUGAN).
