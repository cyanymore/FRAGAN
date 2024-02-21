set -ex
python train.py --dataroot /home/sys120-1/cy/two_sided/FLIR_wash_day --name FLIR --phase train --which_epoch latest
python train.py --dataroot /home/sys120-1/cy/two_sided/kaist_wash_day --name KAIST --phase train --which_epoch latest
python -m visdom.server
