set -ex
python test.py --dataroot /home/sys120-1/cy/two_sided/FLIR_wash_day --name FLIR --phase test --which_epoch latest
python test.py --dataroot /home/sys120-1/cy/two_sided/kaist_wash_day --name KAIST --phase test --which_epoch latest

