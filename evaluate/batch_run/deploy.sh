#!/usr/bin/env bash
sudo apt install nfs-common -y
sudo apt install screen -y

sudo mkdir /mnt/nfs
sudo mount -t nfs 30.30.30.141:/mnt/ngshare/nfs /mnt/nfs
cd /mnt/nfs
pip3 install sh
python3 batch_freesurfer.py