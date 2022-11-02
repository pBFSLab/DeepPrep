import os
path = '/mnt/ngshare2/DeepPrep_UKB/UKB_Recon'
path = '/mnt/ngshare2/DeepPrep_UKB/UKB_BoldPreprocess'
for root, dirs, files in os.walk(path, topdown=False):
    if 'fsaverage' in root:
        continue
    if len(os.listdir(root)) == 0:
        os.rmdir(root)
