```shell
conda create -n fastsurfer python=3.9
conda activate fastsurfer
pip install nibabel scikit-image h5py pandas open3d

git clone https://github.com/Deep-MI/FastSurfer
cd FastSurfer
./run_fastsurfer.sh --sd /mnt/ngshare/DeepPrep/Recon --sid NC_15 --t1 /mnt/ngshare/DeepPrep/NC_15/anat/005/NC_15_mpr005.nii.gz --py python3 --seg_only
```