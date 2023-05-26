```shell
conda create -n fastsurfer python=3.9
conda activate fastsurfer
pip install nibabel scikit-image h5py pandas open3d simpleitk

cd src
git clone https://github.com/Deep-MI/FastSurfer
cd FastSurfer
./run_fastsurfer.sh --sd $HOME/workdata/FastSurfer/recon --sid NC_15 --t1 $HOME/workdata/DeepPrep/NC_15/upload/NC_15/anat/005/NC_15_mpr005.nii.gz --py /home/anning/miniconda3/envs/app/bin/python3.9 --threads 1 --surfreg
```