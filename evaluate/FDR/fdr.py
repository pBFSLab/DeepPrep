from pathlib import Path
from statsmodels.stats.multitest import fdrcorrection
import nibabel as nib
import numpy as np


"""
https://www.statsmodels.org/dev/generated/statsmodels.stats.multitest.fdrcorrection.html
"""


def fdr_pvalue():
    data_path = Path('/mnt/ngshare/DeepPrep/Validation/UKB_100/v1_feature/recon_interp_fsaverage6_DeepPrep_fmriPrep_pvalue')
    data_fdr_path = Path('/mnt/ngshare/DeepPrep/Validation/UKB_100/v1_feature/recon_interp_fsaverage6_DeepPrep_fmriPrep_pvalue_fdr')
    data_fdr_path.mkdir(parents=True, exist_ok=True)
    for pvalue_path in data_path.iterdir():
        if 'log' in pvalue_path.name:
            continue
        print(f'<<< {pvalue_path}')
        data_in = nib.freesurfer.read_morph_data(str(pvalue_path))
        data_fdr = fdrcorrection(data_in)
        # x = np.power(10, data_in * -1)
        # data_fdr = fdrcorrection(x)
        # data_out = -1 * np.log10(data_fdr)

        data_out = -np.log10(data_fdr[1])
        output_path = data_fdr_path / pvalue_path.name
        nib.freesurfer.write_morph_data(str(output_path), data_out)
        print(f'>>> {output_path}')


if __name__ == '__main__':
    fdr_pvalue()
