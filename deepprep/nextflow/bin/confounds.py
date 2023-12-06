# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright 2022 The NiPreps Developers <nipreps@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# We support and encourage derived works from this project, please read
# about our expectations at
#
#     https://www.nipreps.org/community/licensing/
#
"""
Handling confounds.

    .. testsetup::

    >>> import os
    >>> import pandas as pd

"""

import nibabel as nb
import numpy as np
import pandas as pd
from nipype import logging
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    File,
    SimpleInterface,
    TraitedSpec,
    isdefined,
    traits,
)
from nipype.utils.filemanip import fname_presuffix
from niworkflows.utils.timeseries import _cifti_timeseries
from niworkflows.viz.plots import fMRIPlot

LOGGER = logging.getLogger('nipype.interface')

def _nifti_timeseries(
    dataset,
    segmentation=None,
    labels=("Ctx GM", "dGM", "WM+CSF", "Cb", "Crown", "RHM"),
    remap_rois=False,
    lut=None,
):
    """Extract timeseries from NIfTI1/2 datasets."""
    dataset = nb.load(dataset) if isinstance(dataset, str) else dataset
    data = dataset.get_fdata(dtype="float32").reshape((-1, dataset.shape[-1]))

    if segmentation is None:
        return data, None

    # Open NIfTI and extract numpy array
    segmentation = nb.load(segmentation) if isinstance(segmentation, str) else segmentation
    segmentation = np.asanyarray(segmentation.dataobj, dtype=int).reshape(-1)

    remap_rois = remap_rois or (len(np.unique(segmentation[segmentation > 0])) > len(labels))

    # Map segmentation
    if remap_rois or lut is not None:
        if lut is None:
            lut = np.full((2036,), 4, dtype="uint8")  # the rest of brain
            lut[0] = 0  # None
            lut[1000:2036] = 1  # Ctx GM
            lut[[10, 11, 12, 13, 17, 18, 26, 28, 31, 49, 50, 51, 52, 53, 54, 58, 60, 63]] = 2    # dGM
            lut[[2, 24, 41]] = 3     # WM+CSF
        # Apply lookup table
        segmentation = lut[segmentation]

    fgmask = segmentation > 0
    segmentation = segmentation[fgmask]
    seg_dict = {}
    for i in np.unique(segmentation):
        seg_dict[labels[i - 1]] = np.argwhere(segmentation == i).squeeze()

    return data[fgmask], seg_dict



class _FMRISummaryInputSpec(BaseInterfaceInputSpec):
    in_nifti = File(exists=True, mandatory=True, desc="input BOLD (4D NIfTI file)")
    in_cifti = File(exists=False, desc="input BOLD (CIFTI dense timeseries)")
    in_segm = File(exists=True, desc="volumetric segmentation corresponding to in_nifti")
    confounds_file = File(exists=True, desc="BIDS' _confounds.tsv file")

    str_or_tuple = traits.Either(
        traits.Str,
        traits.Tuple(traits.Str, traits.Either(None, traits.Str)),
        traits.Tuple(traits.Str, traits.Either(None, traits.Str), traits.Either(None, traits.Str)),
    )
    confounds_list = traits.List(
        str_or_tuple, minlen=1, desc='list of headers to extract from the confounds_file'
    )
    tr = traits.Either(None, traits.Float, usedefault=True, desc='the repetition time')
    drop_trs = traits.Int(0, usedefault=True, desc="dummy scans")


class _FMRISummaryOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='written file path')


class FMRISummary(SimpleInterface):
    """
    Copy the x-form matrices from `hdr_file` to `out_file`.
    """

    input_spec = _FMRISummaryInputSpec
    output_spec = _FMRISummaryOutputSpec

    def _run_interface(self, runtime):
        self._results['out_file'] = fname_presuffix(
            self.inputs.in_nifti, suffix='_fmriplot.svg', use_ext=False, newpath=runtime.cwd
        )

        has_cifti = isdefined(self.inputs.in_cifti)

        # Read input object and create timeseries + segments object
        seg_file = self.inputs.in_segm if isdefined(self.inputs.in_segm) else None
        dataset, segments = _nifti_timeseries(
            nb.load(self.inputs.in_nifti),
            nb.load(seg_file),
            remap_rois=False,
            labels=(
                ("WM+CSF", "Edge") if has_cifti else ("Ctx GM", "dGM", "WM+CSF", "The rest", "Edge", "RHM")
            ),
        )

        # Process CIFTI
        if has_cifti:
            cifti_data, cifti_segments = _cifti_timeseries(nb.load(self.inputs.in_cifti))

            if seg_file is not None:
                # Append WM+CSF and Edge masks
                cifti_length = cifti_data.shape[0]
                dataset = np.vstack((cifti_data, dataset))
                segments = {k: np.array(v) + cifti_length for k, v in segments.items()}
                cifti_segments.update(segments)
                segments = cifti_segments
            else:
                dataset, segments = cifti_data, cifti_segments

        dataframe = pd.read_csv(
            self.inputs.confounds_file,
            sep="\t",
            index_col=None,
            dtype='float32',
            na_filter=True,
            na_values='n/a',
        )

        headers = []
        units = {}
        names = {}

        for conf_el in self.inputs.confounds_list:
            if isinstance(conf_el, (list, tuple)):
                headers.append(conf_el[0])
                if conf_el[1] is not None:
                    units[conf_el[0]] = conf_el[1]

                if len(conf_el) > 2 and conf_el[2] is not None:
                    names[conf_el[0]] = conf_el[2]
            else:
                headers.append(conf_el)

        if not headers:
            data = None
            units = None
        else:
            data = dataframe[headers]

        colnames = data.columns.ravel().tolist()

        for name, newname in list(names.items()):
            colnames[colnames.index(name)] = newname

        data.columns = colnames

        fig = fMRIPlot(
            dataset,
            segments=segments,
            tr=self.inputs.tr,
            confounds=data,
            units=units,
            nskip=self.inputs.drop_trs,
            paired_carpet=has_cifti,
        ).plot()
        fig.savefig(self._results["out_file"], bbox_inches="tight")
        return runtime
