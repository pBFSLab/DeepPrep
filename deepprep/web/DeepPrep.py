#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------
# @Author : Ning An        @Email : Ning An <ninganme0317@gmail.com>

"""
docker run -it --rm --user $(id -u):$(id -g) --gpus all -p 8501:8501 -v /mnt:/mnt pbfslab/deepprep:24.1.x
"""

import streamlit as st
import streamlit.components.v1 as components

st.write("# Welcome to DeepPrep! 👋")

st.markdown(
    """
    **DeepPrep:** An accelerated, scalable, and robust pipeline for 
    neuroimaging preprocessing empowered by deep learning.
    
DeepPrep is a preprocessing pipeline that can flexibly handle anatomical and functional MRI data end-to-end.
It accommodates various sizes, from a single participant to **LARGE-scale datasets**, achieving a **10-fold acceleration** compared to the state-of-the-art pipeline.
    
"""
)

st.markdown(
    """
    -----------------
    ### DeepPrep GUIs
"""
)

st.page_link("pages/1_🚀Preprocessing of T1w & BOLD.py",
    label="👉Preprocessing of T1w & BOLD🔗"
)

st.page_link("pages/2_⚙️Postprocessing of BOLD.py",
    label="👉Postprocessing of BOLD🔗"
)

st.page_link("pages/3_📝Quick QC.py",
    label="👉Quick QC of T1w & BOLD🔗"
)

st.markdown(
    """
    -----------------
"""
)

st.markdown(
    """
    #### Version

    25.1.0 [What’s new](https://deepprep.readthedocs.io/en/25.1.0/changes.html#what-s-new)


"""
)

st.markdown(
    """
    #### Interested in learning more?

    - Check out [DeepPrep document](https://deepprep.readthedocs.io/en/latest/)
    - Submit your question on our [issue page](https://github.com/pBFSLab/DeepPrep)

"""
)

st.markdown(
    """
    #### Citation  
    Ren, J.\*, An, N.\*, Lin, C., Zhang, Y., Sun, Z., Zhang, W., Li, S., Guo, N., Cui, W., Hu, Q. Wang, W., Wu, X., Wang, Y., Jiang, T., Satterthwaite T. D., Wang, D. and Liu, H. (2025). DeepPrep: an accelerated, scalable and robust pipeline for neuroimaging preprocessing empowered by deep learning. Nature Methods. https://doi.org/10.1038/s41592-025-02599-1
    
"""
)

st.markdown(
    """
    #### License  
    Copyright (c) 2023-2025 pBFS lab, Changping Laboratory All rights reserved.

    Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at  

       [http://www.apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0)  

    Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License. 
    
"""
)

st.markdown(
    """
    #### Methods  
We kindly ask to report results preprocessed with this tool using the following boilerplate.

"""
)

components.html(
"""<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=yes" />
  <title>"DeepPrep citation boilerplate"</title>
  <!--[if lt IE 9]>
    <script src="//cdnjs.cloudflare.com/ajax/libs/html5shiv/3.7.3/html5shiv-printshiv.min.js"></script>
  <![endif]-->
      <style>
        body {
            background-color: #f0f0f0;
            font-family: Arial, sans-serif;
            line-height: 1.6;
            padding: 8px;
        }
    </style>
</head>
<body>
<p style="margin-top:0pt; margin-bottom:14pt; text-align:justify; font-size:13pt;"><a name="OLE_LINK55"></a><a name="OLE_LINK56"><span style="font-family:'Times New Roman';">Anatomical and functional imaging data were preprocessed using DeepPrep 25.1.0 (Ren et al., 2025).</span></a></p>
<p style="margin-top:14pt; margin-bottom:14pt; text-align:justify; font-size:13pt;"><strong><span style="font-family:'Times New Roman';">Anatomical Data Preprocessing</span></strong></p>
<p style="margin-top:14pt; margin-bottom:14pt; text-align:justify; font-size:13pt;"><span style="font-family:'Times New Roman';">For participants with multiple T1-weighted (T1w) images, head motion correction was performed using FreeSurfer (v7.2.0, Fischl., 2012). Intensity non-uniformity in the T1w image was corrected using N4 bias field correction (SimpleITK, v2.3.0, Beare et al., 2018), and the processed T1w image served as the anatomical reference throughout the pipeline. Skull stripping and brain tissue segmentation&mdash;including cerebrospinal fluid (CSF), white matter (WM), and gray matter (GM)&mdash;were performed using&nbsp;</span><em><span style="font-family:'Times New Roman';">FastSurferCNN </span></em><span style="font-family:'Times New Roman';">(FastSurfer v1.1.0, Henschel et al., 2020). Cortical surface reconstruction was conducted using FastCSR v1.0.0 (Ren et al., 2022) and aligned to standard surface templates (</span><em><span style="font-family:'Times New Roman';">fsaverage</span></em><span style="font-family:'Times New Roman';">) via SUGAR v1.0.0 (Ren et al., 2024).&nbsp;</span><a name="OLE_LINK50"></a><a name="OLE_LINK49"><span style="font-family:'Times New Roman';">Spatial normalization to a standard space (</span><em><span style="font-family:'Times New Roman';">MNI152NLin6Asym</span></em><span style="font-family:'Times New Roman';">)</span></a><span style="font-family:'Times New Roman';">&nbsp;was achieved through nonlinear registration with SynthMorph v2 (Hoffman et al., 2024).</span></p>
<p style="margin-top:14pt; margin-bottom:14pt; text-align:justify; font-size:13pt;"><strong><span style="font-family:'Times New Roman';">Functional Data Preprocessing</span></strong></p>
<p style="margin-top:14pt; margin-bottom:14pt; text-align:justify; font-size:13pt;"><span style="font-family:'Times New Roman';">A reference volume was generated by averaging nonsteady-state volumes (dummy scans). Head motion parameters were estimated using&nbsp;</span><em><span style="font-family:'Times New Roman';">mcflirt </span></em><span style="font-family:'Times New Roman';">(FSL v6.0.5.1, Woolrich et al., 2001), and slice-timing correction was applied using&nbsp;</span><em><span style="font-family:'Times New Roman';">3dTshift </span></em><span style="font-family:'Times New Roman';">(AFNI, v24.0.00, Cox and Hyde., 1997) when slice-timing information was available in the BIDS metadata. If fieldmaps were available, susceptibility distortion correction was performed using SDCflow v2.8.1 (Esteban et al., 2020), with the estimated fieldmap rigidly aligned to the reference volume. The BOLD reference was co-registered to the T1w reference using boundary-based registration (</span><em><span style="font-family:'Times New Roman';">bbregister </span></em><span style="font-family:'Times New Roman';">from FreeSurfer; Greve and Fischl, 2009). Preprocessed BOLD timeseries were resampled into both volumetric (</span><em><span style="font-family:'Times New Roman';">MNI152NLin6Asym</span></em><span style="font-family:'Times New Roman';">) and surface templates (</span><em><span style="font-family:'Times New Roman';">fsaverage6</span></em><span style="font-family:'Times New Roman';">) through a single interpolation step, incorporating head motion transformations, susceptibility distortion correction (if applicable), and anatomical co-registration. Volumetric resampling was conducted using&nbsp;</span><em><span style="font-family:'Times New Roman';">DeepPrep&rsquo;s</span></em><span style="font-family:'Times New Roman';">&nbsp;custom methodology, while surface resampling was performed using&nbsp;</span><em><span style="font-family:'Times New Roman';">mri_vol2surf </span></em><span style="font-family:'Times New Roman';">(FreeSurfer).</span></p>
<p style="margin-top:14pt; margin-bottom:14pt; text-align:justify; font-size:13pt;"><span style="font-family:'Times New Roman';">Several confounding time series were extracted from the preprocessed BOLD data:</span></p>
<p style="margin-top:14pt; margin-bottom:14pt; text-align:justify; font-size:13pt;"><a name="OLE_LINK51"></a><a name="OLE_LINK52"><span style="font-family:'Times New Roman';">&bull;&nbsp;</span></a><span style="font-family:'Times New Roman';">Motion-related confounds: Framewise displacement (FD) was computed using both Power&rsquo;s (absolute sum of relative motion; Power et al., 2014) and Jenkinson&rsquo;s (relative root mean square displacement; Jenkinson et al., 2002) formulations. FD and DVARS were calculated for each functional run using their respective implementations in Nipype (Power et al., 2014).</span></p>
<p style="margin-top:14pt; margin-bottom:14pt; text-align:justify; font-size:13pt;"><a name="OLE_LINK53"></a><a name="OLE_LINK54"><span style="font-family:'Times New Roman';">&bull;&nbsp;</span></a><span style="font-family:'Times New Roman';">Global signal regression: Mean time series were extracted from the CSF, WM, and whole-brain masks.</span></p>
<p style="margin-top:14pt; margin-bottom:14pt; text-align:justify; font-size:13pt;"><span style="font-family:'Times New Roman';">&bull; Physiological noise correction: Component-based noise correction (CompCor; Behzadi et al., 2007) was performed using three variants: Temporal CompCor (tCompCor): Principal components were derived from the top 2% most variable voxels within the brain mask after high-pass filtering (128s cutoff). Principal components were retained until they explained 50% of the variance within the nuisance mask. Anatomical CompCor (aCompCor): Probabilistic masks for CSF, WM, and combined CSF+WM (generated from FastSurfer&rsquo;s segmentation) were resampled into BOLD space and binarized at a threshold of 0.99. Principal components were retained until they explained 50% of the variance within each nuisance mask. Background CompCor (bCompCor): BOLD timeseries from non-brain regions within the background of the field of view were extracted, and the top 10 principal components were retained by default.</span></p>
<p style="margin-top:14pt; margin-bottom:14pt; text-align:justify; font-size:13pt;"><span style="font-family:'Times New Roman';">To account for higher-order motion artifacts, temporal derivatives and quadratic terms of motion estimates and global signals were computed (Satterthwaite et al., 2013). Motion outliers were defined as frames exceeding an FD threshold of 0.5 mm or standardized DVARS &gt;1.5. A final nuisance regression step incorporated principal component analysis of signal from a thin band of voxels along the brain&rsquo;s edge (Patriat, Reynolds, and Birn, 2017) to further mitigate motion-related artifacts.</span></p>
<p style="margin-top:14pt; margin-bottom:14pt; text-align:justify; font-size:13pt;"><strong><span style="font-family:'Times New Roman';">Copyright Waiver</span></strong></p>
<p style="margin-top:14pt; margin-bottom:14pt; text-align:justify;"><span style="font-family:'Times New Roman'; font-size:13pt;">The above boilerplate text was automatically generated by DeepPrep with the express intention that users could copy and paste this text into their manuscripts</span><span style="font-family:'Times New Roman'; font-size:13pt;">&nbsp;</span><em><span style="font-family:'Times New Roman'; font-size:13pt;">unchanged</span></em><span style="font-family:'Times New Roman'; font-size:13pt;">. It is released under the</span><span style="font-family:'Times New Roman'; font-size:13pt;">&nbsp;</span><a href="https://creativecommons.org/publicdomain/zero/1.0/" style="text-decoration:none;"><u><span style="font-family:'Times New Roman'; font-size:13pt; color:#0000ff;">CC0</span></u></a><span style="font-family:'Times New Roman'; font-size:13pt;">&nbsp;</span><span style="font-family:'Times New Roman'; font-size:13pt;">license.</span></p>
<p style="margin-top:14pt; margin-bottom:14pt; text-align:justify; font-size:13pt;"><strong><span style="font-family:'Times New Roman';">References</span></strong></p>
<p style="margin-top:0pt; margin-bottom:0pt;"><span style="font-family:'Times New Roman'; font-size:13pt;">Beare, R., Lowekamp, B. C., Yaniv, Z. 2018. &ldquo;Image Segmentation, Registration and Characterization in R with SimpleITK&rdquo;,</span><em><span style="font-family:'Times New Roman'; font-size:13pt;"> J Stat Softw</span></em><span style="font-family:'Times New Roman'; font-size:13pt;">, 86(8),</span><span style="font-family:'Times New Roman'; font-size:13pt;">&nbsp;</span><a href="https://doi.org/10.18637/jss.v086.i08" style="text-decoration:none;"><span style="font-family:'Times New Roman'; font-size:13pt; color:#000000;">https://doi.org/10.18637/jss.v086.i08</span></a><span style="font-family:'Times New Roman'; font-size:13pt;">.</span></p>
<p style="margin-top:0pt; margin-bottom:0pt;"><span style="font-family:'Times New Roman'; font-size:13pt;">Behzadi, Yashar, Khaled Restom, Joy Liau, and Thomas T. Liu. 2007.</span><span style="font-family:'Times New Roman'; font-size:13pt;">&nbsp;</span><span style="font-family:'Times New Roman'; font-size:13pt;">&ldquo;A Component Based Noise Correction Method (CompCor) for</span><span style="font-family:'Times New Roman'; font-size:13pt;">&nbsp;</span><span style="font-family:'Times New Roman'; font-size:13pt;">BOLD</span><span style="font-family:'Times New Roman'; font-size:13pt;">&nbsp;</span><span style="font-family:'Times New Roman'; font-size:13pt;">and Perfusion Based fMRI.&rdquo;</span><span style="font-family:'Times New Roman'; font-size:13pt;">&nbsp;</span><em><span style="font-family:'Times New Roman'; font-size:13pt;">NeuroImage</span></em><span style="font-family:'Times New Roman'; font-size:13pt;">&nbsp;</span><span style="font-family:'Times New Roman'; font-size:13pt;">37 (1): 90&ndash;101.</span><span style="font-family:'Times New Roman'; font-size:13pt;">&nbsp;</span><a href="https://doi.org/10.1016/j.neuroimage.2007.04.042" style="text-decoration:none;"><u><span style="font-family:'Times New Roman'; font-size:13pt; color:#0000ff;">https://doi.org/10.1016/j.neuroimage.2007.04.042</span></u></a><span style="font-family:'Times New Roman'; font-size:13pt;">.</span></p>
<p style="margin-top:0pt; margin-bottom:0pt; font-size:13pt;"><span style="font-family:'Times New Roman';">Cox, R. W. and Hyde, J. S. 1997. &ldquo;Software tools for analysis and visualization of fMRI data.&rdquo;&nbsp;</span><em><span style="font-family:'Times New Roman';">NMR Biomed</span></em><span style="font-family:'Times New Roman';">&nbsp;10, 171-178.&nbsp;</span><a href="https://doi.org:10.1002/(sici)1099-1492(199706/08)10:4/5%3C171::aid-nbm453%3E3.0.co;2-l" style="text-decoration:none;"><u><span style="font-family:'Times New Roman'; color:#800080;">https://doi.org:10.1002/(sici)1099-1492(199706/08)10:4/5&lt;171::aid-nbm453&gt;3.0.co;2-l</span></u></a><span style="font-family:'Times New Roman';">.</span></p>
<p style="margin-top:0pt; margin-bottom:0pt; font-size:13pt;"><span style="font-family:'Times New Roman';">Esteban, O., Markiewicz, C., Blair, R., Poldrack, R. and Gorgolewski, K. 2020. &ldquo;SDCflows: Susceptibility distortion correction workFLOWS.&rdquo;&nbsp;</span><em><span style="font-family:'Times New Roman';">Zenodo. </span></em><a href="https://doi.org/10.5281/zenodo.3758524" style="text-decoration:none;"><u><span style="font-family:'Times New Roman'; color:#800080;">https://doi.org/10.5281/zenodo.3758524</span></u></a><span style="font-family:'Times New Roman';">.&nbsp;</span></p>
<p style="margin-top:0pt; margin-bottom:0pt; font-size:13pt;"><span style="font-family:'Times New Roman';">Fischl, B. 2012. &ldquo;FreeSurfer.&rdquo;&nbsp;</span><em><span style="font-family:'Times New Roman';">Neuroimage</span></em><span style="font-family:'Times New Roman';">&nbsp;</span><strong><span style="font-family:'Times New Roman';">62</span></strong><span style="font-family:'Times New Roman';">, 774-781.&nbsp;</span><a href="https://doi.org:10.1016/j.neuroimage.2012.01.021" style="text-decoration:none;"><u><span style="font-family:'Times New Roman'; color:#800080;">https://doi.org:10.1016/j.neuroimage.2012.01.021</span></u></a><u><span style="font-family:'Times New Roman'; color:#0000ff;">.</span></u></p>
<p style="margin-top:0pt; margin-bottom:0pt;"><span style="font-family:'Times New Roman'; font-size:13pt;">Greve, Douglas N, and Bruce Fischl. 2009.</span><span style="font-family:'Times New Roman'; font-size:13pt;">&nbsp;</span><span style="font-family:'Times New Roman'; font-size:13pt;">&ldquo;Accurate and Robust Brain Image Alignment Using Boundary-Based Registration.&rdquo;</span><span style="font-family:'Times New Roman'; font-size:13pt;">&nbsp;</span><em><span style="font-family:'Times New Roman'; font-size:13pt;">NeuroImage</span></em><span style="font-family:'Times New Roman'; font-size:13pt;">&nbsp;</span><span style="font-family:'Times New Roman'; font-size:13pt;">48 (1): 63&ndash;72.</span><span style="font-family:'Times New Roman'; font-size:13pt;">&nbsp;</span><a href="https://doi.org/10.1016/j.neuroimage.2009.06.060" style="text-decoration:none;"><u><span style="font-family:'Times New Roman'; font-size:13pt; color:#0000ff;">https://doi.org/10.1016/j.neuroimage.2009.06.060</span></u></a><span style="font-family:'Times New Roman'; font-size:13pt;">.</span></p>
<p style="margin-top:0pt; margin-bottom:0pt; font-size:13pt;"><span style="font-family:'Times New Roman';">Henschel, L., Conjeti, S., Estrada, S., Diers, K., Fischl, B., &amp; Reuter, M. (2020). &ldquo;Fastsurfer-a fast and accurate deep learning based neuroimaging pipeline.&rdquo;</span><span style="font-family:'Times New Roman';">&nbsp;</span><em><span style="font-family:'Times New Roman';">NeuroImage</span></em><span style="font-family:'Times New Roman';">,</span><span style="font-family:'Times New Roman';">&nbsp;</span><span style="font-family:'Times New Roman';">219, 117012.&nbsp;</span><a href="https://doi.org:10.1016/j.neuroimage.2020.117012" style="text-decoration:none;"><u><span style="font-family:'Times New Roman'; color:#800080;">https://doi.org:10.1016/j.neuroimage.2020.117012</span></u></a><span style="font-family:'Times New Roman';">.</span></p>
<p style="margin-top:0pt; margin-bottom:0pt; font-size:13pt;"><span style="font-family:'Times New Roman';">Hoffmann, M., Hoopes, A., Greve, D. N., Fischl, B. and Dalca, A. V. 2024. &ldquo;Anatomy-aware and acquisition-agnostic joint registration with SynthMorph.&rdquo;</span><em><span style="font-family:'Times New Roman';"> Imaging Neuroscience</span></em><span style="font-family:'Times New Roman';">&nbsp;2, 1-33.&nbsp;</span><a href="https://doi.org:10.1016/j.neuroimage.2020.117012" style="text-decoration:none;"><u><span style="font-family:'Times New Roman'; color:#800080;">https://doi.org:10.1162/imag_a_00197</span></u></a><span style="font-family:'Times New Roman';">.</span></p>
<p style="margin-top:0pt; margin-bottom:0pt;"><span style="font-family:'Times New Roman'; font-size:13pt;">Jenkinson, Mark, Peter Bannister, Michael Brady, and Stephen Smith. 2002.</span><span style="font-family:'Times New Roman'; font-size:13pt;">&nbsp;</span><span style="font-family:'Times New Roman'; font-size:13pt;">&ldquo;Improved Optimization for the Robust and Accurate Linear Registration and Motion Correction of Brain Images.&rdquo;</span><span style="font-family:'Times New Roman'; font-size:13pt;">&nbsp;</span><em><span style="font-family:'Times New Roman'; font-size:13pt;">NeuroImage</span></em><span style="font-family:'Times New Roman'; font-size:13pt;">&nbsp;</span><span style="font-family:'Times New Roman'; font-size:13pt;">17 (2): 825&ndash;41.</span><span style="font-family:'Times New Roman'; font-size:13pt;">&nbsp;</span><a href="https://doi.org/10.1006/nimg.2002.1132" style="text-decoration:none;"><u><span style="font-family:'Times New Roman'; font-size:13pt; color:#0000ff;">https://doi.org/10.1006/nimg.2002.1132</span></u></a><span style="font-family:'Times New Roman'; font-size:13pt;">.</span></p>
<p style="margin-top:0pt; margin-bottom:0pt;"><span style="font-family:'Times New Roman'; font-size:13pt;">Patriat, R&eacute;mi, Richard C. Reynolds, and Rasmus M. Birn. 2017.</span><span style="font-family:'Times New Roman'; font-size:13pt;">&nbsp;</span><span style="font-family:'Times New Roman'; font-size:13pt;">&ldquo;An Improved Model of Motion-Related Signal Changes in</span><span style="font-family:'Times New Roman'; font-size:13pt;">&nbsp;</span><span style="font-family:'Times New Roman'; font-size:13pt;">fMRI.&rdquo;</span><span style="font-family:'Times New Roman'; font-size:13pt;">&nbsp;</span><em><span style="font-family:'Times New Roman'; font-size:13pt;">NeuroImage</span></em><span style="font-family:'Times New Roman'; font-size:13pt;">&nbsp;</span><span style="font-family:'Times New Roman'; font-size:13pt;">144, Part A (January): 74&ndash;82.</span><span style="font-family:'Times New Roman'; font-size:13pt;">&nbsp;</span><a href="https://doi.org/10.1016/j.neuroimage.2016.08.051" style="text-decoration:none;"><u><span style="font-family:'Times New Roman'; font-size:13pt; color:#0000ff;">https://doi.org/10.1016/j.neuroimage.2016.08.051</span></u></a><span style="font-family:'Times New Roman'; font-size:13pt;">.</span></p>
<p style="margin-top:0pt; margin-bottom:0pt;"><span style="font-family:'Times New Roman'; font-size:13pt;">Power, Jonathan D., Anish Mitra, Timothy O. Laumann, Abraham Z. Snyder, Bradley L. Schlaggar, and Steven E. Petersen. 2014.</span><span style="font-family:'Times New Roman'; font-size:13pt;">&nbsp;</span><span style="font-family:'Times New Roman'; font-size:13pt;">&ldquo;Methods to Detect, Characterize, and Remove Motion Artifact in Resting State fMRI.&rdquo;</span><span style="font-family:'Times New Roman'; font-size:13pt;">&nbsp;</span><em><span style="font-family:'Times New Roman'; font-size:13pt;">NeuroImage</span></em><span style="font-family:'Times New Roman'; font-size:13pt;">&nbsp;</span><span style="font-family:'Times New Roman'; font-size:13pt;">84 (Supplement C): 320&ndash;41.</span><span style="font-family:'Times New Roman'; font-size:13pt;">&nbsp;</span><a href="https://doi.org/10.1016/j.neuroimage.2013.08.048" style="text-decoration:none;"><u><span style="font-family:'Times New Roman'; font-size:13pt; color:#0000ff;">https://doi.org/10.1016/j.neuroimage.2013.08.048</span></u></a><span style="font-family:'Times New Roman'; font-size:13pt;">.</span></p>
<p style="margin-top:0pt; margin-bottom:0pt; font-size:13pt;"><span style="font-family:'Times New Roman';">Ren, J., Hu, Q., Wang, W., Zhang, W., Hubbard, C. S., Zhang, P., An, N., Zhou, Y., Dahmani, L., Wang, D., Fu, X., Sun, Z., Wang, Y., Wang, R., Li, L., and Liu, H. 2022. &ldquo;Fast cortical surface reconstruction from MRI using deep learning.&rdquo;&nbsp;</span><em><span style="font-family:'Times New Roman';">Brain informatics</span></em><span style="font-family:'Times New Roman';">, 9(1), 6.</span><span style="font-family:'Times New Roman';">&nbsp;</span><a href="https://doi.org/10.1186/s40708-022-00155-7" style="text-decoration:none;"><u><span style="font-family:'Times New Roman'; color:#800080;">https://doi.org/10.1186/s40708-022-00155-7</span></u></a><span style="font-family:'Times New Roman';">.</span></p>
<p style="margin-top:0pt; margin-bottom:0pt; font-size:13pt;"><span style="font-family:'Times New Roman';">Ren, J., An, N., Zhang, Y., Wang, D., Sun, Z., Lin, C., Cui, W., Zhou, Y., Zhang, W., Hu, Q., Zhang, P., Hu, D., Wang, D., and Liu, H. 2024. &ldquo;SUGAR: Spherical ultrafast graph attention framework for cortical surface registration.&rdquo;</span><span style="font-family:'Times New Roman';">&nbsp;</span><em><span style="font-family:'Times New Roman';">Medical Image Analysis,</span></em><span style="font-family:'Times New Roman';">&nbsp;</span><span style="font-family:'Times New Roman';">94, 103122.&nbsp;</span><a href="https://doi.org/10.1016/j.media.2024.103122" style="text-decoration:none;"><u><span style="font-family:'Times New Roman'; color:#800080;">https://doi.org/10.1016/j.media.2024.103122</span></u></a><span style="font-family:'Times New Roman';">.</span></p>
<p style="margin-top:0pt; margin-bottom:0pt; font-size:13pt;"><span style="font-family:'Times New Roman';">Ren, J., An, N., Lin, C., Zhang, Y., Sun, Z., Zhang, W., Li, S., Guo, N., Cui, W., Hu, Q. Wang, W., Wu, X., Wang, Y., Jiang, T., Satterthwaite T. D., Wang, D. and Liu, H. 2025. &ldquo;DeepPrep: an accelerated, scalable and robust pipeline for neuroimaging preprocessing empowered by deep learning.&rdquo;&nbsp;</span><em><span style="font-family:'Times New Roman';">Nature Methods. </span></em><a href="https://doi.org/10.1038/s41592-025-02599-1" style="text-decoration:none;"><u><span style="font-family:'Times New Roman'; color:#800080;">https://doi.org/10.1038/s41592-025-02599-1</span></u></a><span style="font-family:'Times New Roman';">.</span></p>
<p style="margin-top:0pt; margin-bottom:0pt;"><span style="font-family:'Times New Roman'; font-size:13pt;">Satterthwaite, Theodore D., Mark A. Elliott, Raphael T. Gerraty, Kosha Ruparel, James Loughead, Monica E. Calkins, Simon B. Eickhoff, et al. 2013.</span><span style="font-family:'Times New Roman'; font-size:13pt;">&nbsp;</span><span style="font-family:'Times New Roman'; font-size:13pt;">&ldquo;An improved framework for confound regression and filtering for control of motion artifact in the preprocessing of resting-state functional connectivity data.&rdquo;</span><span style="font-family:'Times New Roman'; font-size:13pt;">&nbsp;</span><em><span style="font-family:'Times New Roman'; font-size:13pt;">NeuroImage</span></em><span style="font-family:'Times New Roman'; font-size:13pt;">&nbsp;</span><span style="font-family:'Times New Roman'; font-size:13pt;">64 (1): 240&ndash;56.</span><span style="font-family:'Times New Roman'; font-size:13pt;">&nbsp;</span><a href="https://doi.org/10.1016/j.neuroimage.2012.08.052" style="text-decoration:none;"><u><span style="font-family:'Times New Roman'; font-size:13pt; color:#0000ff;">https://doi.org/10.1016/j.neuroimage.2012.08.052</span></u></a><span style="font-family:'Times New Roman'; font-size:13pt;">.</span></p>
<p style="margin-top:0pt; margin-bottom:0pt; font-size:13pt;"><span style="font-family:'Times New Roman';">Woolrich, M. W., Ripley, B. D., Brady, M. and Smith, S. M. 2001. Temporal autocorrelation in univariate linear modeling of FMRI data.&nbsp;</span><em><span style="font-family:'Times New Roman';">Neuroimage </span></em><span style="font-family:'Times New Roman';">14, 1370-1386.&nbsp;</span><a href="https://doi.org:10.1006/nimg.2001.0931" style="text-decoration:none;"><u><span style="font-family:'Times New Roman'; color:#800080;">https://doi.org:10.1006/nimg.2001.0931</span></u></a><span style="font-family:'Times New Roman';">.</span></p>
</body>""", height=3300
)