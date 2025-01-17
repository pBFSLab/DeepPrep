#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------
# @Author : Ning An        @Email : Ning An <ninganme0317@gmail.com>

"""
docker run -it --rm --user $(id -u):$(id -g) --gpus all -p 8501:8501 -v /mnt:/mnt pbfslab/deepprep:24.1.x
"""

import streamlit as st

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

    25.1.0 [What’s new](https://deepprep.readthedocs.io/en/25.1.x/changes.html#what-s-new)


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
    Ren, J.\*, An, N.\*, Lin, C., Zhang, Y., Sun, Z., Zhang, W., Li, S., Guo, N., Cui, W., Hu, Q. Wang, W., Wu, X., Wang, Y., Jiang, T., Satterthwaite T. D., Wang, D. and Liu, H. 2024. DeepPrep: An accelerated, scalable, and robust pipeline for neuroimaging preprocessing empowered by deep learning. *Nature Methods (accepted)*.
    
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
