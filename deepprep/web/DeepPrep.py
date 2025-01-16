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
    DeepPrep: An accelerated, scalable, and robust pipeline for 
    neuroimaging preprocessing empowered by deep learning.
"""
)

st.markdown(
    """
    -----------------
    ### Preprocessing
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

    25.1.0 [What’s new](https://deepprep.readthedocs.io/en/latest/changes.html)

"""
)

st.markdown(
    """
    #### Want to learn more?

    - Check out [DeepPrep document](https://deepprep.readthedocs.io/en/latest/)
    - Ask a question in our [issues page](https://github.com/pBFSLab/DeepPrep)

"""
)

st.markdown(
    """
    #### Citation  
    Ren, J.\*, An, N.\*, Lin, C., Zhang, Y., Sun, Z., Zhang, W., Li, S., Guo, N., Cui, W., Hu, Q. and Wang, W., 2024. *DeepPrep: An accelerated, scalable, and robust pipeline for neuroimaging preprocessing empowered by deep learning*. bioRxiv, pp.2024-03.  
    
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
