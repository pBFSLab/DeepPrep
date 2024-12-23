#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------
# @Author : Ning An        @Email : Ning An <ninganme0317@gmail.com>

import streamlit as st

st.markdown(f'# 🖥️Terminal')

import streamlit as st
from streamlit_ttyd import terminal
import os

# start the ttyd server and display the terminal on streamlit
cmd = st.text_area("cmd:", help="cmd to run in DeepPrep Env")
if st.button("Run", disabled=len(cmd) <= 0):
    st.text("Terminal showing processes running on your system using the top command")
    port = 8601
    os.system(f'kill -9 $(lsof -t -i:{port}) ')

    ttydprocess, port = terminal(cmd=cmd, port=port)

    # info on ttyd port
    st.text(f"ttyd server is running on port : {port}")
