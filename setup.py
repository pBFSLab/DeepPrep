#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" DeepPrep setup script """
import setuptools

packages = ['deepprep']

setuptools.setup(
    name="deepprep",
    version='23.1.0',
    author="Ning An, Cong Lin, Youjia Zhang, Zhenyu Sun, Weiwei Wang",
    author_email="ninganme0317@gmail.com, "
                 "lincong8722@gmail.com, "
                 "ireneyou33@gmail.com, "
                 "sun25939789@gmail.com, "
                 "wangweiwei2027@163.com",
    include_package_data=False,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Development Status :: 3 - Alpha",
        "License :: Apache-2.0 License :: "
    ],
    python_requires='>=3.10',
)
