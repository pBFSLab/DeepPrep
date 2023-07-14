#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" deepprep setup script """
import setuptools

packages = ['deepprep']

setuptools.setup(
    name="deepprep",
    version='v0.0.1',
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
        "License :: OSI Approved :: "
    ],
    python_requires='>=3.3',
    install_requires=['numpy', 'scipy'],
)
