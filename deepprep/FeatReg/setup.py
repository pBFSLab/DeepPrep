import setuptools

packages = ['featreg', 'featreg/utils']


setuptools.setup(
    name="featreg",
    version='v0.1',
    author="",
    author_email="",
    description="",
    packages=packages,
    package_data={
        "utils": [
            "neigh_indices/*.mat",
            "neigh_indices/*.txt",
            "neigh_indices/*.npy",
            "neigh_indices/*.npz",
            "neigh_indices/*.vtk",
                                        ],
    },
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: "
        #        "Operating System :: OS Independent",
    ],
    python_requires='>=3.3',
    # This field lists other packages that your project depends on to run.
    # Any package you put here will be installed by pip when your project is
    # installed, so they must be valid existing projects.
    # For an analysis of "install_requires" vs pip's requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    #    install_requires=['peppercorn'],  # Optional
    install_requires=['numpy', 'scipy'],
)
