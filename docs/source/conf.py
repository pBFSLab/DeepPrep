import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'DeepPrep'
copyright = '2023-2025 pBFS lab, Changping Laboratory All rights reserved'
author = 'deepprep'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [ "sphinx.ext.autodoc",
                "sphinx.ext.doctest",
                "sphinx.ext.intersphinx",
                "sphinx.ext.coverage",
                "sphinx.ext.mathjax",
                "sphinx.ext.linkcode",
                "sphinx.ext.napoleon",
                "sphinxarg.ext",
                "nipype.sphinxext.plot_workflow",
                "sphinx.ext.ifconfig",
                "sphinx.ext.viewcode",
                "sphinx.ext.githubpages",
                "sphinxcontrib.apidoc",
                "nipype.sphinxext.apidoc",
                'sphinx.ext.extlinks',
                'sphinx_toolbox.sidebar_links',
                'sphinx_toolbox.github'
               ]

github_username = 'pBFS'
github_repository = '<pBFS repository>'

# Mock modules in autodoc:
autodoc_mock_imports = [
    "scipy",
    "numpy",
    "nitime",
    "matplotlib",
    "nibabel",
    "ants",
    "tensorflow",
    "voxelmorph"
]
napoleon_custom_sections = [
    ("Inputs", "params_style"),
    ("Outputs", "params_style"),
    ("Source Code", "params_style")

]

templates_path = ['_templates']
# exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
autoclass_content = 'both'
language = 'en'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ['_static']
extensions = ['sphinx.ext.autodoc','sphinx.ext.napoleon','sphinx.ext.viewcode']
# 解析文件格式
source_suffix = {'.rst': 'restructuredtext'}
highlight_options = {'style': 'solarized'}
# Bibliographic Dublin Core info.
epub_title = project
html_copy_source = False
# display logo
html_logo = 'logo_title4.svg'
html_theme_options = {
    'logo_only': True,
    'display_version': False,
}

# The master toctree document.
master_doc = "index"

# -- Options for Texinfo output ----------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (
        master_doc,
        "deepprep",
        author,
        "DeepPrep",
        "One line description of project.",
        "Miscellaneous",
    ),
]
# -- Extension configuration -------------------------------------------------

# apidoc_module_dir = "../deepprep"
# apidoc_output_dir = "api"
# apidoc_excluded_paths = ["model/*"]
# apidoc_separate_modules = True
# apidoc_extra_args = ["--module-first", "-d 1", "-T"]

def setup(app):
    app.add_css_file('custom.css')