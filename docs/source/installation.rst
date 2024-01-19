------------
Installation
------------

There are two container ways to install DeepPrep：
   * using Docker container technology.or
   * using Singularity container technology.

The deepprep command-line adheres to the BIDS-Apps recommendations for the user interface.
Therefore, the command-line has the following structure: ::
    $ deepprep <input_bids_path> <output_derivatives_path> <analysis_level> <named_options>

The deepprep command-line options are documented in the **Usage Notes** section.

Minimal system requirements
    * Operating system: Ubuntu 20.04 or higher
    * RAM: at least 16 GB
    * Swap space: at least 16 GB
    * Hard disk: at least 20 GB
    * GPU VRAM: at least 24 GB
    * NVIDIA Driver: CUDA Toolkit 11.8 or higher

After successfully downloading and installing the container (docker and singularity), load the image
package into the Docker local warehouse::

    $ sudo docker load --input deepprep_v0.0.14ubuntu22.04.tar.gz

load the image package into the Singularity local warehouse::

    $ Singularity 安装 DeepPrep