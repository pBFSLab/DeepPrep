.. include:: links.rst

--------------------------------
FAQ - Frequently Asked Questions
--------------------------------

.. contents::
    :local:
    :depth: 1

================================
1. Run DeepPrep offline
================================

DeepPrep utilizes `TemplateFlow` to manage the downloading of template files. In the event that a required template file for a specific output space is not present in the TemplateFlow cache directory, DeepPrep will automatically download it from the S3 server.

For running DeepPrep in Docker, ensure to bind the TemplateFlow cache directory to the Docker container using the following command:

    $ docker run -v <templateflow_cache_dir_in_host>:/home/deepprep/.cache/templateflow ...

In the case of Singularity, the tool automatically binds the $HOME directory to the Singularity container. Therefore, if the $HOME/.cache/templateflow directory exists, it will be accessible within the container environment without additional configuration.
