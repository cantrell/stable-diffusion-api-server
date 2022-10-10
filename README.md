# Stable Diffusion API Server

The local inference REST API server for the [Stable Diffusion Photoshop plugin](https://christiancantrell.com/#ai-ml).

## Requirements

1. Install Python.
1. Install [Conda](https://conda.io/projects/conda/en/latest/user-guide/install/download.html).
1. Download this repo.
1. cd into the repo's directory.
1. Set up a [Conda](https://conda.io) environment named `sd-api-server` using the following commands:

```
% conda env create -f environment.yaml
% conda activate sd-api-server
```

If you are updating the server, make sure to update your Conda environment:

```
% conda env update -f environment.yaml
% conda activate sd-api-server
```

You'll also need [a HuggingFace token](https://huggingface.co/settings/tokens). Paste your token into a `token.txt` file in this directory, save it, and run the server.

## To Run

```
% python3 server.py
```
