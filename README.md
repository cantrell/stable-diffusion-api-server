# Stable Diffusion API Server

A local inference REST API server for the [Stable Diffusion Photoshop plugin](https://christiancantrell.com/#ai-ml). (Also a generic Stable Diffusion REST API for whatever you want.)

The server will run on Windows and Linux machines with NVIDIA GPUs, and on M1 Macs. M1 Mac support using MPS (Metal Performance Shaders) is highly experimental — and not easy to configure — but it does work, and it will get better over time.

If you can swing it, for best results, use a dedicated Linux box. Performance on Windows is also very good, but I recommend a dedicated machine with no other apps running. You can run Photoshop on the same machine if you have to, but you will be giving up some of your GPU memory which is good for the Photoshop user experience, but bad for optimal local inference.

**Note that this project includes the content safety filter.**

## Installation

### Windows, Linux, and Mac Instructions

1. Install Python.
1. Install [Conda](https://conda.io/projects/conda/en/latest/user-guide/install/download.html).
1. Download this repo.
1. cd into the repo's directory.
1. Set up a [Conda](https://conda.io) environment named `sd-api-server` by running the following command:

Windows and Linux:

```
% conda env create -f environment.yaml
```

M1 Macs:

```
% conda env create -f environment-m1.yaml
```

Then activate the Conda environment:

```
% conda activate sd-api-server
```

If you are updating the server, make sure to update your Conda environment (using the platform-specific `yaml` file):

```
% conda env update -f environment.yaml
% conda activate sd-api-server
```

If you want to remove an old environment and create it from scratch (using the platform-specific `yaml` file):

```
% conda env remove -n sd-api-server
% conda env create -f environment.yaml
% conda activate sd-api-server
```

You'll also need [a HuggingFace token](https://huggingface.co/settings/tokens). Copy your token and paste it into a `token.txt` file in same directory as the repo and save it.

Windows and Linux users, you're good to go! All you have to do now is start the server:

```
% python3 server.py
```

### M1 Mac Additional Instructions

Note that this is highly experimental, and may not work for you. But it will probably get easier with the next release of [PyTorch](https://pytorch.org/).

#### Method 1: Nightly Builds

In Terminal, at the Conda prompt, **with the `sd-api-server` environment activated**:

```
% conda install pytorch torchvision -c pytorch-nightly
% conda deactivate
% conda activate sd-api-server
% python3 server.py
```

You might have noticed that you just installed a nightly build of PyTorch and Torchvision. Nightly builds come with neither warranties nor guarantees. If your server starts and you can generate images, you just won the nightly Lottery! If not, you can play again tomorrow. This is a temporary situation and probably won't be necessary with the next release of PyTorch.

#### Method 2: Environment Variables

If the nightly build didn't work for you — or if you're simply allergic to nightly builds — you can tell PyTorch to use use the CPU in addition to MPS. If you already installed the nightly build, remove your Conda environment using the command above, start all over again, skip the nightly build step, and try this (with the `sd-api-server` environment active):

```
% conda env config vars set PYTORCH_ENABLE_MPS_FALLBACK=1
% conda activate sd-api-server
% python3 server.py
```

Now go be creative!

## REST API

All `POST` requests use the `application/x-www-form-urlencoded` content type, and all images are base64 encoded strings.

`GET /ping`

#### Response

```
{'status':'success'}
```

`POST /txt2img`

Parameters:

- `prompt`: A text description.
- `seed`: A numeric seed.
- `num_outputs`: The number of images you want to get back.
- `width`: The width of your results.
- `height`: The height of your results.
- `num_inference_steps`: The number of steps (more steps mean higher quality).
- `guidance_scale`: Prompt strength.

#### Response

```
{
  'status':'success | failure',
  'message':'Only if there was a failure',
  'images': [
    {
      'base64': 'base64EncodedImage==',
      'seed': 123456789,
      'mimetype': 'image/png',
      'nsfw': true | false
    }
  ]
}
```

`POST /img2img`

Parameters:

- `prompt`: A text description.
- `seed`: A numeric seed.
- `num_outputs`: The number of images you want to get back.
- `num_inference_steps`: The number of steps (more steps mean higher quality).
- `guidance_scale`: Prompt strength.
- `init_image`: The initial input image.
- `strength`: The image strength.

#### Response

```
{
  'status':'success | failure',
  'message':'Only if there was a failure',
  'images': [
    {
      'base64': 'base64EncodedImage==',
      'seed': 123456789,
      'mimetype': 'image/png',
      'nsfw': true | false
    }
  ]
}
```

`POST /masking`

Parameters:

- `prompt`: A text description.
- `seed`: A numeric seed.
- `num_outputs`: The number of images you want to get back.
- `num_inference_steps`: The number of steps (more steps mean higher quality).
- `guidance_scale`: Prompt strength.
- `init_image`: The initial input image.
- `strength`: The image strength.
- `mask_image`: A mask representing the pixels to replace.

#### Response

```
{
  'status':'success | failure',
  'message':'Only if there was a failure',
  'images': [
    {
      'base64': 'base64EncodedImage==',
      'seed': 123456789,
      'mimetype': 'image/png',
      'nsfw': true | false
    }
  ]
}
```
