# Stable Diffusion API Server

A local inference REST API server for the [Stable Diffusion Photoshop plugin](https://christiancantrell.com/#ai-ml). (Also a generic Stable Diffusion REST API for whatever you want.)

The API server currently supports:

1. Stable Diffusion weights automatically downloaded from Hugging Face.
1. Custom fine-tuned models in the Hugging Face diffusers file format like those created with [DreamBooth](https://github.com/XavierXiao/Dreambooth-Stable-Diffusion).

(Note that loading checkpoint files directly is not currently supported, but you can easily convert `.ckpt` files into the diffusers format using the aptly named [`convert_original_stable_diffusion_to_diffusers.py`](https://github.com/huggingface/diffusers/blob/main/scripts/convert_original_stable_diffusion_to_diffusers.py) script.)

The server will run on Windows and Linux machines with NVIDIA GPUs, and on M1 Macs. M1 Mac support using MPS (Metal Performance Shaders) is highly experimental (and not easy to configure) but it does work, and it will get better over time.

If you can swing it, for best results, use a dedicated Linux box. Performance on Windows is also very good, but I recommend a dedicated machine with no other apps running. You can run Photoshop on the same machine if you have to, but you will be giving up some of your GPU memory which is good for the Photoshop user experience, but bad for optimal local inference.

**Note that this project uses the content safety filter.**

## Installation

🤞 If anyone wants to make a detailed installation video, I would love to embed it right here. 🤞

### Windows, Linux, and Mac Instructions

1. Install Python.
1. Install [Conda](https://conda.io/projects/conda/en/latest/user-guide/install/download.html).
1. Download this repo.
1. cd into the repo's directory.
1. Set up a [Conda](https://conda.io) environment named `sd-api-server` by running the following command:

Windows and Linux:

**(Note that the '%' character below is meant to denote the command prompt; do not include it when copying and pasting.)**

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

### Hugging Face Configuration

There are two things you need to configure with Hugging Face in order to run the Stable Diffusion model locally:

1. You need to [agree to share your username and email address with Hugging Face](https://huggingface.co/CompVis/stable-diffusion-v1-4) in order to access the model.
1. You also need to set up [a Hugging Face token](https://huggingface.co/settings/tokens). Once you've created a read-only token, copy and paste it into the `config.json` file as the value to the `hf_token` key (and don't forget to save the file).

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

If you get the message `ModuleNotFoundError: No module named 'flask'`, it probably means you're using the wrong version of Python. If you used `python3` then try `python`. If you used `python` then try `python3`. (These are the joys of old versions of Python being preinstalled on Macs.)

## Configuring Custom Models

If you want to use the server (and the Photoshop plugin) with custom-trained models, the first thing you need are the custom-trained models themselves. Instructions for how to do so are beyond the scope of this README, but here are some resources:

- [My custom fork of the DreamBooth repo](https://github.com/cantrell/Dreambooth-Stable-Diffusion-Tweaked) (dramatically simplified).
- [A DreamBooth Stable Diffusion Colab notebook](https://colab.research.google.com/github/ShivamShrirao/diffusers/blob/main/examples/dreambooth/DreamBooth_Stable_Diffusion.ipynb) (much easier than training locally).
- [A good YouTube tutorial on using the Colab notebook](https://www.youtube.com/watch?v=FaLTztGGueQ).
- [The original DreamBooth paper](https://arxiv.org/abs/2208.12242).

Loading checkpoint files directly is not currently supported, but you can easily convert `.ckpt` files into the diffusers format using the aptly named [`convert_original_stable_diffusion_to_diffusers.py`](https://github.com/huggingface/diffusers/blob/main/scripts/convert_original_stable_diffusion_to_diffusers.py) script.

Once you have the models trained, the rest is easy. All you have to do is:

1. Replace your `config.json` file with the `config-custom-models.json` template (rename `config-custom-models.json` to `config.json`).
1. Make sure you copy and paste your Hugging Face token into the new `config.json` file.
1. Fill in the `custom_model` array of the config file appropriately.

Here's an explanation of what the key/value pairs mean:

- `model_path`: The full path to the directory which contains the `model_index.json` file (just the directory; don't include the file itself). **Do not** escape spaces, but **do** escape backslashes with backslashes (e.g. `G:\\My Drive\\stable_diffusion_weights\\MyCustomModelOutput`).
- `ui_label`: The name of the model as you want it to appear in the Photoshop plugin.
- `url_path`: A unique, URL-friendly value that will be used as the endpoint path (see the REST API section below).
- `requires_safety_checker`: Whether or not your custom model expects the safety checker. For models in the Hugging Face diffusers file format, this will be true; for models compiled from checkpoint files into the diffusers file format, this will probably be false.

Once your config file is ready, (re)start the server. If the Photoshop plugin is already loaded, you may need to restart it (or you can just click on the 'Reload Plugin' link in the lower right-hand corner of the 'Generate' tab).

Note that the `custom_model` section of the `config.json` file is an array. That means you can include as many custom models as you want. Here's what it should look like for more than one custom-trained model:

```
{
  "hf_token": "your_hugging_face_token",
  "custom_models": [
    {
      "model_path": "/path/to/directory/containing/model_index.json",
      "ui_label": "My First Model",
      "url_path": "my_first_model",
      "requires_safety_checker": true
    }
  ],
  [
    {
      "model_path": "/path/to/another/directory/containing/model_index.json",
      "ui_label": "My Second Model",
      "url_path": "my_second_model",
      "requires_safety_checker": true
    }
  ]
}
```

To see your custom models in the Generate tab of the Stable Diffusion Photoshop plugin, make sure you've configured your local inference server in the API Key tab.

## REST API

Note that all `POST` requests use the `application/x-www-form-urlencoded` content type, and all images are base64 encoded strings.

`GET /ping`

#### Response

```
{'status':'success'}
```

`GET /custom_models`

#### Response

```
[
  {
    "model_path": "/path/to/directory/containing/model_index.json",
    "ui_label": "My First Model",
    "url_path": "my_first_model",
    "requires_safety_checker": "true | false"
  }
], [...]
```

(If no custom models are configured, you will get back an empty array.)

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

`POST /custom/<url_path>`

`url_path` refers to the `url_path` key/value pair you defined in your `config.json` file.

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
