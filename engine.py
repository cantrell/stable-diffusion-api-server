import re
import sys
import flask
import torch
import diffusers
import requests
import json
from PIL import Image

from utils import retrieve_param, pil_to_b64, b64_to_pil, get_compute_platform

class Engine(object):
    def __init__(self):
        pass

    def process(self, kwargs):
        return []

class ProxyEngine(Engine):
    def __init__(self, base_url):
        super().__init__()
        self.base_url = base_url

    def process(self, url, args_dict):
        response = requests.post(url, json=args_dict)
        if response.status_code != 200:
            raise RuntimeError(response.content)
        return response

    def run(self):
        pass


class EngineStableDiffusion(Engine):
    def __init__(self, pipe, sibling=None, custom_model_path=None, requires_safety_checker=True):
        super().__init__()
        if sibling == None:
            self.engine = pipe.from_pretrained( 'runwayml/stable-diffusion-v1-5', use_auth_token=hf_token.strip() )
        elif custom_model_path:
            if requires_safety_checker:
                self.engine = diffusers.StableDiffusionPipeline.from_pretrained(custom_model_path,
                                                                                safety_checker=sibling.engine.safety_checker,
                                                                                feature_extractor=sibling.engine.feature_extractor)
            else:
                self.engine = diffusers.StableDiffusionPipeline.from_pretrained(custom_model_path,
                                                                                feature_extractor=sibling.engine.feature_extractor)
        else:
            self.engine = pipe(
                vae=sibling.engine.vae,
                text_encoder=sibling.engine.text_encoder,
                tokenizer=sibling.engine.tokenizer,
                unet=sibling.engine.unet,
                scheduler=sibling.engine.scheduler,
                safety_checker=sibling.engine.safety_checker,
                feature_extractor=sibling.engine.feature_extractor
            )
        self.engine.to( get_compute_platform('engine') )

    def process(self, kwargs):
        output = self.engine( **kwargs )
        return {'image': output.images[0], 'nsfw': output.nsfw_content_detected[0]}

    def prepare_args(self, task):
        seed = retrieve_param( 'seed', flask.request.form, int, 0 )
        prompt = flask.request.form[ 'prompt' ]

        args_dict = {
            'prompt' : [ prompt ],
            'seed': seed,
            'num_inference_steps' : retrieve_param( 'num_inference_steps', flask.request.form, int,   100 ),
            'guidance_scale' : retrieve_param( 'guidance_scale', flask.request.form, float, 7.5 ),
            'eta' : retrieve_param( 'eta', flask.request.form, float, 0.0 )
        }

        if (task == 'txt2img'):
            args_dict[ 'width' ] = retrieve_param( 'width', flask.request.form, int,   512 )
            args_dict[ 'height' ] = retrieve_param( 'height', flask.request.form, int,   512 )
        if (task == 'img2img' or task == 'masking'):
            init_img_b64 = flask.request.form[ 'init_image' ]
            init_img_b64 = re.sub( '^data:image/png;base64,', '', init_img_b64 )
            init_img_pil = b64_to_pil( init_img_b64 )
            args_dict[ 'init_image' ] = init_img_pil
            args_dict[ 'strength' ] = retrieve_param( 'strength', flask.request.form, float, 0.7 )
        if (task == 'masking'):
            mask_img_b64 = flask.request.form[ 'mask_image' ]
            mask_img_b64 = re.sub( '^data:image/png;base64,', '', mask_img_b64 )
            mask_img_pil = b64_to_pil( mask_img_b64 )
            args_dict[ 'mask_image' ] = mask_img_pil
        return args_dict

      
    def run(self, task):
        total_results = []
        output_data = {}
        count = retrieve_param( 'num_outputs', flask.request.form, int, 1 )
        for i in range( count ):
            args_dict = self.prepare_args(task)
            if (args_dict['seed'] == 0):
                generator = torch.Generator( device=get_compute_platform('generator') )
            else:
                generator = torch.Generator( device=get_compute_platform('generator') ).manual_seed( args_dict['seed'] )
            args_dict['generator'] = generator
            new_seed = generator.seed()
            # Perform inference:
            pipeline_output = self.process( args_dict )
            pipeline_output[ 'seed' ] = new_seed
            total_results.append( pipeline_output )
        # Prepare response
        output_data[ 'status' ] = 'success'
        images = []
        for result in total_results:
            images.append({
                'base64' : pil_to_b64( result['image'].convert( 'RGB' ) ),
                'seed' : result['seed'],
                'mime_type': 'image/png',
                'nsfw': result['nsfw']
            })
        output_data[ 'images' ] = images
        return output_data

class A1111EngineStableDiffusion(ProxyEngine):
    def prepare_args(self, task):
        args_dict = {
            'prompt' : flask.request.form[ 'prompt' ],
            'steps' : retrieve_param( 'num_inference_steps', flask.request.form, int,   100 ),
            'cfg_scale' : retrieve_param( 'guidance_scale', flask.request.form, float, 7.5 ),
            'eta' : retrieve_param( 'eta', flask.request.form, float, 0.0 ),
            'n_iter': retrieve_param( 'num_outputs', flask.request.form, int,   1 ),
            'seed': retrieve_param( 'seed', flask.request.form, int, -1 )
        }
        
        if (task == 'txt2img'):
            args_dict[ 'width' ] = retrieve_param( 'width', flask.request.form, int,   512 )
            args_dict[ 'height' ] = retrieve_param( 'height', flask.request.form, int,   512 )
            self.endpoint_url = '/sdapi/v1/txt2img'
        if (task == 'img2img' or task == 'masking'):
            init_img_b64 = flask.request.form[ 'init_image' ]
            init_img_b64 = 'data:image/png;base64,' + init_img_b64 if init_img_b64[0:4] != 'data' else mask_img_b64
            args_dict[ 'init_images' ] = (init_img_b64,)
            args_dict[ 'denoising_strength' ] = 1.0 - retrieve_param( 'strength', flask.request.form, float, 0.7 )
            self.endpoint_url = '/sdapi/v1/img2img'
        if (task == 'masking'):
            mask_img_b64 = flask.request.form[ 'mask_image' ]
            mask_img_b64 = 'data:image/png;base64,' + mask_img_b64 if mask_img_b64[0:4] != 'data' else mask_img_b64
            args_dict[ 'mask' ] = mask_img_b64
        return args_dict

    def run(self, task):
        total_results = []
        output_data = {}
        args_dict = self.prepare_args(task)
        response = self.process(self.base_url + self.endpoint_url, args_dict)
        if response.status_code != 200:
            raise RuntimeError(response.text)
        output_data[ 'status' ] = 'success'
        images = []
        data = response.json()
        info = json.loads(data[ 'info' ])

        for idx, result in enumerate(data[ 'images' ]):
            images.append({
                'base64': result,
                'seed': info['all_seeds'][idx],
                'mime_type': 'image/png',
                'nsfw': False
            })
        output_data[ 'images' ] = images
        return output_data
        
class InvokeAIEngineStableDiffusion(ProxyEngine):
    def prepare_args(self, task):
        args_dict = {
            'prompt' : flask.request.form[ 'prompt' ],
            'steps' : retrieve_param( 'num_inference_steps', flask.request.form, int,   100 ),
            'cfg_scale' : retrieve_param( 'guidance_scale', flask.request.form, float, 7.5 ),
            'eta' : retrieve_param( 'eta', flask.request.form, float, 0.0 ),
            'seed': retrieve_param( 'seed', flask.request.form, int, -1 ),
            'iterations': retrieve_param( 'num_outputs', flask.request.form, int, 1 ),
            'sampler_name': 'k_lms',
            'width': retrieve_param( 'width', flask.request.form, int,   512 ),
            'height': retrieve_param( 'height', flask.request.form, int,   512 ),
            'threshold': 0,
            'perlin': 0,
            'variation_amount': 0,
            'with_variations': '',
            'initimg': None,
            'strength': 0.99,
            'fit': 'on',
            'facetool_strength': 0.0,
            'upscale_level': '',
            'upscale_strength': 0,
            'initimg_name': ''
        }

        if (task == 'img2img' or task == 'masking'):
            init_img_b64 = flask.request.form[ 'init_image' ]
            init_img_b64 = 'data:image/png;base64,' + init_img_b64 if init_img_b64[0:4] != 'data' else mask_img_b64
            args_dict[ 'initimg' ] = init_img_b64
            args_dict[ 'imitimg_name' ] = 'temp.png'
            args_dict[ 'strength' ] = 1.0 - retrieve_param( 'strength', flask.request.form, float, 0.7 )
            endpoint_url = '/sdapi/v1/img2img'
        if (task == 'masking'):
            mask_img_b64 = flask.request.form[ 'mask_image' ]
            mask_img_b64 = 'data:image/png;base64,' + mask_img_b64 if mask_img_b64[0:4] != 'data' else mask_img_b64
            args_dict[ 'mask' ] = mask_img_b64
        return args_dict
    
    def run(self, task):
        total_results = []
        output_data = {}
        args_dict = self.prepare_args(task)
        response = self.process(self.base_url, args_dict)
        output_data[ 'status' ] = 'success'
        images = []
        json_data = '[{}]'.format(','.join(response.text.split('\n'))[:-1])
        data = json.loads(json_data)
        data = [item for item in data if item['event'] == 'result']
        for result in data:
            url = self.base_url + '/' + result['url']
            images.append({
                'base64': pil_to_b64(Image.open(requests.get(url, stream=True).raw)),
                'seed': result[ 'seed' ],
                'mime_type': 'image/png',
                'nsfw': False
            })
        output_data[ 'images' ] = images
        return output_data

class EngineManager(object):
    def __init__(self):
        self.engines = {}

    def has_engine(self, name):
        return ( name in self.engines )

    def add_engine(self, name, engine):
        if self.has_engine( name ):
            return False
        self.engines[ name ] = engine
        return True

    def get_engine(self, name):
        if self.has_engine( 'universal' ):
          return self.engines[ 'universal' ]
        if not self.has_engine( name ):
            return None
        engine = self.engines[ name ]
        return engine
