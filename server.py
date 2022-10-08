import mimetypes
import re
import time
import inspect
import flask
import base64
from PIL import Image
from io import BytesIO

from pipeline import *

##################################################
# Utils

def retrieve_param(key, data, cast, default):
    if key in data:
        value = flask.request.form[ key ]
        value = cast( value )
        return value
    return default

def pil_to_b64(input):
    buffer = BytesIO()
    input.save( buffer, 'PNG' )
    output = base64.b64encode( buffer.getvalue() ).decode( 'utf-8' ).replace( '\n', '' )
    buffer.close()
    return output

def b64_to_pil(input):
    output = Image.open( BytesIO( base64.b64decode( input ) ) )
    return output

def get_compute_platform():
    try:
        import torch
        if torch.cuda.is_available():
            return 'cuda'
        else:
            return 'cpu'
    except ImportError:
        return 'cpu'

##################################################
# Engines

class Engine(object):
    hf_token = ""
    def __init__(self):
        pass

    def process(self, kwargs):
        return []

class EngineStableDiffusion(Engine):
    def __init__(self, pipe, sibling=None):
        super().__init__()
        if sibling == None:
            token_file = open("token.txt", "r")
            token = token_file.read()
            self.engine = pipe.from_pretrained( "CompVis/stable-diffusion-v1-4", use_auth_token=token.strip() )
        else:
            self.engine = pipe(
                vae=sibling.engine.vae,
                text_encoder=sibling.engine.text_encoder,
                tokenizer=sibling.engine.tokenizer,
                unet=sibling.engine.unet,
                scheduler=sibling.engine.scheduler,
                feature_extractor=sibling.engine.feature_extractor
            )
        self.engine.to( get_compute_platform() )

    def process(self, kwargs):
        images_pil = self.engine( **kwargs )
        return images_pil

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
        if not self.has_engine( name ):
            return None
        engine = self.engines[ name ]
        return engine

##################################################
# App

# Initialize app:
app     = flask.Flask( __name__ )
# Initialize engine manager:
manager = EngineManager()
# Add supported engines to manager:
manager.add_engine( 'stable_txt2img', EngineStableDiffusion( StableDiffusionPipeline,        sibling=None ) )
manager.add_engine( 'stable_img2img', EngineStableDiffusion( StableDiffusionImg2ImgPipeline, sibling=manager.get_engine( 'stable_txt2img' ) ) )
manager.add_engine( 'stable_masking', EngineStableDiffusion( StableDiffusionMaskingPipeline, sibling=manager.get_engine( 'stable_txt2img' ) ) )

@app.route('/txt2img', methods=['POST'])
def stable_txt2img():
    # Retrieve engine:
    engine = manager.get_engine( 'stable_txt2img' )
    # Prepare output container:
    output_data = {}
    # Handle request:
    try:
        # Get input text from request:
        input_text = flask.request.form[ 'prompt' ]
        # Get config:
        config_width    = retrieve_param( 'width',               flask.request.form, int,   512 )
        config_height   = retrieve_param( 'height',              flask.request.form, int,   512 )
        config_steps    = retrieve_param( 'num_inference_steps', flask.request.form, int,   100 )
        config_guidance = retrieve_param( 'guidance_scale',      flask.request.form, float, 7.5 )
        config_count    = retrieve_param( 'num_outputs',         flask.request.form, int,   1 )
        config_seed     = retrieve_param( 'seed',                flask.request.form, int,   0 )
        # Prepare seeder:
        generator = torch.Generator( device=get_compute_platform() ).manual_seed( config_seed )
        # Formulate args:
        args_dict = {
            'prompt' : [ input_text ]*config_count,
            'width' : config_width, 
            'height' : config_height, 
            'num_inference_steps' : config_steps,
            'guidance_scale' : config_guidance,
            'generator' : generator,
        }
        # Perform inference:
        outs_pil = engine.process( args_dict )
        output_data[ 'status' ] = 'success'
        images = []
        for image in outs_pil:
            images.append({
                'base64' : pil_to_b64( image.convert( 'RGB' ) ),
                'mimetype': 'image/png'
            })
        output_data[ 'images' ] = images
    except RuntimeError as e:
        # Append output:
        output_data[ 'status' ] = 'failure'
        output_data[ 'message' ] = 'A RuntimeError occurred. You probably ran out of GPU memory. Check the server logs for more details.'
        print(str(e))
    return flask.jsonify( output_data )

@app.route('/img2img', methods=['POST'])
def stable_img2img():
    # Retrieve engine:
    engine = manager.get_engine( 'stable_img2img' )
    # Prepare output container:
    output_data = {}
    # Handle request:
    try:
        # Get input text from request:
        input_text = flask.request.form[ 'prompt' ]
        # Get init image from request:
        init_img_b64 = flask.request.form[ 'init_image' ]
        init_img_b64 = re.sub( "^data:image/png;base64,", "", init_img_b64 )
        init_img_pil = b64_to_pil( init_img_b64 )
        # Get config:
        config_steps    = retrieve_param( 'num_inference_steps', flask.request.form, int,   100 )
        config_guidance = retrieve_param( 'guidance_scale',      flask.request.form, float, 7.5 )
        config_stength  = retrieve_param( 'strength',            flask.request.form, float, 0.8 )
        config_ddim_eta = retrieve_param( 'eta',                 flask.request.form, float, 0.0 )
        config_count    = retrieve_param( 'num_outputs',         flask.request.form, int,   1 )
        config_seed     = retrieve_param( 'seed',                flask.request.form, int,   0 )
        # Prepare seeder:
        generator = torch.Generator( device=get_compute_platform() ).manual_seed( config_seed )
        # Formulate args:
        args_dict = {
            'prompt' : [ input_text ]*config_count,
            'init_image' : init_img_pil,
            'num_inference_steps' : config_steps,
            'guidance_scale' : config_guidance,
            'strength' : config_stength,
            'eta' : config_ddim_eta,
            'generator' : generator,
        }
        # Perform inference:
        outs_pil = engine.process( args_dict )
        # Append output:
        output_data[ 'status' ] = 'success'
        images = []
        for image in outs_pil:
            images.append({
                'base64' : pil_to_b64( image.convert( 'RGB' ) ),
                'mimetype': 'image/png'
            })
        output_data[ 'images' ] = images
    except RuntimeError as e:
        output_data[ 'status' ] = 'failure'
        output_data[ 'message' ] = 'A RuntimeError occurred. You probably ran out of GPU memory. Check the server logs for more details.'
        print(str(e))
    return flask.jsonify( output_data )

@app.route('/masking', methods=['POST'])
def stable_masking():
    # Retrieve engine:
    engine = manager.get_engine( 'stable_masking' )
    # Prepare output container:
    output_data = {}
    # Handle request:
    try:
        # Get input text from request:
        input_text = flask.request.form[ 'prompt' ]
        # Get init image from request:
        init_img_b64 = flask.request.form[ 'init_image' ]
        init_img_b64 = re.sub( "^data:image/png;base64,", "", init_img_b64 )
        init_img_pil = b64_to_pil( init_img_b64 )
        # Get mask image from request:
        mask_img_b64 = flask.request.form[ 'mask_image' ]
        mask_img_b64 = re.sub( "^data:image/png;base64,", "", mask_img_b64 )
        mask_img_pil = b64_to_pil( mask_img_b64 )
        # Get config:
        config_steps    = retrieve_param( 'num_inference_steps', flask.request.form, int,   100 )
        config_guidance = retrieve_param( 'guidance_scale',      flask.request.form, float, 7.5 )
        config_stength  = retrieve_param( 'strength',            flask.request.form, float, 0.8 )
        config_ddim_eta = retrieve_param( 'eta',                 flask.request.form, float, 0.0 )
        config_count    = retrieve_param( 'num_outputs',         flask.request.form, int,   1 )
        config_seed     = retrieve_param( 'seed',                flask.request.form, int,   0 )
        # Prepare seeder:
        generator = torch.Generator( device=get_compute_platform() ).manual_seed( config_seed )
        # Formulate args:
        args_dict = {
            'prompt' : [ input_text ]*config_count,
            'init_image' : init_img_pil,
            'mask_image' : mask_img_pil,
            'num_inference_steps' : config_steps,
            'guidance_scale' : config_guidance,
            'strength' : config_stength,
            'eta' : config_ddim_eta,
            'generator' : generator,
        }
        # Perform inference:
        outs_pil = engine.process( args_dict )
        output_data[ 'status' ] = 'success'
        images = []
        for image in outs_pil:
            images.append({
                'base64' : pil_to_b64( image.convert( 'RGB' ) ),
                'mimetype': 'image/png'
            })
        output_data[ 'images' ] = images        
    except RuntimeError as e:
        output_data[ 'status' ] = 'failure'
        output_data[ 'message' ] = 'A RuntimeError occurred. You probably ran out of GPU memory. Check the server logs for more details.'
        print(str(e))
    return flask.jsonify( output_data )


if __name__ == '__main__':
    app.run( host='0.0.0.0', port=8080, debug=False )