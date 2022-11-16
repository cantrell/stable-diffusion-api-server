import json
import flask
import sys
import diffusers

from engine import EngineManager, EngineStableDiffusion, A1111EngineStableDiffusion, InvokeAIEngineStableDiffusion

# Load and parse the config file:
try:
    config_file = open ('config.json', 'r')
except:
    sys.exit('config.json not found.')

config = json.loads(config_file.read())

hf_token = config['hf_token']

if (hf_token == None and config.get('mode') != 'proxy'):
    sys.exit('No Hugging Face token found in config.json.')

custom_models = config['custom_models'] if 'custom_models' in config else []

# Initialize app:
app = flask.Flask( __name__ )

# Initialize engine manager:
manager = EngineManager()

# Add supported engines to manager:
if (config.get('mode') != 'proxy'):
    manager.add_engine( 'txt2img', EngineStableDiffusion( diffusers.StableDiffusionPipeline,        sibling=None ) )
    manager.add_engine( 'img2img', EngineStableDiffusion( diffusers.StableDiffusionImg2ImgPipeline, sibling=manager.get_engine( 'txt2img' ) ) )
    manager.add_engine( 'masking', EngineStableDiffusion( diffusers.StableDiffusionInpaintPipeline, sibling=manager.get_engine( 'txt2img' ) ) )
    for custom_model in custom_models:
        manager.add_engine( custom_model['url_path'],
                        EngineStableDiffusion( diffusers.StableDiffusionPipeline, sibling=manager.get_engine( 'txt2img' ),
                        custom_model_path=custom_model['model_path'],
                        requires_safety_checker=custom_model['requires_safety_checker'] ) )
else:
    engine = None
    if config['base_provider'] == 'AUTOMATIC1111':
        engine = A1111EngineStableDiffusion(config['base_url'])
    elif config['base_provider'] == 'InvokeAI':
        engine = InvokeAIEngineStableDiffusion(config['base_url'])
    manager.add_engine('universal', engine)

# Define routes:
@app.route('/ping', methods=['GET'])
def stable_ping():
    return flask.jsonify( {'status':'success'} )

@app.route('/custom_models', methods=['GET'])
def stable_custom_models():
    if custom_models == None:
        return flask.jsonify( [] )
    else:
        return custom_models

@app.route('/txt2img', methods=['POST'])
def stable_txt2img():
    return _generate('txt2img')

@app.route('/img2img', methods=['POST'])
def stable_img2img():
    return _generate('img2img')

@app.route('/masking', methods=['POST'])
def stable_masking():
    return _generate('masking')

@app.route('/custom/<path:model>', methods=['POST'])
def stable_custom(model):
    return _generate('txt2img', model)

def _generate(task, engine=None):
    # Retrieve engine:
    if engine == None:
        engine = task

    engine = manager.get_engine( engine )

    # Prepare output container:
    output_data = {}

    # Handle request:
    try:
        output_data = engine.run(task)
    except RuntimeError as e:
        output_data[ 'status' ] = 'failure'
        output_data[ 'message' ] = 'A RuntimeError occurred. You probably ran out of GPU memory. Check the server logs for more details.'
        print(str(e))
    return flask.jsonify( output_data )

if __name__ == '__main__':
    app.run( host='0.0.0.0', port=1337, debug=False )