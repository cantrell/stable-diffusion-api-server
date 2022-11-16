from PIL import Image
from io import BytesIO
import flask
import base64
import torch

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

def get_compute_platform(context):
    try:
        import torch
        if torch.cuda.is_available():
            return 'cuda'
        elif torch.backends.mps.is_available() and context == 'engine':
            return 'mps'
        else:
            return 'cpu'
    except ImportError:
        return 'cpu'
