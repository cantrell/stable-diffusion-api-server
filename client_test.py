import json
import requests
import base64
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt

def load_image_from_path(img_path):
    img = Image.open( img_path )
    return img

def load_image_from_url(img_url):
    res = requests.get( img_url )
    img = Image.open( BytesIO( res.content ) )
    return img

def resize_image_preserve_aspect(img_pil, w):
    wp = ( w / float( img_pil.size[0] ) )
    hs = int( float( img_pil.size[1] ) * float( wp ) )
    return img_pil.resize( ( w, hs ), Image.ANTIALIAS )

def pil_to_b64(input):
    buffer = BytesIO()
    input.save( buffer, 'PNG' )
    output = base64.b64encode( buffer.getvalue() ).decode( 'utf-8' ).replace( '\n', '' )
    buffer.close()
    return output

def b64_to_pil(input):
    output = Image.open( BytesIO( base64.b64decode( input ) ) )
    return output

def test_txt2img():
    ENDPOINT = "http://localhost:1337/txt2img"
    
    data = {
        'prompt':'a photo of a dog sitting on a bench',
        'width':str( 512 ),
        'height':str( 512 ),
        'num_inference_steps':str( 100 ),
        'guidance_scale':str( 7.5 ),
        'num_outputs':str( 2 ),
        'seed':str( 0 ),
    }

    response = json.loads( requests.post( url=ENDPOINT, data=data ).text )

    def b64_to_pil(input):
        output = Image.open( BytesIO( base64.b64decode( input ) ) )
        return output

    if 'status' in response:
        if response[ 'status' ] == 'success':
            images = response[ 'images' ]
            for i, image in enumerate( images ):
                plt.imshow( b64_to_pil( image['base64'] ) )
                plt.show( block=True )
                plt.pause( 10 )
                plt.close()

def test_img2img():
    ENDPOINT = "http://localhost:1337/img2img"    
    IMG_URL  = 'https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg'

    #IMG_URL  = 'https://minitravellers.co.uk/wp-content/uploads/2019/05/40612080213_81852c19fc_k.jpg'

    data = {
        'prompt':'a family of pixar characters on vacation in new york',
        'init_image':pil_to_b64( resize_image_preserve_aspect( load_image_from_url( IMG_URL ).convert( 'RGB' ), 512 ) ),
        'num_inference_steps':str( 100 ),
        'guidance_scale':str( 7.5 ),
        'num_outputs':str( 2 ),
        'seed':str( 0 ),
        'strength':str( 0.5 ),
        'eta':str( 0.0 ),
    }

    response = json.loads( requests.post( url=ENDPOINT, data=data ).text )

    if 'status' in response:
        if response[ 'status' ] == 'success':
            images = response[ 'images' ]
            for i, image in enumerate( images ):
                plt.imshow( b64_to_pil( image['base64'] ) )
                plt.show( block=True )
                plt.pause( 10 )
                plt.close()

def test_inpaint():
    ENDPOINT = "http://localhost:1337/masking"    
    IMG_URL  = 'https://raw.githubusercontent.com/CompVis/stable-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png'
    MSK_URL  = 'https://raw.githubusercontent.com/CompVis/stable-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png'

    data = {
        'prompt':'a cat sitting on a bench',
        'init_image':pil_to_b64( resize_image_preserve_aspect( load_image_from_url( IMG_URL ).convert( 'RGB' ), 512 ) ),
        'mask_image':pil_to_b64( resize_image_preserve_aspect( load_image_from_url( MSK_URL ).convert( 'RGB' ), 512 ) ),
        'num_inference_steps':str( 100 ),
        'guidance_scale':str( 7.5 ),
        'num_outputs':str( 2 ),
        'seed':str( 0 ),
        'strength':str( 0.8 ),
        'eta':str( 0.0 ),
    }

    response = json.loads( requests.post( url=ENDPOINT, data=data ).text )

    if 'status' in response:
        if response[ 'status' ] == 'success':
            images = response[ 'images' ]
            for i, image in enumerate( images ):
                plt.imshow( b64_to_pil( image['base64'] ) )
                plt.show( block=True )
                plt.pause( 10 )
                plt.close()


# Run tests:

# test_txt2img()
# test_img2img()
test_inpaint()