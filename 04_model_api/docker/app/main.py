#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
{Core Image ML Model RestAPI- Python 1
RestAPI for model prediction with images as binary post request body and 
response as JSON.}

{INTERNAL USE ONLY}
"""

__author__ = '{Malte Schade}'
__copyright__ = 'Copyright {2022}, {Core Image ML Model RestAPI- Python 1}'
__version__ = '{1}.{0}.{0}'
__maintainer__ = '{Malte Schade}'
__email__ = '{contact@malteschade.com}'
__status__ = '{FINISHED}'

# built-in modules
import os
import io
import traceback

# other modules
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from fastapi import FastAPI, File
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

# constants
MODEL_NAME_8 = '8_model_best'
MODEL_NAME_9 = '9_model_best'
MODEL_FOLDER = 'app/models'

# settings
Image.MAX_IMAGE_PIXELS = 10000000000


# load trained Keras CNN models
model_8 = load_model(os.path.join(MODEL_FOLDER, MODEL_NAME_8+'.h5'))
model_9 = load_model(os.path.join(MODEL_FOLDER, MODEL_NAME_9+'.h5'))

# create new FastAPI app
app = FastAPI()


@app.post("/predict")
async def predict_file(file: bytes = File(...), mode: int = None) -> JSONResponse:
    """
    Predict bounding box position and core number with binary image file. Return
    response as JSON or Exception information.

    Parameters
    ----------
    file: bytes
        Image as binary file

    mode: int
        Prediction mode:
            8 -> Predicition of bounding box
            9 -> Prediction of bounding box and core number

    Returns
    ----------
    response: JSONResponse
        Predicted, rescaled, and rounded response variables in JSON format.
    """
    try:
        # load image from Post body
        img = Image.open(io.BytesIO(file))

        # get image size (w,h)
        size = img.size

        # get input shape
        res = model_8.input_shape[2] if mode == 8 else model_9.input_shape[2]

        # convert image to RGB color profile and resize to model input layer
        img = img.convert('RGB')
        img = img.resize((res, res), Image.NEAREST)

        # convert image to numpy array (res, res, 3)
        img = [img_to_array(img)]

        # normalize color values to the range [0, 1]
        model_data = np.array(img, dtype='float32') / 255.0

        # prediciton without core number
        if mode == 8:
            # predict with model
            pred = model_8.predict(model_data)

            # rescale predicted values to original size
            pred[:, ::2] = np.round(np.multiply(pred[:, ::2], size[0]))
            pred[:, 1::2] = np.round(np.multiply(pred[:, 1::2], size[1]))

            # define response keys
            keys = ['p1x', 'p1y', 'p2x', 'p2y', 'p3x', 'p3y', 'p4x', 'p4y']

        # prediciton with core number
        elif mode == 9:
            # predict with model
            pred = np.roll(model_9.predict(model_data), 1, 1)

            # rescale predicted values to original size
            pred[:, 0] = np.round(pred[:, 0]*10)
            pred[:, 1::2] = np.round(np.multiply(pred[:, 1::2], size[0]))
            pred[:, 2::2] = np.round(np.multiply(pred[:, 2::2], size[1]))

            # define response keys
            keys = ['row_nb', 'p1x', 'p1y', 'p2x',
                    'p2y', 'p3x', 'p3y', 'p4x', 'p4y']

        # create result dictionary from keys and predicted values
        result = dict(zip(keys, *pred.tolist()))

        # encode dictionary as JSON response
        json_result = jsonable_encoder(result)
        response = JSONResponse(content=json_result)
        return  response

    except Exception as e:
        # catch exception and receive exception type and content
        e_type = 'Exception type:'+str(e.__class__.__name__)
        e_content = traceback.format_exc()

        # return exception information as response
        return e_type+'\n'+e_content
