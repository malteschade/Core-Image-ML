#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
{Core Image ML Model Building - Python Main
CNN Model building and training for bounding box regression.}

{INTERNAL USE ONLY}
"""

__author__ = '{Malte Schade}'
__copyright__ = 'Copyright {2022}, {Core Image ML Model Building - Python Main}'
__version__ = '{1}.{0}.{0}'
__maintainer__ = '{Malte Schade}'
__email__ = '{contact@malteschade.com}'
__status__ = '{FINISHED}'

# built-in modules
import os
import importlib

# other modules
from joblib import Parallel, delayed
import pandas as pd
import numpy as np
from PIL import Image, ImageFile
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Input
keras_applications = importlib.import_module('tensorflow.keras.applications')
keras_optimizers = importlib.import_module('tensorflow.keras.optimizers')

# constants
MODEL_NAME = 'model'
MODEL_FOLDER = 'models'
IMG_FOLDER = 'images'
META_FILE = 'metadata.csv'
STRICT_RATIO = True
MIN_RESO_X = 1024
MIN_RESO_Y = 1024
MAX_RESO_X = None
MAX_RESO_Y = None

config = dict(
    predictor_mode=9,
    epochs=100,
    batch_size=32,
    learning_rate=1e-5,
    optimizer='Adam',
    target_size=224,
    loss='mse',
    test_size=0.30,
    trainable=False,
    model='VGG19',
    weights='imagenet'
)

# settings
Image.MAX_IMAGE_PIXELS = 10000000000
ImageFile.LOAD_TRUNCATED_IMAGES = True


def clean_data(data: pd.Series) -> pd.Series:
    """
    Filters the metadataset dependant on the specified configuration.
    Scales the response variables to the range [0, 1].

    Parameters
    ----------
    data: pd.Series
        Series with the data path and nine response variables.

    Returns
    ----------
    None/data: pd.Series
        Returns either None when filtered or the scaled input Series.
    """
    path = data[0]

    # return if image path is nan
    if type(path) != str:
        return

    # return if image path does not exist
    if not os.path.exists(path):
        return

    # get (w,h) size of image
    size = Image.open(path).size

    # return if any response variable is nan
    if not any(data[2:]):
        return

    # return if image aspect ratio is not 4/3
    if STRICT_RATIO == True:
        if size[0]/size[1] != 4/3:
            return

    # return if image resolution is smaller/larger than specified thresholds
    if MIN_RESO_X:
        if size[0] < MIN_RESO_X:
            return
    if MIN_RESO_Y:
        if size[0] < MIN_RESO_Y:
            return
    if MAX_RESO_X:
        if size[0] < MAX_RESO_X:
            return
    if MAX_RESO_Y:
        if size[0] < MAX_RESO_Y:
            return

    # scale point response variables to the range [0,1]
    data[2:][::2] = [x/size[0] for x in data[2:][::2]]
    data[2:][1::2] = [y/size[1] for y in data[2:][1::2]]
    return data


def build_model(config: dict) -> Model:
    """
    Construct a CNN for the bounding box regression with custom specified 
    parameters like optimization algorithm, pre-trained weights, model blueprint,
    image target size, predictor mode, loss function, and learning rate.

    Parameters
    ----------
    config: dict
        Dictionary with all settings for the CNN model building.

    Returns
    ----------
    model: Model
    Keras multi-layer neural network model with configured input/output layer size,
    optimizer, and loss function.
    """

    # load defined optimizer and model blueprint
    model_class = getattr(keras_applications, config['model'])
    opt_class = getattr(keras_optimizers, config['optimizer'])

    # create raw model with definable pre-trained weights and with a specified input shape
    model = model_class(weights=config['weights'], include_top=False,
                        input_tensor=Input(shape=[config['target_size'], config['target_size']]+[3]))

    # set trainability of model blueprint weights
    model.trainable = config['trainable']

    # append pyramidal dense layers to reach desired output tensor shape
    flatten = model.output
    flatten = Flatten()(flatten)
    bboxHead = Dense(256, activation="relu")(flatten)
    bboxHead = Dense(128, activation="relu")(bboxHead)
    bboxHead = Dense(64, activation="relu")(bboxHead)
    bboxHead = Dense(32, activation="relu")(bboxHead)

    # get final number of output values in the range [0,1] (sigmoid)
    bboxHead = Dense(config['predictor_mode'], activation="sigmoid")(bboxHead)

    # create model
    model = Model(inputs=model.input, outputs=bboxHead)

    # define optimization function for the model
    opt = opt_class(learning_rate=config['learning_rate'])
    model.compile(loss=config['loss'], optimizer=opt)
    return model


def transform_image(data: pd.Series, config: dict) -> tuple([np.array, pd.Series]):
    """
    Load image with file path, convert to configured target size and convert to numpy array (ts,ts,3).

    Parameters
    ----------
    data: pd.Series
        Series with the data path and nine response variables.

    config: dict
        Dictionary with all settings for the CNN model building.

    Returns
    ----------
    img: np.array
        Numpy array with three channel image information.

    data: pd.Series
        Series with point response variables.
    """
    img = img_to_array(load_img(data[0], target_size=[
                       config['target_size'], config['target_size']]))
    return img, data[2:]


def train(data: pd.DataFrame) -> Model:
    """
    Prepare training images with parallel multi threading. Build CNN model.
    Train CNN model with model data and model targets.

    Parameters
    ----------
    data: pd.DataFrame
        Metadata information about image file paths and response variables.

    Returns
    ----------
    model: Model
        Trained CNN model.
    """

    # build configured CNN model
    model = build_model(config)

    # prepare training data with parallel multi threading
    columns = list(zip(*Parallel(n_jobs=-1, backend='threading')(delayed(transform_image)
                                                                 (data.iloc[i], config) for i in range(data.__len__()))))

    # prediciton without core number
    if config['predictor_mode'] == 8:
        # normalize color values to the range [0, 1]
        model_data = np.array(columns[0], dtype='float32') / 255.0
        model_targets = np.array(columns[1], dtype='float32')

    # prediction with core number
    elif config['predictor_mode'] == 9:
        model_targets = pd.DataFrame(columns[1])

        # normalize core number to the range [0, 1]
        model_targets['n_cores'] = pd.DataFrame(data).iloc[:, 1]/10

        # normalize color values to the range [0, 1]
        model_data = np.array(columns[0], dtype='float32') / 255.0
        model_targets = np.array(model_targets, dtype='float32')

    # fit keras model to the training data
    h = model.fit(x=model_data, y=model_targets, validation_split=config['test_size'],
                  verbose=2, epochs=config['epochs'])
    return model


# read metadata information into DataFrame
metadata = pd.read_csv(os.path.join(os.getcwd(), META_FILE), index_col=0)

# filter and transform the metadata with parallel multi threading
metadata = Parallel(n_jobs=-1)(delayed(clean_data)
                               (metadata.iloc[i]) for i in range(metadata.__len__()))
metadata = pd.concat(metadata, axis=1).T.reset_index(drop=True)

# construct and train CNN model
trained_model = train(metadata)

# save trained model as .h5 file
trained_model.save(os.path.join(os.getcwd(), MODEL_FOLDER, MODEL_NAME+'.h5'))
