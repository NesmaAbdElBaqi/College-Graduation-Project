import numpy as np
from PIL import Image


def preprocess_image(im, IM_WIDTH, IM_HEIGHT):
    im = np.resize(im, (IM_WIDTH, IM_HEIGHT, im.shape[-1]))
    im = np.array(im) / 255.0
    return im

