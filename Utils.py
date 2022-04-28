import numpy as np
import cv2


def convert_to_gray_scale(images):
    list_images = []
    for image in images:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        list_images.append(image)
    return list_images


def resize_image(images, img_w, img_h):
    images_list = []
    for image in images:
        image = cv2.resize(image, (img_w, img_h))
        images_list.append(image)
    return np.array(images_list)
