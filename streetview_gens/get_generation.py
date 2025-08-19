import streetlevel.streetview
from streetlevel.dataclasses import Size
from .gens import CameraGeneration
from .model import predict
import cv2
import numpy as np
def get_generation(panoid):
    pano = streetlevel.streetview.find_panorama_by_id(panoid)
    if not pano:
        raise ValueError("No panorama found")
    max_image_size = pano.image_sizes[-1]
    if max_image_size == Size(x=3328, y=1664):
        return CameraGeneration.GEN1,1
    elif max_image_size == Size(x=16384, y=8192):
        return CameraGeneration.GEN4,1
    elif max_image_size == Size(x=13312, y=6656):
        image = streetlevel.streetview.get_panorama(pano)
        im_array_rgb = np.array(image)
        im_array_bgr = cv2.cvtColor(im_array_rgb, cv2.COLOR_RGB2BGR)
        confidences = predict(im_array_bgr)[0]
        return CameraGeneration(confidences.argmax()), confidences.max()
    else:
        raise ValueError(f"unknown resolution: {max_image_size}")


