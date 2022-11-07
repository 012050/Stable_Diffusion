import time
import keras_cv
from tensorflow  import keras
import matplotlib.pyplot as plt

# model = keras_cv.models.StableDiffusion(img_width=512, img_height=512)
model = keras_cv.models.StableDiffusion(img_width=320, img_height=320)
