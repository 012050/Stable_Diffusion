import time

import keras_cv
import matplotlib.pyplot as plt
from tensorflow import keras

from fn import plot_images
from var import model

images = model.text_to_image("cookie", batch_size=1)

plot_images(images)

print("\n 실행 완료\n")

# https://youtu.be/JTw4WNC1Dy4