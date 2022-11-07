import time
import keras_cv
from tensorflow  import keras
import matplotlib.pyplot as plt
from var import model
from fn import plot_images

images = model.text_to_image("cookie", batch_size=1)

plot_images(images)

print("\n 실행 완료\n")

# https://youtu.be/JTw4WNC1Dy4