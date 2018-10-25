from keras.preprocessing.image import img_to_array, load_img, ImageDataGenerator
from skimage import io, color
import matplotlib.pyplot as plt
import numpy as np
image_size = 64
image_path = "data/"
def read_image(img_id, dir):
    try:
        img = load_img(dir + "/" + img_id, target_size=(image_size, image_size))
        img = img_to_array(img)
        return img
    except:
        return None

def show_image(image):
    plt.imshow(image/255.)
    plt.show()

def printOutput(file, output):
    for y in range(256):
        row = ""
        for x in range(256):
            row += str(output[y, x]) + " "
        file.write(row + "\n")
    file.close()

def preprocess(examples):
    examples = examples*(1.0/255)
    examples = color.rgb2lab(examples)
    #examples = meanCenter(Xtrain, examples)
    examples = examples[:, :, :, 0]
    examples = examples - 50
    examples = examples/100
    examples = examples.reshape(examples.shape+(1,))
    return examples

def meanCenter(trainData, dataToCenter):
  mean = np.mean(trainData, 0, dtype=np.float32)
  mean[:, :, 1:] = 0 #center only the lightness channel, which is the input
  return dataToCenter - mean