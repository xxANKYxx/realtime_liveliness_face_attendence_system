import os
from PIL import Image
from PIL import Image
# PIL--> python image library
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import model_from_json
from imageio import imread,imsave
from tensorflow.keras import  Sequential

import numpy as np

IMG_SIZE=24

def load_model():
    json_file=open('model.json')
    loaded_file=json_file.read()
    json_file.close()
    loaded_model=model_from_json(loaded_file)
    loaded_model.load_weights("model.h5")
    loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return loaded_model

def predict(img, model):
	img = Image.fromarray(img, 'RGB').convert('L')
	img = img.resize((IMG_SIZE, IMG_SIZE))
	img_arr = np.asarray(img)
	img_arr = img_arr.reshape(1, IMG_SIZE, IMG_SIZE, 1)
	prediction = model.predict(img_arr)
	if prediction < 0.1:
		prediction = 'closed'
	elif prediction > 0.9:
		prediction = 'open'
	else:
		prediction = 'idk'
	return prediction