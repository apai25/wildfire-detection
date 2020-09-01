from tensorflow.keras.models import load_model
import numpy as np 
from tensorflow.keras.preprocessing.image import load_img, img_to_array

image_path = 'data/test/not_wildfire/images(1).png'

def predict(image_path):
    cnn = load_model('model')
    image = load_img(f'{image_path}', target_size=(128, 128))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    prediction = int(cnn.predict(image)[0][0])
    return prediction
    
prediction = predict(image_path)

if prediction == 1:
    print('Wildfire')
elif prediction == 0:
    print('Not a wildfire')