import tensorflow as tf
import numpy as np
import cv2

model = tf.keras.models.load_model('FinalModel.h5')
image = cv2.imread('testing_images/images/SYEBQDSN[20230725-223708].png')
input = cv2.resize(image, (299, 299))
input = np.array(input, dtype=np.float32)
input /= 255.0
input = input.reshape((1,299,299, 3))
# testing_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1./255,validation_split=0)
# testing_data = testing_datagen.flow_from_directory('testing_images',
#     target_size=(299,299),
#     class_mode=None,
#     batch_size=1,
#     subset="training")
# print(testing_data)
output = model.predict(input)

# for i in range(len(output)):
#     print(np.argmax(output[i]))
print(output[0][np.argmax(output)])

category = []
file = open('data/name of the animals.txt', 'r')
while True:
    line = file.readline()
    if not line:
        break
    category.append(line)
print(category[np.argmax(output)])
