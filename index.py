import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
## Data preprocessing
## Training Image Preprocessing

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip=True)
training_set = train_datagen.flow_from_directory(
    'Dataset\\train',target_size=(224,224),batch_size=32,shuffle=True,class_mode='binary')
test_datagen = ImageDataGenerator(rescale=1./255)
test_set = test_datagen.flow_from_directory(
        'Dataset\\train',target_size=(224,224),batch_size=16,shuffle=False,class_mode='binary')

cnn = tf.keras.models.Sequential()
cnn.add(tf.keras.layers.Conv2D(filters=224 , kernel_size=3 , activation='relu' , input_shape=[224,224,3]))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))
cnn.add(tf.keras.layers.Conv2D(filters=224 , kernel_size=3 , activation='relu' , input_shape=[224,224,3]))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))
cnn.add(tf.keras.layers.Dropout(0.5))
cnn.add(tf.keras.layers.Flatten())
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
cnn.add(tf.keras.layers.Dense(units=1 , activation='sigmoid'))
cnn.compile(optimizer = 'Adam' , loss = 'binary_crossentropy' , metrics = ['accuracy'])

cnn.fit(x = training_set , validation_data = test_set , epochs = 1)

test_image = tf.keras.utils.load_img('Dataset/prediction/yes3.jpg',target_size=(224,224))
test_image = tf.keras.utils.img_to_array(test_image)
test_image = np.expand_dims(test_image,axis=0)
result = cnn.predict(test_image)
training_set.class_indices
print(result)
if result[0][0] == 1:
    print('yes')
else:
    print('no')