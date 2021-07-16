
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator

# initiallize classifier
clf = Sequential()

# step 1: convolution
# input_shape 3,256,256 takes too long, here 3 is 3 color channels.
clf.add(Convolution2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))

# step 2: pooling
clf.add(MaxPooling2D(pool_size=(2, 2),data_format='channels_first'))

# add a second conv layer
# we need to change input shape
# we don't need input_shape parameter now.
clf.add(Convolution2D(32, (3, 3), activation='relu')) # change
clf.add(MaxPooling2D(pool_size=(2, 2), data_format='channels_first'))
##   ----- before this layer we had got 75% accuracy
## this will give 80% accuracy, but will take too long time.



# step 3: flattening
clf.add(Flatten())

# step 4: full connection
clf.add(Dense(units=128, activation='relu'))
clf.add(Dense(units=1, activation='sigmoid'))

# compile the cnn
clf.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# part2: fitting the cnn to the images
# https://keras.io/preprocessing/image/
# we create two instances with the same arguments
train_datagen = ImageDataGenerator(
    rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

training_set = train_datagen.flow_from_directory(
    '../dataset/training_set',  # change
    target_size=(64, 64),  # change
    batch_size=32,
    class_mode='binary')  # binary for cats and dogs

test_set = test_datagen.flow_from_directory(
    '../dataset/test_set',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary')

clf.fit_generator(
    training_set,
    steps_per_epoch=8000,
    epochs=25,
    validation_data=test_set,  # change
    validation_steps=2000)  # change

# Part 3: Making new predictions
import numpy as np
from keras.preprocessing import image

test_image = image.load_img('../dataset/single_prediction/cat_or_dog_1.jpg', target_size=(64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = clf.predict(test_image)

print(training_set.class_indices)

if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'
    

print(prediction)





