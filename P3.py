import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from numpy.random import random
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Flatten, Lambda, Cropping2D, Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint

# placeholders to store the image and angle data
car_images = []
steering_angles = []


DATA_DIRS = ['/opt/carnd_p3/track1_recovery/','/opt/carnd_p3/data/']

for DATA_DIR in DATA_DIRS:
    if DATA_DIR == '/opt/carnd_p3/data/':
        driving_log = pd.read_csv(DATA_DIR + 'driving_log.csv')  # read the driving log from the csv file
    else:
        driving_log = pd.read_csv(DATA_DIR + 'driving_log.csv',names=["center","left","right","steering","throttle","break","speed"])

    # get the steering angle and apply correction factor to the left and right cameras
    for _,line in driving_log.iterrows():
        steering_center = float(line["steering"])

        # drop 10% of straight driving images to balance curve driving samples
        if (steering_center > 1) or ((steering_center <=1) and (random()>0.1)):
            # create adjusted steering measurements for the side camera images
            correction = 0.08 # this is a parameter to tune
            steering_left = steering_center + correction
            steering_right = steering_center - correction

            path = DATA_DIR+"IMG/"
            # data fetched from Linux
            if DATA_DIR == '/opt/carnd_p3/data/':
                split_token = '/'
            else: # data fetched from Windows
                split_token = '\\'
            # read images from their path in the driving log
            img_center = np.asarray(Image.open(path + line["center"].split(split_token)[-1]))
            img_left = np.asarray(Image.open(path + line["left"].split(split_token)[-1]))
            img_right = np.asarray(Image.open(path + line["right"].split(split_token)[-1]))

            # add images and angles to data set
            car_images.extend([img_center, img_left, img_right])
            steering_angles.extend([steering_center, steering_left, steering_right])

# ## Augment Images
# apply horiontal flip to augment the datasets
augmented_images, augmented_measurements = [],[]
for image,measurement in zip(car_images,steering_angles):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image,1))
    augmented_measurements.append(measurement*-1.0)

X = np.array(augmented_images,ndmin=4)
y = np.array(augmented_measurements)

print(X.shape)

# Model Architecture. From https://developer.nvidia.com/blog/deep-learning-self-driving-cars/
def drivingModel():
    model = Sequential()
    model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((70,25),(0,0))))
    model.add(Conv2D(24,(5,5),strides=(2,2),activation='relu'))
    model.add(Conv2D(36,(5,5),strides=(2,2),activation='relu'))
    model.add(Conv2D(48,(5,5),strides=(2,2),activation='relu'))
    model.add(Conv2D(64,(3,3),strides=(1,1),activation='relu'))
    model.add(Conv2D(64,(3,3),strides=(1,1),activation='relu'))
    model.add(Flatten())
    model.add(Dense(1164,activation='relu'))
    model.add(Dense(100,activation='relu'))
    model.add(Dense(50,activation='relu'))
    model.add(Dense(10,activation='relu'))
    model.add(Dense(1))
    return model


model = drivingModel()
print(model.summary())


# Callbacks
model_checkpoint_callback = ModelCheckpoint(
    filepath='/home/workspace/CarND-Behavioral-Cloning-P3/model.h5',
    save_best_only=True,
    monitor='val_loss',
    mode='min')

callbacks = [model_checkpoint_callback]

num_epochs=5
model.compile(loss='mse',optimizer='adam')
history_object = model.fit(X,y,epochs=num_epochs,validation_split=0.2,shuffle=True,callbacks=callbacks)

### plot the training and validation loss for each epoch
plt.figure()
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig("train_val_loss.png")
plt.close()
