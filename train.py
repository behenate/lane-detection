#%%
import numpy as np
import cv2

import keras 
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Convolution2D, MaxPooling2D, Dropout, Flatten, Dense

from sklearn.utils import shuffle

import random
import copy 
import os
import pickle
import matplotlib.pyplot as plt

from image_alteration import perlin_shadows, lens_distort, zoom_pan_rotate, blur, color_transforms, gaussian_noise, draw_points

#%%
labels_path = 'labels/'
labels_pahts = os.listdir(labels_path)

test_label = pickle.load(open(labels_path + labels_pahts[300], "rb"))
print(test_label['img'].shape)
# cv2.namedWindow("pre-crop", cv2.WINDOW_KEEPRATIO)
# cv2.namedWindow("crop", cv2.WINDOW_KEEPRATIO)
#%%
def img_crop(img, rl, ll):  
    img = img[70:,:,:]
    # print(rl[0,1])
    rl = np.asarray(rl)
    ll = np.asarray(ll)
    rl[:, 1] -= 70
    ll[:, 1] -= 70
    return img, rl, ll
#%%
def img_preprocess(img, right_lane, left_lane):
    img, rl, ll = img_crop(img, right_lane, left_lane)
    img = cv2.resize(img, (480, 200))
    # rl[:,0] = rl[:,0] / img.shape[1]

    rl_div_x = rl[:,0] / img.shape[1]
    rl_div_y = rl[:,1] / img.shape[0]
    rl = [rl_div_x, rl_div_y]
    rl = np.asarray(rl)
    rl = rl.transpose()

    # img = cv2.cvtColor(img, cv2.COLOR)
    img = cv2.GaussianBlur(img,  (3, 3), 0)

    img = img/255
    return img, rl, ll
#%%
def random_augment(img, right_lane, left_lane):
    if np.random.rand() < 0.5:
        img = perlin_shadows(img)
    if np.random.rand() < 0.3:
        img, right_lane, left_lane = lens_distort(img, right_lane, left_lane, 35)
    if np.random.rand() < 0.5:
        img, right_lane, left_lane = zoom_pan_rotate(img, right_lane, left_lane, 0.7)
    if np.random.rand() < 0.2:
        img = blur(img, 1)
    if np.random.rand() < 0.5:
        img = color_transforms(img)
    if np.random.rand() < 0.5:
        img = gaussian_noise(img)
    return img, right_lane, left_lane
#%%
def batch_generator(labels_path, batch_size, istraining):
    labels_pahts = os.listdir(labels_path)
    while True:
        batch_img = []
        batch_points = []
        for i in range(batch_size):
            random_index = random.randint(0, len(labels_pahts) - 1)
            label = pickle.load(open(labels_path + labels_pahts[random_index], "rb"))
            img, rl, ll = label['img'], label['right_lane'], label['left_lane']

            if istraining:
                img, rl, ll = random_augment(img, rl, ll)


            img, rl, ll = img_preprocess(img, rl, ll)
            # Flatten and concat points
            rl_ravel = np.ravel(rl)
            ll_ravel = np.ravel(ll)
            points = np.concatenate((rl_ravel, ll_ravel))
            
            batch_img.append(img)
            batch_points.append(points)
        yield (np.asarray(batch_img), np.asarray(batch_points))  
#%%
for i in range(1):
    imgs, points = next(batch_generator(labels_path, 10, True))
    for point in points:
        print(point.shape)
#%%
def nvidia_model():
    model = Sequential()
    # Subsample is the same as stride length
    model.add(Convolution2D(24, 5, 5, subsample=(2,2), input_shape=(200,480,3), activation='elu'))
    model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation='elu'))
    model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation='elu'))
    model.add(Convolution2D(64, 3, 3, activation='elu'))
    model.add(Convolution2D(64, 3, 3, activation='elu'))
#     model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(1000, activation='elu'))
#     model.add(Dropout(0.5))
    model.add(Dense(500, activation='elu'))
#     model.add(Dropout(0.5))
    model.add(Dense(24))
    
    model.compile(optimizer = Adam(lr=1e-4), loss = 'mse')
    return model
#%%
model = nvidia_model()
h = model.fit_generator(batch_generator(labels_path, 100, 1), 
                                        steps_per_epoch=300,
                                        epochs=10, 
                                        validation_data=batch_generator(labels_path, 100, 0),
                                        validation_steps=200, 
                                        verbose=True, 
                                        shuffle=True)
#%%
for label in labels_pahts:
    data = pickle.load(open(labels_path + label, "rb"))
    cv2.imshow("pre-crop", data['img'])
    img, right_lane, left_lane = random_augment(data['img'], data['right_lane'], data['left_lane'])
    img, right_lane, left_lane = img_preprocess(img, right_lane, left_lane)
    cv2.imshow("crop", img)
    key = cv2.waitKey(0)
    del data, img, right_lane, left_lane
    if key == ord('q'):
        break