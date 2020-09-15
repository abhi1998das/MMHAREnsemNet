import numpy as np
import os
from glob import glob
import math
import tensorflow as tf
from keras import optimizers
from keras.models import Sequential
from keras.models import load_model
from keras.models import Model
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import SpatialDropout2D
from keras.layers import Flatten, concatenate
from keras.layers import Input
from keras.layers import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Permute
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint
from keras.callbacks import LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import CSVLogger
from keras import layers, models
from keras import backend as K
import matplotlib.pyplot as plt

data_format = K.image_data_format()
K.set_image_data_format(data_format)
np.random.seed(42)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

num_of_classes = 27

angdiffinput = Input(shape=( 65, 3420, 1))


angdiffmodel = Conv2D(32, (3,3), kernel_regularizer=l2(0.001),kernel_initializer='glorot_normal' , activation = 'relu')(angdiffinput)
angdiffmodel = Conv2D(32, (3,3), kernel_regularizer=l2(0.001),kernel_initializer='glorot_normal', activation = "relu")(angdiffmodel)

angdiffmodel = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(angdiffmodel)
angdiffmodel = BatchNormalization()(angdiffmodel)


angdiffmodel = Conv2D(64, (3,3), kernel_regularizer=l2(0.001),kernel_initializer='glorot_normal', activation = "relu")(angdiffmodel)
angdiffmodel = Conv2D(64, (3,3), kernel_regularizer=l2(0.001),kernel_initializer='glorot_normal', activation = "relu")(angdiffmodel)

angdiffmodel = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(angdiffmodel)
angdiffmodel = BatchNormalization()(angdiffmodel)



angdiffmodel = Conv2D(128, (3,3), kernel_regularizer=l2(0.001),kernel_initializer='glorot_normal', activation = "relu")(angdiffmodel)
angdiffmodel = Conv2D(128, (3,3), kernel_regularizer=l2(0.001),kernel_initializer='glorot_normal', activation = "relu")(angdiffmodel)

angdiffmodel = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(angdiffmodel)
angdiffmodel = BatchNormalization()(angdiffmodel)



angdiffmodel = Conv2D(256, (3,3), kernel_regularizer=l2(0.001),kernel_initializer='glorot_normal', activation = "relu")(angdiffmodel)
angdiffmodel = Conv2D(256, (3,3), kernel_regularizer=l2(0.001),kernel_initializer='glorot_normal', activation = "relu")(angdiffmodel)

angdiffmodel = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(angdiffmodel)
angdiffmodel = BatchNormalization()(angdiffmodel)

angdiffmodel = Flatten()(angdiffmodel)
angdiffmodel = Dense(256, activation='relu', kernel_regularizer=l2(0.001),kernel_initializer='glorot_normal')(angdiffmodel)
angdiffmodel = BatchNormalization()(angdiffmodel)
angdiffmodel = Dropout(0.5)(angdiffmodel)
angdiffmodel = Dense(256, activation='relu', kernel_regularizer=l2(0.001),kernel_initializer='glorot_normal')(angdiffmodel)
angdiffmodel = Dense(num_of_classes, activation = 'softmax')(angdiffmodel)

angdiffmodel = Model(inputs = angdiffinput, outputs = angdiffmodel)
angdiffmodel.summary()

from keras.preprocessing.image import ImageDataGenerator

train_datagen_angle_diff = ImageDataGenerator(
                                   #samplewise_center=True,
                                   #samplewise_std_normalization=True
                                   )
val_datagen_angle_diff = ImageDataGenerator(
                                   #samplewise_center=True,
                                   #samplewise_std_normalization=True
                                 )


training_set_angle_diff= train_datagen_angle_diff.flow_from_directory(
    traindiranglediff,
    target_size=(65, 3420),
    color_mode='grayscale',
    batch_size=5,
    class_mode='categorical',
    seed = 42
   )

val_set_angle_diff= val_datagen_angle_diff.flow_from_directory(
    valdiranglediff,
    target_size=(65, 3420),
    color_mode='grayscale',
    batch_size=32,
    class_mode='categorical',
    shuffle = False, 
    seed = 42
   )

sgd = optimizers.SGD(lr = 0.01, momentum = 0.9,clipnorm = 1.0)
adam = optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999,clipnorm = 1.0)
angdiffmodel.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

checkpoint1 = ModelCheckpoint('modelangdiff.h5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.7, min_delta = 0.0005,
                              patience=20, min_lr=0.0001, verbose = 1)
callbacks_list = [checkpoint1,reduce_lr]

H = angdiffmodel.fit_generator(
    training_set_angle_diff,
    steps_per_epoch=87, #No of train images / batch size
    #steps_per_epoch=1,
    epochs=1000,
    validation_data = val_set_angle_diff,
    validation_steps = 87, #No of validation images / batch size
    callbacks=callbacks_list)

!cp /content/modelangdiff_9226.h5 '/content/drive/My Drive/UT-MHAD_models/metalearner-models/fulltrainingdata'
angdiffmodel = load_model('/content/modelangdiff_9226.h5')

from sklearn.metrics import classification_report, confusion_matrix

#Confution Matrix and Classification Report
Y_pred = angdiffmodel.predict_generator(val_set_angle_diff)
y_pred = np.argmax(Y_pred, axis=1)

matrix = confusion_matrix(val_set_angle_diff.classes, y_pred,labels=None)

from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt

fig, ax = plot_confusion_matrix(conf_mat=matrix,
                                #colorbar=True,
                                cmap = 'viridis',
                                show_absolute=True,
                                show_normed=False,
                                figsize = (10,10)
                                )
plt.savefig('modelangdiffcm.png')
!cp /content/modelangdiffcm.png '/content/drive/My Drive/UT-MHAD_models/metalearner-models/fulltrainingdata'