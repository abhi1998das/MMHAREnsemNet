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

distdiffinput = Input(shape=(65, 190, 1))


distdiffmodel = Conv2D(64, (3,3), kernel_regularizer=l2(0.001),kernel_initializer='glorot_normal' , activation = 'relu')(distdiffinput)
distdiffmodel = Conv2D(64, (3,3), kernel_regularizer=l2(0.001),kernel_initializer='glorot_normal', activation = "relu")(distdiffmodel)

distdiffmodel = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(distdiffmodel)
distdiffmodel = BatchNormalization()(distdiffmodel)


distdiffmodel = Conv2D(128, (3,3), kernel_regularizer=l2(0.001),kernel_initializer='glorot_normal', activation = "relu")(distdiffmodel)
distdiffmodel = Conv2D(128, (3,3), kernel_regularizer=l2(0.001),kernel_initializer='glorot_normal', activation = "relu")(distdiffmodel)

distdiffmodel = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(distdiffmodel)
distdiffmodel = BatchNormalization()(distdiffmodel)



distdiffmodel = Conv2D(256, (3,3), kernel_regularizer=l2(0.001),kernel_initializer='glorot_normal', activation = "relu")(distdiffmodel)
distdiffmodel = Conv2D(256, (3,3), kernel_regularizer=l2(0.001),kernel_initializer='glorot_normal', activation = "relu")(distdiffmodel)
distdiffmodel = Conv2D(256, (3,3), kernel_regularizer=l2(0.001),kernel_initializer='glorot_normal', activation = "relu")(distdiffmodel)

distdiffmodel = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(distdiffmodel)
distdiffmodel = BatchNormalization()(distdiffmodel)



distdiffmodel = Conv2D(512, (3,3), kernel_regularizer=l2(0.001),kernel_initializer='glorot_normal', activation = "relu")(distdiffmodel)
distdiffmodel = Conv2D(512, (2,2), kernel_regularizer=l2(0.001),kernel_initializer='glorot_normal', activation = "relu")(distdiffmodel)

#max pool added
#distdiffmodel = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(distdiffmodel)
distdiffmodel = BatchNormalization()(distdiffmodel)

distdiffmodel = Flatten()(distdiffmodel)
distdiffmodel = Dense(256, activation='relu', kernel_regularizer=l2(0.001),kernel_initializer='glorot_normal')(distdiffmodel)
distdiffmodel = BatchNormalization()(distdiffmodel)
distdiffmodel = Dropout(0.5)(distdiffmodel)
distdiffmodel = Dense(256, activation='relu', kernel_regularizer=l2(0.001),kernel_initializer='glorot_normal')(distdiffmodel)
distdiffmodel = Dense(num_of_classes, activation = 'softmax')(distdiffmodel)

distdiffmodel = Model(inputs = distdiffinput, outputs = distdiffmodel)
distdiffmodel.summary()

sgd = optimizers.SGD(lr = 0.05, momentum = 0.9, clipnorm = 1.0)
adam = optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999,clipnorm = 1.0)
distdiffmodel.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

checkpoint1 = ModelCheckpoint('modeldistdiff.h5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.7, min_delta = 0.0005,
                              patience=20, min_lr=0.0001, verbose = 1)
callbacks_list = [checkpoint1,reduce_lr]

from keras.preprocessing.image import ImageDataGenerator
train_datagen_dist_diff = ImageDataGenerator(
                                   #samplewise_center=True,
                                   #samplewise_std_normalization=True
                                   )
val_datagen_dist_diff = ImageDataGenerator(
                                   #samplewise_center=True,
                                   #samplewise_std_normalization=True
                                 )


training_set_dist_diff= train_datagen_dist_diff.flow_from_directory(
    traindirdistdiff,
    target_size=(65, 190),
    color_mode='grayscale',
    batch_size=5,
    class_mode='categorical',
    seed = 42
   )

val_set_dist_diff= val_datagen_dist_diff.flow_from_directory(
    valdirdistdiff,
    target_size=(65, 190),
    color_mode='grayscale',
    batch_size=5,
    class_mode='categorical',
    shuffle = False, 
    seed = 42
   )

H = distdiffmodel.fit_generator(
    training_set_dist_diff,
    steps_per_epoch=87, #No of train images / batch size
    #steps_per_epoch=1,
    epochs=1000,
    validation_data = val_set_dist_diff,
    validation_steps = 87, #No of validation images / batch size
    callbacks=callbacks_list)

!cp /content/modeldistdiff_8620.h5 '/content/drive/My Drive/UT-MHAD_models/metalearner-models/fulltrainingdata'
distdiffmodel = load_model('/content/drive/My Drive/UT-MHAD_models/metalearner-models/fulltrainingdata/modeldistdiff_8620.h5')

from sklearn.metrics import classification_report, confusion_matrix

#Confution Matrix and Classification Report
Y_pred = distdiffmodel.predict_generator(val_set_dist_diff)
y_pred = np.argmax(Y_pred, axis=1)

matrix = confusion_matrix(val_set_dist_diff.classes, y_pred,labels=None)

from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt

fig, ax = plot_confusion_matrix(conf_mat=matrix,
                                #colorbar=True,
                                cmap = 'viridis',
                                show_absolute=True,
                                show_normed=False,
                                figsize = (10,10)
                                )
plt.savefig('modeldistdiffcm.png')
!cp /content/modeldistdiffcm.png '/content/drive/My Drive/UT-MHAD_models/metalearner-models/fulltrainingdata'