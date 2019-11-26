#%% md
# CIFAR10 Transfer Learning based Classifier

#This notebook outlines the steps to build a classifier to leverage concepts of Transfer Learning by utilizing a pretrained Deep-CNN.
#Particularly in this case based on VGG16
#%%
# Pandas and Numpy for data structures and util fucntions
import sys,os
#sys.path.append("pydevd-pycharm.egg")
#import pydevd_pycharm

#pydevd_pycharm.settrace('103.46.128.41', port=15512, stdoutToServer=True,
#                        stderrToServer=True)
sys.path.append(os.pardir)

import scipy as sp
import numpy as np
import pandas as pd
from numpy.random import rand
pd.options.display.max_colwidth = 600

# Scikit Imports
from sklearn import preprocessing
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.model_selection import train_test_split

import cnn_utils as utils
from model_evaluation_utils import get_metrics

# Matplot Imports
import matplotlib.pyplot as plt
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (15, 5),
          'axes.labelsize': 'x-large',
          'axes.titlesize':'x-large',
          'xtick.labelsize':'x-large',
          'ytick.labelsize':'x-large'}

plt.rcParams.update(params)
#%matplotlib inline

# pandas display data frames as tables
from IPython.display import display, HTML

import warnings
warnings.filterwarnings('ignore')
#%%
import tensorflow as tf
from keras import callbacks
from keras import optimizers
from keras.datasets import cifar10
from keras.engine import Model
from keras.applications import vgg16 as vgg
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D,BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
#%% md
## Load and Prepare DataSet
#%%
BATCH_SIZE = 32
EPOCHS = 40
NUM_CLASSES = 10
LEARNING_RATE = 1e-4
MOMENTUM = 0.9
#%%
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
#%% md

# Split training dataset in train and validation sets
#%%
X_train, X_val, y_train, y_val = train_test_split(X_train,
                                                  y_train,
                                                  test_size=0.15,
                                                  stratify=np.array(y_train),
                                                  random_state=42)
#%% md
# Transform target variable/labels into one hot encoded form
#%%
Y_train = np_utils.to_categorical(y_train, NUM_CLASSES)
Y_val = np_utils.to_categorical(y_val, NUM_CLASSES)
Y_test = np_utils.to_categorical(y_test, NUM_CLASSES)
#%% md
### Preprocessing

#Since we are about to use VGG16 as a feature extractor, the minimum size of an image it takes is 48x48.
#We utilize ```scipy`` to resize images to required dimensions
#%%
X_train = np.array([sp.misc.imresize(x,
                                     (48, 48)) for x in X_train])
X_val = np.array([sp.misc.imresize(x,
                                   (48, 48)) for x in X_val])
X_test = np.array([sp.misc.imresize(x,
                                    (48, 48)) for x in X_test])
#%% md
## Prepare the Model

#* Load VGG16 without the top classification layer
#* Prepare a custom classifier
#* Stack both models on top of each other
#%%
base_model = vgg.VGG16(weights='imagenet',
                       include_top=False,
                       input_shape=(48, 48, 3))
#%%
# Extract the last layer from third block of vgg16 model
last = base_model.get_layer('block3_pool').output
#%%
# Add classification layers on top of it
x = GlobalAveragePooling2D()(last)
x= BatchNormalization()(x)
x = Dense(256, activation='relu')(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.6)(x)
pred = Dense(NUM_CLASSES, activation='softmax')(x)
model = Model(base_model.input, pred)
#%% md
#Since our objective is to only train the custom classifier, we freeze the layers of VGG16
#%%
for layer in base_model.layers:
     layer.trainable = False
#%%
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.Adam(lr=LEARNING_RATE),
              metrics=['accuracy'])
#%%
model.summary()
#%% md
## Data Augmentation

#To help model generalize and overcome the limitations of a small dataset, we prepare augmented datasets using
#```keras ``` utilities
#%%
# prepare data augmentation configuration
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_datagen = ImageDataGenerator( rescale=1./255,  horizontal_flip=False )

train_datagen.fit(X_train)
train_generator = train_datagen.flow(X_train,
                                     Y_train,
                                     batch_size=BATCH_SIZE)
#%%
val_datagen = ImageDataGenerator(rescale=1. / 255,
    horizontal_flip=False)

val_datagen.fit(X_val)
val_generator = val_datagen.flow(X_val,
                                 Y_val,
                                 batch_size=BATCH_SIZE)
#%% md
## Train the Model
#%%
train_steps_per_epoch = X_train.shape[0] // BATCH_SIZE
val_steps_per_epoch = X_val.shape[0] // BATCH_SIZE

history = model.fit_generator(train_generator,
                              steps_per_epoch=train_steps_per_epoch,
                              validation_data=val_generator,
                              validation_steps=val_steps_per_epoch,
                              epochs=EPOCHS,
                              verbose=1)
#%% md
## Analyze Model Performance
#%%
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
t = f.suptitle('Deep Neural Net Performance', fontsize=12)
f.subplots_adjust(top=0.85, wspace=0.3)

epochs = list(range(1,EPOCHS+1))
ax1.plot(epochs, history.history['acc'], label='Train Accuracy')
ax1.plot(epochs, history.history['val_acc'], label='Validation Accuracy')
ax1.set_xticks(epochs)
ax1.set_ylabel('Accuracy Value')
ax1.set_xlabel('Epoch')
ax1.set_title('Accuracy')
l1 = ax1.legend(loc="best")

ax2.plot(epochs, history.history['loss'], label='Train Loss')
ax2.plot(epochs, history.history['val_loss'], label='Validation Loss')
ax2.set_xticks(epochs)
ax2.set_ylabel('Loss Value')
ax2.set_xlabel('Epoch')
ax2.set_title('Loss')
l2 = ax2.legend(loc="best")
#%%
predictions = model.predict(X_test/255.)
#%%
test_labels = list(y_test.squeeze())
predictions = list(predictions.argmax(axis=1))
#%%
get_metrics(true_labels=y_test,
                predicted_labels=predictions)
#%% md
## Visualize Predictions
#%%
label_dict = {0:'airplane',
             1:'automobile',
             2:'bird',
             3:'cat',
             4:'deer',
             5:'dog',
             6:'frog',
             7:'horse',
             8:'ship',
             9:'truck'}
#%%
utils.plot_predictions(model=model,dataset=X_test/255.,
                       dataset_labels=Y_test,
                       label_dict=label_dict,
                       batch_size=16,
                       grid_height=4,
                       grid_width=4)