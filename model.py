
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense,GaussianNoise,GlobalAveragePooling2D,Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LeakyReLU
import tensorflow.keras
import tensorflow as tf
import numpy as np
import random
def getSmallModel(input_shape, classes, w2v_model=None):

    model = Sequential()
    model.add(Conv2D(64, kernel_size=(2, 2), activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, kernel_size=(2, 2), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, kernel_size=(2, 2), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())  # Flattening the 2D arrays for fully connected layers

    model.add(Dense(128, activation=tf.nn.relu))
    model.add(Dropout(0.2))

    model.add(Dense(classes,activation=tf.nn.softmax))
    model.compile(loss ="sparse_categorical_crossentropy",#loss=keras.losses.categorical_crossentropy,
              optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.01,clipvalue=1),
              metrics=['accuracy'])
    return model
def getACLModel(input_shape, classes, w2v_model=None):

    # model = Sequential()
    # model.add(Conv2D(16, kernel_size=(3, 3), input_shape=input_shape))
    # model.add(LeakyReLU())
    # model.add(Dropout(0.33))
    #
    #
    # model.add(Conv2D(16, kernel_size=(3, 3),   input_shape=input_shape))
    # model.add(LeakyReLU())
    # model.add(Dropout(0.33))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    #
    #
    # model.add(Conv2D(16, kernel_size=(3, 3),  input_shape=input_shape))
    # model.add(LeakyReLU())
    # model.add(Dropout(0.33))
    #
    # model.add(Conv2D(16, kernel_size=(3, 3),  input_shape=input_shape))
    # model.add(LeakyReLU())
    # model.add(Dropout(0.33))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Flatten())  # Flattening the 2D arrays for fully connected layers
    #
    # model.add(Dense(classes,activation=tf.nn.softmax))

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(5, 5), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, kernel_size=(5, 5), activation='relu'))
    model.add(AveragePooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, kernel_size=(5, 5), activation='relu'))
    model.add(AveragePooling2D(pool_size=(2, 2)))
    model.add(Flatten())  # Flattening the 2D arrays for fully connected layers

    model.add(Dense(64, activation=tf.nn.relu))
    model.add(Dropout(0.2))

    model.add(Dense(classes, activation=tf.nn.softmax))

    model.compile(loss ="sparse_categorical_crossentropy",#loss=keras.losses.categorical_crossentropy,
              optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.01,clipvalue=1),
              metrics=['accuracy'])
    return model


def getLargeModel(input_shape=(32,32,3), classes=10, w2v_model=None):
    """

    :param input_shape:input share in tuples
    :return:
    """
    model = Sequential()
    model.add(GaussianNoise(0.15,input_shape = input_shape))
    model.add(Conv2D(128,(3,3),  padding="same"))
    model.add(LeakyReLU(0.1))
    model.add(Conv2D(128, (3, 3), padding="same" ))
    model.add(LeakyReLU(0.1))
    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(LeakyReLU(0.1))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Dropout(0.5))
    model.add(Conv2D(256,(3,3),  padding="same"))
    model.add(LeakyReLU(0.1))
    model.add(Conv2D(256, (3, 3), padding="same"))
    model.add(LeakyReLU(0.1))
    model.add(Conv2D(256, (3, 3), padding="same"))
    model.add(LeakyReLU(0.1))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Dropout(0.5))
    model.add(Conv2D(512,(3,3),  padding="valid"))
    model.add(LeakyReLU(0.1))
    model.add(Conv2D(256, (1, 1)))
    model.add(LeakyReLU(0.1))
    model.add(Conv2D(128, (1, 1) ))
    model.add(LeakyReLU(0.1))
    model.add(GlobalAveragePooling2D())
    model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
    model.add(Dense(128, activation=tf.nn.relu))
    model.add(Dropout(0.1))
    model.add(Dense(classes,activation=tf.nn.softmax))

    model.compile(loss="sparse_categorical_crossentropy",  # loss=keras.losses.categorical_crossentropy,
                  optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.01),
                  metrics=['accuracy'])
    return model
def RemoveSelectedElements(x,indices):
    return np.delete(x,indices,axis=0)
def SelectSamples(x,y,samples_per_class=10):
    x_=[]  # Data will be used for training
    y_=[] # label of data
    x_u = None
    status=True
    y_u = []
    xy = np.copy(y)
    xy = xy.reshape((xy.shape[0]))
    for i in np.unique(xy):
        indx=[]
        rng = np.random.default_rng()

        for j in random.sample(range(x[xy==i].shape[0]),samples_per_class):  # randomly select the samples
            indx.append(j)
            x_.append(x[xy==i][j])
            y_.append(i)
        for k in range(x[i==xy].shape[0]-len(indx)):
            y_u.append(i)
        if status:
            status=False
            x_u = RemoveSelectedElements(np.copy(x[xy==i]), indx)

        else:
            x_u = np.concatenate((x_u,RemoveSelectedElements(np.copy(x[xy==i]), indx)),axis=0)

    return np.array(x_), np.array(y_), x_u, np.array(y_u)

# This is the pytorch version
"""import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

class LeakyRelu(nn.Module):
    def __init__(self,negative_slope=0.1):
        super(LeakyRelu, self).__init__()
        self.negative_slope=negative_slope
    def forward(self,x):
        x[x<0] = x[x<0]*self.negative_slope
        return x
class LargeNet(nn.Module):
    def __init__(self, channels=3, classes=10):
        super(LargeNet, self).__init__()
        self.conv1 = nn.Conv2d(channels, 128, 3)
        self.LRelu1 = LeakyRelu (0.1)
        self.conv2 = nn.Conv2d(128, 128, 3)
        self.LRelu2 = LeakyRelu(0.1)
        self.conv3 = nn.Conv2d(128, 128, 3)
        self.LRelu3 = LeakyRelu(0.1)
        self.pool1 = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout(0.5)

        self.conv4 = nn.Conv2d(128, 256, 3)
        self.LRelu4 = LeakyRelu(0.1)
        self.conv5 = nn.Conv2d(256, 256, 3)
        self.LRelu5 = LeakyRelu(0.1)
        self.conv6 = nn.Conv2d(256,256, 3)
        self.LRelu6 = LeakyRelu(0.1)
        self.pool2 = nn.MaxPool2d(2)
        self.dropout2 = nn.Dropout(0.5)

        self.conv7 = nn.Conv2d(256, 512, 3)
        self.LRelu7 = LeakyRelu(0.1)
        self.conv8= nn.Conv2d(512, 256, 1)
        self.LRelu8 = LeakyRelu(0.1)
        self.conv9 = nn.Conv2d(256,128, 1)
        self.LRelu9 = LeakyRelu(0.1)

        self.averagePool  = nn.AvgPool2d(1)
        self.fc1 = nn.Linear(128, classes)

    def forward(self, x):

        x = self.conv1(x)
        x = self.LRelu1(x)
        x = self.conv2(x)
        x = self.LRelu2(x)
        x = self.conv3(x)
        x = self.LRelu3(x)
        x = self.pool1(x)
        x= self.dropout1(x)

        x = self.conv4(x)
        x = self.LRelu4(x)
        x = self.conv5(x)
        x = self.LRelu5(x)
        x = self.conv6(x)
        x = self.LRelu6(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        x = self.conv7(x)
        x= self.LRelu7(x)
        x = self.conv8(x)
        x= self.LRelu8(x)
        x = self.conv9(x)
        x= self.LRelu9(x)
        print("Size : ", x.size())
        x = self.averagePool(x)


        x=self.fc1(x)
        print("Size : ", x.size())
        x = torch.flatten(x, 2)
        output = F.log_softmax(x, dim=1)
        return output """