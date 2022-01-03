# from tensorflow.keras import backend as k
import tensorflow.keras as keras
from scipy import signal
from scipy.io import wavfile
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import initializers
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.keras.models import Sequential
# import keras
import tensorflow as tf
import pandas as pd
import random
# from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D,BatchNormalization,LeakyReLU
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense,GaussianNoise,GlobalAveragePooling2D,Dropout
import argparse
from model import *
import time
import pandas as pd
start_time = None
from sklearn.utils import class_weight
from sklearn import preprocessing
from tensorflow.keras.models import load_model
# from gensim.models import Word2Vec
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--learning', type=str, default="SS",required=True, help="[SSL, SS]")
parser.add_argument("--examples", type=int, default=10, required=True, help=[10,15])
parser.add_argument("--augmentation", type=int, default=0, required=True, help=["0 for no-aug amd 1 for aig"])
parser.add_argument("--dataset", type=str, default="GC", required=True, help=["ESC10, GC, fmnist, mnist"])
parser.add_argument("--model", type=str, default="small", required=True, help=["small, large"])
dic = {}
# CUDA_VISIBLE_DEVICES=0 python /root/volume/DataSets/DataSets/test.py --learning SSL --examples 25 --augmentation 1 --dataset mnist --model large

# w2v_model  = Word2Vec.load("word2vec.model")
# for run in range(2):  # This loop is for three runs
args = parser.parse_args()
args.learning="SS"
epochs=100
x_train, y_train, x_test, y_test, x_val, y_val=None,None, None,None, None,None
classes=None
input_shape = None
folder= None
batch_size=None
if args.dataset=="eurosat":
    batch_size = 64
    folder = "/root/volume/DataSets/DataSets/eurosat"
    if not os.path.exists(folder):
        os.mkdir(folder)
    x=np.load("/root/volume/DataSets/DataSets/EuroSAT/x_rgb.npy",allow_pickle=True)
    y=np.load("/root/volume/DataSets/DataSets/EuroSAT/y_rgb.npy",allow_pickle=True)
    train_, x_test, ytrain, y_test = train_test_split(
        x, y, test_size=0.20, random_state=34)
    x_train, x_test = x_train[..., np.newaxis] / 255.0, x_test[..., np.newaxis] / 255.0
    x_train = x_train.reshape((x_train.shape[0], 64, 64, 3))
    x_test = x_test.reshape((x_test.shape[0], 64, 64, 3))
    train_, x_val, ytrain, y_val = train_test_split(
        x_train, y_train, test_size=0.20, random_state=34)
    classes = 10
    input_shape = (64, 64,1)
elif args.dataset=="mnist":
    batch_size = 64
    folder = "/root/volume/DataSets/DataSets/MNIST"
    if not os.path.exists(folder):
        os.mkdir(folder)
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train[..., np.newaxis] / 255.0, x_test[..., np.newaxis] / 255.0
    x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
    x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))
    train_, x_val, ytrain, y_val = train_test_split(
        x_train, y_train, test_size=0.20, random_state=34)
    classes = 10
    input_shape = (28, 28, 1)
elif args.dataset=="fmnist":
    batch_size = 64
    folder = "/root/volume/DataSets/DataSets/fashion_MNIST"
    if not os.path.exists(folder):
        os.mkdir(folder)
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_train, x_test = x_train[..., np.newaxis] / 255.0, x_test[..., np.newaxis] / 255.0
    x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
    x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))
    train_, x_val, ytrain, y_val = train_test_split(
        x_train, y_train, test_size=0.20, random_state=34)
    classes = 10
    input_shape = (28, 28, 1)
elif args.dataset=="ESC10":
    batch_size=64
    folder = "/root/volume/DataSets/DataSets/ESC10"
    x = np.load("/root/volume/DataSets/PreparedDatasets/ESC10/X.npy", allow_pickle=True)
    y = np.load("/root/volume/DataSets/PreparedDatasets/ESC10/y.npy", allow_pickle=True)
    input_shape = (128, 216, 1)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.20, random_state=34)
    x_train = x_train.reshape((x_train.shape[0],128,216,1))
    x_test = x_test.reshape((x_test.shape[0],128,216,1))
    y_train -=1
    y_test-=1
    train_, x_val, ytrain, y_val = train_test_split(
        x_train, y_train, test_size=0.20, random_state=34)
    classes=10
    input_shape = (128,216,1)
elif args.dataset=="ER": # Emotion recognition data
    batch_size = 64
    folder = "/root/volume/DataSets/DataSets/ER"
    x = np.load("/root/volume/DataSets/PreparedDatasets/EmotionRecognition/x.npy", allow_pickle=True)
    y = np.load("/root/volume/DataSets/PreparedDatasets/EmotionRecognition/y.npy", allow_pickle=True)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.20, random_state=34)
    x_train = x_train.reshape((x_train.shape[0], 32, 32, 1))
    x_test = x_test.reshape((x_test.shape[0], 32, 32, 1))
    train_, x_val, ytrain, y_val = train_test_split(
        x_train, y_train, test_size=0.20, random_state=34)
    classes = 7
    input_shape = (32, 32, 1)
elif args.dataset=="AMNIST":
    batch_size=64
    folder = "/root/volume/DataSets/DataSets/AMNIST"
    x_train = np.load("/root/volume/DataSets/PreparedDatasets/MNIST/x_train.npy", allow_pickle=True)
    y_train = np.load("/root/volume/DataSets/PreparedDatasets/MNIST/y_train.npy", allow_pickle=True)
    x_test = np.load("/root/volume/DataSets/PreparedDatasets/MNIST/x_test.npy", allow_pickle=True)
    y_test = np.load("/root/volume/DataSets/PreparedDatasets/MNIST/y_test.npy", allow_pickle=True)
    train_, x_val, ytrain, y_val = train_test_split(
        x_train, y_train, test_size=0.20, random_state=34)
    classes=10
    input_shape = (32,32,1)
elif args.dataset=="GC":
    batch_size=64
    folder = "/root/volume/DataSets/DataSets/GCTrainedModel"
    x_train = np.load("/root/volume/DataSets/PreparedDatasets/GoogleCommands/x_train.npy", allow_pickle=True)
    y_train = np.load("/root/volume/DataSets/PreparedDatasets/GoogleCommands/y_train.npy", allow_pickle=True)
    x_test = np.load("/root/volume/DataSets/PreparedDatasets/GoogleCommands/x_test.npy", allow_pickle=True)
    y_test = np.load("/root/volume/DataSets/PreparedDatasets/GoogleCommands/y_test.npy", allow_pickle=True)
    x_val = np.load("/root/volume/DataSets/PreparedDatasets/GoogleCommands/x_val.npy", allow_pickle=True)
    y_val = np.load("/root/volume/DataSets/PreparedDatasets/GoogleCommands/y_val.npy", allow_pickle=True)
    x_train = x_train.reshape((x_train.shape[0], 32, 32, 1))
    x_test = x_test.reshape((x_test.shape[0], 32, 32, 1))
    x_val = x_val.reshape((x_val.shape[0], 32, 32, 1))
    classes=32
    input_shape=(32,32,1)
import numpy as np
examples = args.examples
selected_x , selected_y = None, None

# Due to memory issue of random samples
selected_x = np.zeros((examples*classes, x_train.shape[1],x_train.shape[2],x_train.shape[3]), dtype=x_train.dtype)
selected_y = np.zeros((examples*classes,), dtype=y_train.dtype)
i=0
y_train = y_train.reshape((y_train.shape[0]))
for eachClass in np.unique(y_train):
    selected_x[i*examples:(i+1)*examples,:,:,:] = x_train[eachClass==y_train,:,:,:][0:examples,:,:,:]
    selected_y[i * examples:(i + 1) * examples,] = y_train[eachClass == y_train][0:examples,]
    # x_train[eachClass==y_train,:,:,:] = x_train[eachClass==y_train,:,:,:][examples :,:,:,:]
    i+=1
aug = "NO_Aug"

if args.augmentation==1:
    aug="Aug"
threshold=0.4
accuracies = []
Exe_time = []
datagen = None

if args.augmentation == 1:
    if args.dataset in ["mnist", "cifar10", "eurosat"]:
        datagen = ImageDataGenerator(rotation_range=5,
                                     width_shift_range=0.1,
                                     height_shift_range=0.1,
                                     shear_range=0.05,
                                     zoom_range=0.05,
                                     horizontal_flip=False,
                                     vertical_flip=False,
                                     fill_mode='nearest'
                                     )
    else:
        datagen = ImageDataGenerator(
            # width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            # height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=False,  # randomly flip images
            vertical_flip=False,  # randomly flip images
            width_shift_range=0.05,
            height_shift_range=0.05
        )
args.learning="PSSL"
if args.learning=="PSSL":
    print("PSSSSSSSSSSSSSSSssssssssssssssssssssssssSSSSSSSSSSSSSL")
    start_time = time.time()
    args.learning="PSSL"
    x= np.copy(selected_x)
    y= np.copy(selected_y)
    count = 0
    if not os.path.exists(folder + "/PSSL"):
        os.mkdir(folder + "/PSSL")
    # fl = folder + "/PSSL/" + aug + "_" + args.learning + "_" + str(args.examples) + "_" + str(count) + ".h5"
    fl = folder + "/PSSL/" + args.model + "_" + aug + "_" + args.learning + "_" + str(args.examples) + "_" + str(count) + ".h5"
    print(fl)
    model = load_model(fl)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    prediction = np.argmax(model.predict(x_test),axis=1)
    if not os.path.exists(folder + "/PSSL_npy"):
        os.mkdir(folder + "/PSSL_npy")

    fl = folder + "/PSSL_npy/" + args.model.replace("\r","") + "_" + aug + "_" + args.learning + "_" + str(
        args.examples) + "_y_test_predicted"
    np.save(fl,prediction)
    print(prediction.shape)
    print(np.unique(prediction))

    args.learning="SSL"
    # selected_x=np.copy(x)
    # selected_y=np.copy(y)
    # accuracies.append(model.evaluate(x_test, tf.keras.utils.to_categorical(y_test, num_classes=classes), verbose=1)[1])
    # Exe_time.append(time.time() - start_time)
if args.learning=="SSL":

    count=0
    fl = folder+"/SSL/" +args.model+ "_" + aug + "_" + args.learning + "_" + str(args.examples) + "_" + str(count) + ".h5"
    print(fl.replace("\r",""))
    model = load_model(fl)
    prediction = np.argmax(model.predict(x_test),axis=1)
    if not os.path.exists(folder + "/SSL_npy/"):
        os.mkdir(folder + "/SSL_npy/")
    fl = folder + "/SSL_npy/" + args.model.replace("\r", "") + "_" + aug + "_" + args.learning + "_" + str(
        args.examples) + "_y_test_predicted"
    np.save(fl, prediction)
    # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # print("SSL Testing accuracy ", model.evaluate(x_test, tf.keras.utils.to_categorical(y_test,num_classes=classes,), verbose=1))
    # accuracies.append(model.evaluate(x_test, tf.keras.utils.to_categorical(y_test,num_classes=classes,), verbose=1)[1])
#     # Exe_time.append(time.time() - start_time)
# dic["Run"+str(run)+" Accuracy"] = accuracies
# dic["Run"+str(run)+" Time"] = Exe_time
#
# print(accuracies)
# print(Exe_time)
# print(dic)
# df= pd.DataFrame(dic)
# df.to_csv("/root/volume/DataSets/DataSets/AccuracyFiles/"+args.dataset+"_"+args.model+"_"+str(args.examples)+"_"+str(args.augmentation)+"_.csv")