# from tensorflow.keras import backend as k
# import keras
from scipy import signal
from scipy.io import wavfile
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import initializers
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
# from keras.models import Sequential
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
parser.add_argument("--examples", type=int, default=10, required=True, help=[10,15]) # number of sample per class 
parser.add_argument("--augmentation", type=int, default=0, required=True, help=["0 for no-aug amd 1 for aig"])
parser.add_argument("--dataset", type=str, default="GC", required=True, help=["ESC10, GC, fmnist, mnist"])
# for this paper, we used large model, 13 layers. you can use small model if you want fast training  
parser.add_argument("--model", type=str, default="small", required=True, help=["small, large"])

dic = {}
# w2v_model  = Word2Vec.load("word2vec.model")
for run in range(5):  # This loop is for five runs
    args = parser.parse_args()
    args.learning="PSSL"
    epochs=100  # no of epochs  
    x_train, y_train, x_test, y_test, x_val, y_val=None,None, None,None, None,None # to store train, tesst and validation set 
    classes=None
    input_shape = None
    folder= None
    batch_size=None
	# dataset loading
    if args.dataset == "eurosat":
        batch_size = 64
        folder = "/root/volume/DataSets/DataSets/eurosat"
        if not os.path.exists(folder):
            os.mkdir(folder)
        x = np.load("/root/volume/DataSets/DataSets/EuroSAT/x_gray.npy", allow_pickle=True)
        print(x.shape)
        y = np.load("/root/volume/DataSets/DataSets/EuroSAT/y_gray.npy", allow_pickle=True)
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.20, random_state=34)
        # x_train, x_test = x_train[..., np.newaxis] / 255.0, x_test[..., np.newaxis] / 255.0
        x_train = x_train.reshape((x_train.shape[0], 64, 64, 1))
        x_test = x_test.reshape((x_test.shape[0], 64, 64, 1))
        train_, x_val, ytrain, y_val = train_test_split(
            x_train, y_train, test_size=0.20, random_state=34)
        classes = 10
        input_shape = (64, 64, 1)
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
        batch_size=8
        folder = "/root/volume/DataSets/DataSets/GCTrainedModel"
        x_train = np.load("/root/volume/DataSets/PreparedDatasets/GoogleCommands/x_train.npy", allow_pickle=True)
        y_train = np.load("/root/volume/DataSets/PreparedDatasets/GoogleCommands/y_train.npy", allow_pickle=True)
        x_test = np.load("/root/volume/DataSets/PreparedDatasets/GoogleCommands/x_test.npy", allow_pickle=True)
        y_test = np.load("/root/volume/DataSets/PreparedDatasets/GoogleCommands/y_test.npy", allow_pickle=True)
        x_val = np.load("/root/volume/DataSets/PreparedDatasets/GoogleCommands/x_val.npy", allow_pickle=True)
        y_val = np.load("/root/volume/DataSets/PreparedDatasets/GoogleCommands/y_val.npy", allow_pickle=True)
        x_train = x_train.reshape((x_train.shape[0], 64, 64, 1))
        x_test = x_test.reshape((x_test.shape[0], 64, 64, 1))
        x_val = x_val.reshape((x_val.shape[0], 64, 64, 1))
        classes=32
        input_shape=(64,64,1)
        # x_train = x_train + 160
        # x_test  = x_test+160
        # x_val = x_val +160
        # x_train = (x_train - np.min(x_train)) / (np.max(x_train) - np.min(x_train))-np.min(x_train)
        # x_test = (x_test - np.min(x_test)) / (np.max(x_test) - np.min(x_test))-np.min(x_test)
        # x_val = (x_val - np.min(x_val)) / (np.max(x_val) - np.min(x_val))-np.min(x_val)

        print(np.min(x_train), np.max(x_train))
        print(np.min(x_test), np.max(x_test))
        print(np.min(x_val), np.max(x_val))
        # exit(0)
    import numpy as np
    examples = args.examples
    selected_x , selected_y = None, None
    if args.examples>=50:
        # Due to memory issue of random samples, select the example in sequential way 
        selected_x = np.zeros((examples * classes, x_train.shape[1], x_train.shape[2], x_train.shape[3]),
                              dtype=x_train.dtype)
        selected_y = np.zeros((examples * classes,), dtype=y_train.dtype) - 1
        i = 0
        y_train = y_train.reshape((y_train.shape[0]))
        eachClass = 0

        for eachClass in np.unique(y_train):
            print("Class : ", eachClass, "  samples are : ", x_train[eachClass == y_train].shape,i * examples,"  ",(i + 1) * examples)
            selected_x[i * examples:(i + 1) * examples, :, :, :] = x_train[eachClass == y_train, :, :, :][:examples, :, :, :]
            selected_y[i * examples:(i + 1) * examples, ] = eachClass
            print(selected_x[selected_y == eachClass].shape)
            # x_train[eachClass==y_train,:,:,:] = x_train[eachClass==y_train,:,:,:][examples :,:,:,:]
            i += 1
        i = 0
    else:
        selected_x, selected_y, x_train, y_train = SelectSamples(x_train, y_train, examples)
    # if args.augmentation==1:
    #     temp = np.zeros((selected_x.shape), dtype= selected_x.dtype)
    #     selected_y = np.concatenate((selected_y, selected_y))
    #     # y_test = tf.keras.utils.to_categorical(y_test,num_classes=classes)
    #     for i in range(selected_x.shape[0]):
    #         temp[i] = np.copy(selected_x[i])+np.random.normal(size=(selected_x[i].shape[1],selected_x[i].shape[2]))
    #     selected_x = np.copy(np.concatenate((selected_x, temp),axis=0))
    #     temp = np.copy(x_train)
    #     y_train = np.concatenate((y_train, y_train))
    #     hel="d"
    #     for i in range(temp.shape[0]):
    #         temp[i] += np.random.normal()
    #     x_train = np.concatenate((x_train, temp), axis=0)

    # y_val = tf.keras.utils.to_categorical(y_val,num_classes=10) 
	# if you face memory, issue then reduce batch size
    if args.examples>75:
        batch_size=32
    # batch_size=16
    aug = "NO_Aug"
    if args.augmentation==1:
        aug="Aug"
    threshold=0.5 # semi-supervised pseudo labeling threshold 
    accuracies = []
    Exe_time = []
    datagen = None
    if args.augmentation == 1:
        datagen = ImageDataGenerator(rotation_range=5, width_shift_range=0.1,
                                         height_shift_range=0.1,
                                         shear_range=0.05,
                                         zoom_range=0.05,
                                         horizontal_flip=False,
                                         vertical_flip=False,
                                         fill_mode='nearest'
                                         )

    if args.learning=="SS":
        count = 0
        start_time = time.time()
        if not os.path.exists(folder+"/supervised"):
            os.mkdir(folder+"/supervised")
        fl = folder+"/supervised/"+args.model+"_"+aug+"_" + args.learning + "_" + str(args.examples) + "_" + str(count) + ".h5"
        # this loop to deal multi-run file and new model name saving with each run 
		while os.path.exists(fl):
            count += 1
            fl = folder+"/supervised/"+args.model+"_"+aug+"_" + args.learning + "_" + str(args.examples) + "_" + str(count) + ".h5"
        callbacks = ModelCheckpoint(fl, monitor='val_accuracy',
                                    mode='max',
                                    save_best_only=True,
                                    verbose=1)
		# uncomment this if you face tensorflow version 
        # y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
        # selected_y = tf.keras.utils.to_categorical(selected_y, num_classes=10)
        if args.model=="small":
            model = getSmallModel(input_shape=input_shape, classes=classes)
        else:
            model = getLargeModel(input_shape=input_shape, classes=classes)
        model.summary()
        if args.augmentation == 1:
            history = model.fit_generator(datagen.flow(selected_x, selected_y, batch_size=batch_size),
                                          epochs=epochs,
                                          validation_data=(x_val, y_val),
                                          verbose=1, callbacks=[callbacks])
            # history = model.fit(selected_x, selected_y, validation_data=(x_val, y_val), batch_size=batch_size, epochs=100,
            #                     verbose=1, callbacks=[callbacks])
            print(selected_x.shape, selected_y.shape)
            print(x_test.shape, y_test.shape)
            print("SSL With Augmentation testing accuracy ", model.evaluate(x_test, y_test, verbose=1))
        else:
            history = model.fit(selected_x, selected_y,batch_size=batch_size,  epochs=epochs,  verbose=1, callbacks=[callbacks],validation_data=(x_val, y_val))
        model = load_model(fl)

        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        print("SSL With Augmentation testing accuracy ", model.evaluate(x_test, y_test, verbose=1))
        accuracies.append(model.evaluate(x_test, y_test, verbose=1)[1])
        Exe_time.append(time.time()-start_time)
        args.learning="PSSL"
    if args.learning=="PSSL":  # pseudo-labeling 
        start_time = time.time()
        args.learning="FLL"
        x= np.copy(selected_x)
        y= np.copy(selected_y)
        for i in range(2):
            count = 0
            if not os.path.exists(folder + "/PSSL"):
                os.mkdir(folder + "/PSSL")
            # fl = folder + "/PSSL/" + aug + "_" + args.learning + "_" + str(args.examples) + "_" + str(count) + ".h5"
            fl = folder + "/PSSL/" + args.model + "_" + aug + "_" + args.learning + "_" + str(args.examples) + "_" + str(count) + ".h5"
            while os.path.exists(fl):
                count += 1
                fl = folder + "/PSSL/" + args.model + "_" + aug + "_" + args.learning + "_" + str(args.examples) + "_" + str(count) + ".h5"
            callbacks = ModelCheckpoint(fl, monitor='val_accuracy',
                                        mode='max',
                                        save_best_only=True,
                                        verbose=1)
            if args.model == "small":
                model = getSmallModel(input_shape=input_shape, classes=classes)
            else:
                model = getLargeModel(input_shape=input_shape, classes=classes)
            if args.augmentation == 1:
                history = model.fit_generator(datagen.flow(selected_x, selected_y, batch_size=batch_size),
                                              epochs=epochs,
                                              validation_data=(x_val, y_val),
                                              verbose=1, callbacks=[callbacks])
            else:

                history = model.fit(selected_x, selected_y, validation_data=(x_val, y_val), batch_size=batch_size, epochs=epochs,
                                    verbose=1, callbacks=[callbacks])
            model = load_model(fl)

            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

            prediction = model.predict(x_train)

            new_pred = prediction

            # arr[np.any(arr > 3, axis=1)]
            arr = prediction > threshold
            row, col = np.where(arr == True)
            selected_x = np.concatenate((x_train[row,:,:,:],selected_x), axis=0)
            selected_y = np.concatenate((col, selected_y))
            if args.augmentation==1:
                print("With Augmentation Testing accuracy ", model.evaluate(x_test, tf.keras.utils.to_categorical(y_test,num_classes=classes), verbose=1))
            else:
                print("Without Augmentation Testing accuracy ",
                      model.evaluate(x_test, tf.keras.utils.to_categorical(y_test,num_classes=classes), verbose=1))
            if i==0:
                args.learning="PSSL"
        # args.learning="SSL"
        selected_x=np.copy(x)
        selected_y=np.copy(y)
        accuracies.append(model.evaluate(x_test, tf.keras.utils.to_categorical(y_test, num_classes=classes), verbose=1)[1])
        Exe_time.append(time.time() - start_time)
    if args.learning=="SSL":  # proposed BSSL 
        count = 0
        acc= []
        start_time = time.time()
		# setting directories 
        if not os.path.exists(folder+"/SSL"):  
            os.mkdir(folder+"/SSL")
        if not os.path.exists(folder+"/SSL/"+aug+"_"+ args.learning + "_" + str(args.examples)):
            os.mkdir(folder+"/SSL/"+aug+"_" + args.learning + "_" + str(args.examples))
        store_x  = np.copy(x_train)
        store_y  = np.copy(y_train)
        for trial in range(1):   # if you want to repeat more you can increase from 1 to more times 
            status=False
            new_x = None  # to store filtered examples 
            new_y = None
            for i in np.unique(selected_y):  # for each class 
                temp= None
                temp1=None

                #Handling class imbalance problem
                examples_per_class = args.examples
				# for current class examp_per_class *(c-1) with label 1 and same for reamining class but with 0 label
                temp = np.zeros((examples_per_class * 2 * (classes - 1)), dtype=selected_y.dtype)
                temp[0:examples_per_class * (classes - 1)] = 1  # assign 1 to current class, and remaning will be conisdered as 0 
                selected_x1 = np.zeros(
                    (examples_per_class * (classes - 1), selected_x.shape[1], selected_x.shape[2], selected_x.shape[3]),
                    dtype=selected_x.dtype)
                for (class_i_exmp, j) in enumerate(np.unique(y_train)):
                    selected_x1[class_i_exmp * examples_per_class:(class_i_exmp + 1) * examples_per_class, :, :,
                    :] = selected_x[i == selected_y, :, :, :] # for current class i samples  
                for j in random.sample(range(selected_x[selected_y != i].shape[0]),
                                          examples_per_class * (classes - 1)):  # randomly select the samples
                    exmp = selected_x[selected_y != i][j]  # any smaple of any class except i class 
                    exmp = exmp.reshape((1, exmp.shape[0], exmp.shape[1], exmp.shape[2]))
                    # print(selected_x1.shape, exmp.shape)
                    selected_x1 = np.concatenate((selected_x1, exmp), axis=0)

                temp1 = np.zeros((y_val.shape), dtype=y_val.dtype)
                temp1[i == y_val] = 1
				# can be tried other method of class imbalancing problem 
                # selected_x1 = None
                # examples_per_class = args.examples
                # temp = np.zeros((selected_y.shape), dtype=selected_y.dtype)
                # temp[i==selected_y] = 1
                # temp1 = np.zeros((y_val.shape), dtype=y_val.dtype)
                # temp1[i == y_val] = 1
                # le = preprocessing.LabelEncoder()
                # le.fit(temp)
                # class_weights = class_weight.compute_class_weight('balanced',
                #                                                   np.unique(le.transform(temp)),
                #                                                   le.transform(temp))
                # class_weights = {l: c for l, c in zip(np.unique(temp), class_weights)}
                acc = []
                if len(x_train)==0:
                    break
                if args.model == "small":
                    model = getSmallModel(input_shape=input_shape, classes=2)
                else:
                    model = getLargeModel(input_shape=input_shape, classes=2)
                print(i, " out of ", len(np.unique(selected_y)))
                # print(class_weights, np.unique(temp), temp.shape, np.unique(temp1), temp1.shape)
                # fl = folder+"/SSL/"+aug+"_" + args.learning + "_" + str(args.examples)+"/"+str(i)+".h5"
                count=0
                fl = folder+"/SSL/"+aug+"_"+args.learning + "_" + str(args.examples)+"/"+args.model+ "_" + aug + "_" + args.learning + "_" + str(args.examples) + "_" + str(i) +"_"+str(count)+ ".h5"
                while os.path.exists(fl):
                    count+=1
                    fl = folder + "/SSL/" + aug + "_" + args.learning + "_" + str(
                        args.examples) + "/" + args.model + "_" + aug + "_" + args.learning + "_" + str(
                        args.examples) + "_" + str(i) + "_" + str(count) + ".h5"

                callbacks = ModelCheckpoint(fl, monitor='val_accuracy',
                                    mode='max',
                                    save_best_only=True,
                                    verbose=1)
                if args.augmentation==1:

                    history = model.fit_generator(datagen.flow(selected_x1, temp, batch_size=int(batch_size/2)),epochs=int(epochs/2),verbose=1, callbacks=[callbacks],
                                                  validation_data=(x_val,temp1))#,class_weight=class_weights)

                else:
                    history = model.fit(selected_x1, temp,  batch_size=int(batch_size/2),  epochs=int(epochs/2),   verbose=1,validation_data=(x_val,temp1), callbacks=[callbacks])#,class_weight=class_weights)
                model = load_model(fl)
                model.compile(loss="sparse_categorical_crossentropy",  # loss=keras.losses.categorical_crossentropy,
                              optimizer=tf.keras.optimizers.Adagrad(),
                              metrics=['accuracy'])
                prediction = model.predict(x_train)
                print("Train shape ", x_train.shape)
                new_pred = prediction
                predicted_class = 1
                new_shape = x_train[np.argmax(prediction,axis=1)==predicted_class,:,:,:].shape

                if new_shape[0]!=0:
                    if status==False:
                        status=True
                        new_x = np.copy(x_train[np.argmax(prediction,axis=1)==predicted_class,:,:,:])
                        new_y = np.zeros(new_shape[0], dtype=selected_y.dtype)+i
                    else:
                        new_x = np.concatenate((new_x,np.copy(x_train[np.argmax(prediction,axis=1)==predicted_class,:,:,:])),axis=0)
                        new_y = np.concatenate((new_y, np.zeros(new_shape[0], dtype=selected_y.dtype)+i,), axis=0)
                        # Remaining examples
                    x_train = x_train[np.argmax(prediction,axis=1)!=predicted_class,:,:,:]
                    y_train = y_train[np.argmax(prediction, axis=1) != predicted_class]
                    print(new_x.shape, new_y.shape)
                if len(x_train)==0: #x_train.shape[0]==0:
                    break
                print(selected_x.shape, x_train.shape, y_train.shape, selected_y.shape)
            if status==True:
                selected_x = np.concatenate((selected_x, new_x), axis=0)
                selected_y = np.concatenate((selected_y,new_y),axis=0)
          # print(one_Hot)
        if args.model == "small":  # if model is small
            model = getSmallModel(input_shape=input_shape, classes=classes)
        else:
            model = getLargeModel(input_shape=input_shape, classes=classes)
        # selected_y = tf.keras.utils.to_categorical(selected_y, num_classes=10)
        count=0
        fl = folder+"/SSL/" +args.model+ "_" + aug + "_" + args.learning + "_" + str(args.examples) + "_" + str(count) + ".h5"
        while os.path.exists(fl):
            count += 1
            fl = folder+"/SSL/" +args.model+ "_" + aug + "_" + args.learning + "_" + str(args.examples) + "_" + str(count) + ".h5"
        callbacks = ModelCheckpoint(fl, monitor='val_accuracy',
                                    mode='max',
                                    save_best_only=True,
                                    verbose=1)
        if args.augmentation == 1:
            history = model.fit_generator(datagen.flow(selected_x, selected_y, batch_size=batch_size),
                                          epochs=epochs,
                                          verbose=1, callbacks=[callbacks],validation_data=(x_val, y_val))
        else:
            print(selected_x.shape, selected_y.shape)
            print(x_val.shape, y_val.shape)
            history = model.fit(selected_x, selected_y, batch_size=batch_size, epochs=epochs, verbose=1,callbacks=[callbacks],validation_data=(x_val, y_val))
        model = load_model(fl)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        print("SSL Testing accuracy ", model.evaluate(x_test, tf.keras.utils.to_categorical(y_test,num_classes=classes,), verbose=1))
        accuracies.append(model.evaluate(x_test, tf.keras.utils.to_categorical(y_test,num_classes=classes,), verbose=1)[1])
        Exe_time.append(time.time() - start_time)
    dic["Run"+str(run)+" Accuracy"] = accuracies
    dic["Run"+str(run)+" Time"] = Exe_time

print(accuracies)
print(Exe_time)
print(dic)

df= pd.DataFrame(dic)
df1=None
df2=None
if os.path.exists("/root/volume/DataSets/DataSets/AccuracyFiles/"+args.dataset+"_"+args.model+"_"+str(args.examples)+"_"+str(args.augmentation)+"_.csv"):
    df1 = pd.read_csv("/root/volume/DataSets/DataSets/AccuracyFiles/"+args.dataset+"_"+args.model+"_"+str(args.examples)+"_"+str(args.augmentation)+"_.csv")
    df2= pd.concat([df1,df],axis=0)
else:
    df2=df
df2.to_csv("/root/volume/DataSets/DataSets/AccuracyFiles/"+args.dataset+"_"+args.model+"_"+str(args.examples)+"_"+str(args.augmentation)+"_.csv")