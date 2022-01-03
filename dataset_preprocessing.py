import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import cv2
from sklearn.model_selection import train_test_split

def CalculateMelSpectrogram(file_location):
  print(file_location)
  y, sr = librosa.load(file_location)
  melSpec = librosa.feature.melspectrogram(y=y, sr=sr)
  melSpec_dB = librosa.power_to_db(melSpec, ref=np.max)
  dim = (64, 64)
  resized = cv2.resize(melSpec_dB, dim, interpolation = cv2.INTER_AREA)
  return resized
def PrepareDataset():
  ## Dictionary preparing for labels
  dic= {}
  count=0
  mainPath= "./GoogleSpeechCommands"

  for i in os.listdir(mainPath):
    dic[i]=count
    count+=1
  ###### ##
  testpath=mainPath+"/testing_list.txt"
  valpath=mainPath+"/validation_list.txt"
  testpaths = open(testpath,"r").readlines()
  valpaths = open(valpath,"r").readlines()
  testpaths = [mainPath+"/"+i for i in testpaths]
  valpaths = [mainPath+"/"+i for i in valpaths]
  trainpaths = []
  for i in os.listdir(mainPath):
    if i.endswith(".txt"):
      continue
    for j in os.listdir(mainPath+"/"+i):
      trainingfile = mainPath+"/"+i+"/"+j
      if trainingfile in testpaths or trainingfile in valpaths:
        continue
      trainpaths.append(trainingfile)
  x_train, y_train, x_test, y_test, x_val, y_val = [],[],[],[],[],[]
  bg=[]
  count=0
  for i in trainpaths:
    count+=1
    print("Processed ", count," / ", len(trainpaths))
    x_train.append(CalculateMelSpectrogram(i))
    y_train.append(dic[i.split("/")[2]])
  count=0
  for i in testpaths:
    count += 1
    print("Processed ", count, " / ", len(testpaths))
    if "\n" in i:
      i = i.replace("\n","")
    x_test.append(CalculateMelSpectrogram(i))
    y_test.append(dic[i.split("/")[2]])
  count=0
  for i in valpaths:
    count += 1
    print("Processed ", count, " / ", len(valpaths))
    if "\n" in i:
      i = i.replace("\n","")
    x_val.append(CalculateMelSpectrogram(i))
    y_val.append(dic[i.split("/")[2]])
  # bgNse = []
  # for bg in backgroundNoise:
  #   bgNse.append(CalculateMelSpectrogram(bg))
  #
  return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test), np.array(x_val), np.array(y_val)
x_train, y_train, x_test, y_test, x_val, y_val  = PrepareDataset()
np.save("/root/volume/DataSets/PreparedDatasets/GoogleCommands/x_train", x_train)
np.save("/root/volume/DataSets/PreparedDatasets/GoogleCommands/y_train", y_train)
np.save("/root/volume/DataSets/PreparedDatasets/GoogleCommands/x_test", x_test)
np.save("/root/volume/DataSets/PreparedDatasets/GoogleCommands/y_test", y_test)
np.save("/root/volume/DataSets/PreparedDatasets/GoogleCommands/x_val", x_val)
np.save("/root/volume/DataSets/PreparedDatasets/GoogleCommands/y_val", y_val)
# x_train = np.load("googleCommandDataset/x_train.npy",allow_pickle=True)
# y_train = np.load("googleCommandDataset/y_train.npy",allow_pickle=True)
# x_test = np.load("googleCommandDataset/x_test.npy",allow_pickle=True)
# y_test = np.load("googleCommandDataset/y_test.npy",allow_pickle=True)
# x_val = np.load("googleCommandDataset/x_val.npy",allow_pickle=True)
# y_val = np.load("googleCommandDataset/y_val.npy",allow_pickle=True)
# x_train = x_train.reshape((x_train.shape[0],32,32,1))
# x_test = x_test.reshape((x_test.shape[0],32,32,1))
# x_val = x_val.reshape((x_val.shape[0],32,32,1))
#
# print(x_train.shape, y_train.shape, x_test.shape, y_test.shape, x_val.shape, y_val.shape)
# print(np.unique(y_train))
def PrepareMNISTData():
  path= "/root/BSSL/DataSets/MNIST/"
  x =[]
  y= []
  for i in os.listdir(path):
    x.append(CalculateMelSpectrogram(path+i))
    y.append(int(i.split("_")[0]))
  x = np.array(x)
  x=x.reshape((x.shape[0],32,32,1))
  y= np.array(y)
  x_train, y_train, x_test, y_test = None,None, None, None
  status=True
  for i in np.unique(y):
    X_train, X_test, y_tra, y_te = train_test_split(x[i==y], y[i==y], test_size = 0.2,random_state = 42)
    if status:
      status=False
      x_train,  x_test, y_train, y_test = X_train, X_test, y_tra, y_te
    x_train = np.concatenate((x_train, X_train), axis=0)
    y_train = np.concatenate((y_train, y_tra),axis=0)
    x_test = np.concatenate((x_test, X_test), axis=0)
    y_test = np.concatenate((y_test, y_te),axis=0)

  return x_train, y_train, x_test, y_test

# x_train, y_train, x_test, y_test= PrepareMNISTData()

# np.save("/root/BSSL/DataSets/PreparedDatasets/MNIST/x_train", x_train)
# np.save("/root/BSSL/DataSets/PreparedDatasets/MNIST/y_train", y_train)
# np.save("/root/BSSL/DataSets/PreparedDatasets/MNIST/x_test", x_test)
# np.save("/root/BSSL/DataSets/PreparedDatasets/MNIST/y_test", y_test)
x_train = np.load("/root/BSSL/DataSets/PreparedDatasets/MNIST/x_train.npy",allow_pickle=True)
y_train = np.load("/root/BSSL/DataSets/PreparedDatasets/MNIST/y_train.npy",allow_pickle=True)
x_test = np.load("/root/BSSL/DataSets/PreparedDatasets/MNIST/x_test.npy",allow_pickle=True)
y_test = np.load("/root/BSSL/DataSets/PreparedDatasets/MNIST/y_test.npy",allow_pickle=True)
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
print(np.unique(y_train), np.unique(y_test))
# x_train, y_train, x_test, y_test=PrepareMNISTData()
print(np.unique(y_train),np.unique(y_test))
# x_train, y_train, x_test, y_test, x_val, y_val = PrepareDataset()
# np.save("/root/BSSL/DataSets/PreparedDatasets/GoogleCommands/x_train", x_train)
# np.save("/root/BSSL/DataSets/PreparedDatasets/GoogleCommands/y_train", y_train)
# np.save("/root/BSSL/DataSets/PreparedDatasets/GoogleCommands/x_test", x_test)
# np.save("/root/BSSL/DataSets/PreparedDatasets/GoogleCommands/y_test", y_test)
# np.save("/root/BSSL/DataSets/PreparedDatasets/GoogleCommands/x_val", x_val)
# np.save("/root/BSSL/DataSets/PreparedDatasets/GoogleCommands/y_val", y_val)
x_train = np.load("/root/BSSL/DataSets/PreparedDatasets/GoogleCommands/x_train.npy", allow_pickle=True)
y_train = np.load("/root/BSSL/DataSets/PreparedDatasets/GoogleCommands/y_train.npy", allow_pickle=True)
x_test = np.load("/root/BSSL/DataSets/PreparedDatasets/GoogleCommands/x_test.npy", allow_pickle=True)
y_test = np.load("/root/BSSL/DataSets/PreparedDatasets/GoogleCommands/y_test.npy",allow_pickle=True )
x_val = np.load("/root/BSSL/DataSets/PreparedDatasets/GoogleCommands/x_val.npy", allow_pickle=True )
y_val = np.load("/root/BSSL/DataSets/PreparedDatasets/GoogleCommands/y_val.npy", allow_pickle=True)

print(x_train.shape,y_train.shape, x_test.shape, y_test.shape, x_val.shape, y_val.shape)
print("Total Samples ", x_train.shape[0]+x_test.shape[0]+x_val.shape[0])