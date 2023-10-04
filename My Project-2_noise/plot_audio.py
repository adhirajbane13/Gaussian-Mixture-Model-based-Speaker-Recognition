import pickle as cPickle
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import wiener
import librosa as lib
from sklearn import mixture
from feature_extr import extract_features
import warnings
warnings.filterwarnings("ignore")

source   = "Train-Data/" #Here, the training data is saved
dest = "Sp_Models/"
train_file = "Traindatapath.txt" #Training data path is stored in text file       
file_paths = open(train_file,'r')


audio,sr = lib.load(source + 'Adhiraj-speech/Adhiraj-1.wav')
plt.subplot(2,2,1)
plt.plot(audio)
plt.xlabel('time')
plt.ylabel('audio signal')
plt.title('audio without noise')
j = 2
A = [8,6,4]
for i in A:
    noise = np.random.normal(0,np.sqrt(pow(10,-i)),audio.shape)
    new_audio = audio + noise
    plt.subplot(2,2,j)
    plt.plot(new_audio)
    plt.xlabel('time')
    plt.ylabel('audio signal')
    plt.title('audio with noise of variance 10^-'+str(i))
    j+=1
plt.show()
