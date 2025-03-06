import pickle as cPickle
import matplotlib.pyplot as plt
import numpy as np
from scipy.io.wavfile import read
import librosa as lib
from sklearn import mixture
from feature_extr import extract_features
import warnings
warnings.filterwarnings("ignore")

source   = "Train-Data/" #Here, the training data is saved
dest = "Sp_Models/"
train_file = "Traindatapath.txt" #Training data path is stored in text file       
file_paths = open(train_file,'r')

count = 1
# Extracting features for each speaker (3 files per speaker)
features = np.asarray(())
for path in file_paths:    
    path = path.strip()   
    print ("Reading Audio:",path)
    
    # read the audio to obtain audio signal and sample rate
    audio,sr = lib.load(source + path)
    print(sr)
    
    # extract 40 dimensional MFCC & delta MFCC features
    vector   = extract_features(audio,sr)
    
    if features.size == 0:
        features = vector
    else:
        features = np.vstack((features, vector))#Stack of feature vectors
    if count == 3:  #After loading 3 files for each speaker, the trained gaussian model is created for each of them from their extracted features  
        gmm = mixture.GaussianMixture(n_components = 16, max_iter = 200, covariance_type='diag',n_init = 3)
        gmm = gmm.fit(features)
        
        # dumping the trained gaussian model
        picklefile = path.split("-")[0]+".gmm"
        cPickle.dump(gmm,open(dest + picklefile,'wb'))
        print ('+ modeling completed for speaker:',path.split("-")[0]," with data point = ",features.shape)    
        features = np.asarray(())
        count = 0
    count = count + 1
