import pickle as cPickle
import numpy as np
from scipy.io.wavfile import read
from sklearn import mixture
from featureextraction import extract_features
import warnings
warnings.filterwarnings("ignore")

source   = "Train-Data/"
dest = "Sp_Models/"
train_file = "Traindatapath.txt"        
file_paths = open(train_file,'r')

count = 1
# Extracting features for each speaker (3 files per speaker)
features = np.asarray(())
for path in file_paths:    
    path = path.strip()   
    print (path)
    
    # read the audio
    sr,audio = read(source + path)
    
    # extract 40 dimensional MFCC & delta MFCC features
    vector   = extract_features(audio,sr)
    
    if features.size == 0:
        features = vector
    else:
        features = np.vstack((features, vector))
    if count == 3:    
        gmm = mixture.GaussianMixture(n_components = 20, max_iter = 500, covariance_type='diag',n_init = 3)
        gmm.fit(features)
        
        # dumping the trained gaussian model
        picklefile = path.split("-")[0]+".gmm"
        cPickle.dump(gmm,open(dest + picklefile,'wb'))
        print ('+ modeling completed for speaker:',picklefile," with data point = ",features.shape)    
        features = np.asarray(())
        count = 0
    count = count + 1
