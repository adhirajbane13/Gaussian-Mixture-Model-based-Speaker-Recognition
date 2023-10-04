import numpy as np
from sklearn import preprocessing
import librosa as lib

#Determining 20 dimensional MFCC features and 20 dimensional delta MFCC features
def extract_features(audio,rate):
    mfcc_feature = lib.feature.mfcc(audio,rate,n_mfcc = 20,hop_length=int(0.010 * rate), n_fft = int(0.025 * rate))
    mfcc_feature = preprocessing.scale(mfcc_feature)
    delta = lib.feature.delta(mfcc_feature, order = 1)
    mfcc_feature = np.transpose(mfcc_feature)
    delta = np.transpose(delta)
    combined = np.hstack((mfcc_feature,delta))#stacking to form 40 dimensional MFCC & delta MFCC features
    return combined
