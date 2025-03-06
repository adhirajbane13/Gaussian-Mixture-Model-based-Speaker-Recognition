import os
import _pickle as cPickle
import numpy as np
from scipy.io.wavfile import read
import librosa as lib
from sklearn import preprocessing
from feature_extr import extract_features
import warnings
warnings.filterwarnings("ignore")
import time

#path to Test data
source="Test-Data/"

#path where Training speakers' Models will be saved
modelpath="Sp_Models/"

gmm_files=[os.path.join(modelpath,fname) for fname in os.listdir(modelpath) if fname.endswith('.gmm')]

#Load the Gaussian Mixture Models
models=[cPickle.load(open(fname,'rb')) for fname in gmm_files]
speakers=[fname.split("/")[-1].split(".gmm")[0] for fname in gmm_files]
error = 0
total_sample = 0.0


take = int(input("Do you want to Test a Single Audio: Press '1' or The complete Test Audio Sample: Press '0' ? "))
if take == 1:
        path = input("Enter the File name from Test Audio Sample Collection : ")
        print("Testing Audio : ", path)
        audio,sr = lib.load(source + path)
        vector   = extract_features(audio,sr)
        log_likelihood = np.zeros(len(models))
        for i in range(len(models)):
                gmm = models[i]  #checking with each model one by one
                scores = np.array(gmm.score(vector))
                log_likelihood[i] = scores.sum()


        winner = np.argmax(log_likelihood)
        print ("\tdetected as - ", speakers[winner])
        if speakers[winner] == path.split("_")[0]:
              print("Hurray ! Speaker identified. Mission Accomplished Successfully.")
        else:
              print("Speaker Identification Failed")  
        time.sleep(1.0)
elif take == 0:
        test_file = "Test_Datapath.txt"
        file_paths = open(test_file,'r')
        # Read the test directory and get the list of test audio files
        for path in file_paths:
                total_sample += 1.0
                path = path.strip()
                print ("Testing Audio : ", path)
                audio,sr = lib.load(source + path)
                noise = np.random.normal(0,np.sqrt(pow(10,0)),audio.shape)
                audio = audio + noise
                vector  = extract_features(audio,sr)

                log_likelihood = np.zeros(len(models))

                for i in range(len(models)):
                        gmm    = models[i]  #checking with each model one by one
                        scores = np.array(gmm.score(vector))
                        log_likelihood[i] = scores.sum()

                winner = np.argmax(log_likelihood)
                print("\tdetected as - ", speakers[winner])

                checker_name = path.split("_")[0]
                if speakers[winner] != checker_name:
                        error += 1
        time.sleep(1.0)

        print("Error and Sample No. respectively:",error,"&", total_sample)
        accuracy = ((total_sample - error) / total_sample) * 100

        print("The Accuracy Percentage for the current testing Performance with MFCC + GMM is : ", accuracy, "%")
        if accuracy == 100:
              print("Hurray ! Speakers identified. Mission Accomplished Successfully.")
        else:
              print("Speaker Identification Failed")      

 

