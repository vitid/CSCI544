import sys
import os
from os.path import isdir, join, isfile
import random
import json
from random import shuffle
from collections import Counter

def readHamOrSpam(type,folderPath):
    global ham_files
    global spam_files
    fs = [join(folderPath,f) for f in os.listdir(folderPath) if isfile(join(folderPath,f))]
    if(type=="ham"):
        ham_files = ham_files + fs
    else:
        spam_files = spam_files + fs


def readFolder(folderPath,folderName):
    if(folderName == "ham"):
        readHamOrSpam("ham",join(folderPath,folderName))
    if(folderName == "spam"):
        readHamOrSpam("spam",join(folderPath,folderName))
    
    else:
        dirs = [d for d in os.listdir(join(folderPath,folderName)) if isdir(join(folderPath,folderName,d))]
        for d in dirs:
            readFolder(join(folderPath,folderName),d)

def readAllWordFeatures(labelFilePathTuples):
    """
    read word frequency of all document
    """
    fileFeatureDict = {}

    for t in labelFilePathTuples:
        wordFeatures = readWordFeatures(t)
        fileFeatureDict[t[1]] = wordFeatures
    return fileFeatureDict

def readWordFeatures(labelFilePathTuple):
    wordFeatures = {}
    f = labelFilePathTuple[1]
    try:
        filestream = open(f,"r",encoding="latin1")
        content = filestream.read()
        tokens = content.split()
        wordFeatures = dict(Counter(tokens))
    except:
        print("Could not process file {0}".format(f))
    finally:
        filestream.close()
    return wordFeatures

def trainPerceptron(labelFilePathTuples,fileFeatureDict,maxIteration=30):
    """
    For Ham, set Label = -1, for Spam, set Label = +1

    return (weights,b)
    """
    weights = {}
    b = 0

    #For averaged Perceptron
    us = {}
    beta = 0
    c = 1

    for i in range(0,maxIteration):
        #randomize file index
        shuffle(labelFilePathTuples)

        for t in labelFilePathTuples:
            #f - filepath
            f = t[1]
            
            if t[0] == "ham":
                trueLabel = -1
            else:
                trueLabel = 1

            wordCounts = fileFeatureDict[f]

            alpha = 0    
            for word, wordCount in wordCounts.items():
                if(word not in weights):
                    weights[word] = 0
                    us[word] = 0
                wordWeight = weights[word]
                
                alpha += wordWeight*wordCount
            #alpha is a prediction result
            alpha += b
            if(trueLabel * alpha <= 0):
                for word in wordCounts.keys():
                    weights[word] = weights[word] + trueLabel*wordCounts[word]
                    us[word] = us[word] + trueLabel*c*wordCounts[word]
                b = b + trueLabel
                beta = beta + trueLabel*c
            c += 1

    #averaging parameters
    for word in us:
        us[word] = weights[word] - (1/c)*us[word]
    beta = b - (1/c)*beta
    return (us,beta)
            
#usage: avg_per_learn.py train_folder_path [downsamplingRatio]
#Example:
#$ python3 avg_per_learn.py /home/vitidn/mydata/repo_git/CSCI544/Assignment1/data/train
#$ python3 avg_per_learn.py /home/vitidn/mydata/repo_git/CSCI544/Assignment1/data/train 0.1
if __name__ == "__main__":
    #list of all "HAM" fiels
    ham_files = []
    #list of all "SPAM" files
    spam_files = []

    rootFolder = sys.argv[1]
    if(len(sys.argv)==3):
        downsamplingRatio = float(sys.argv[2])
        #for reproducibility
        random.seed(100)
    else:
        downsamplingRatio = 1.0

    rIndex = rootFolder.rfind("/")
    readFolder(rootFolder[0:rIndex],rootFolder[rIndex+1:])

    if(downsamplingRatio < 1.0):
        #number of all files
        numFiles = len(ham_files) + len(spam_files)
        #select only {downsamplingRatio}% to train, Pick half between Spam and Ham
        numTrainFiles = int(round(downsamplingRatio*numFiles*0.5))
        #for a case that a number of Ham/Spam is less than 5% of data
        numTrainFiles = min(numTrainFiles,len(ham_files),len(spam_files))
        
        ham_files = random.sample(ham_files,numTrainFiles)
        spam_files = random.sample(spam_files,numTrainFiles)

    hamCount = len(ham_files)
    spamCount = len(spam_files)
    
    spam_files = [("spam",filepath) for filepath in spam_files]
    ham_files = [("ham",filepath) for filepath in ham_files]
    labelFilePathTuples = spam_files + ham_files
    #read features of all files(cached)
    fileFeatureDict = readAllWordFeatures(labelFilePathTuples)
    parameters = trainPerceptron(labelFilePathTuples,fileFeatureDict)

    with open('per_model.txt','w',encoding="latin1") as writefile:
        json.dump(parameters,writefile)
