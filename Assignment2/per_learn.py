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
    read word frequency of all document and collect all distinct words
    """
    global distinctWords
    fileFeatureDict = {}

    for t in labelFilePathTuples:
        wordFeatures = readWordFeatures(t)
        fileFeatureDict[t[1]] = wordFeatures
    return fileFeatureDict

def readWordFeatures(labelFilePathTuple):
    global distinctWords
    wordFeatures = {}
    f = labelFilePathTuple[1]
    print("reading:" + f)
    try:
        filestream = open(f,"r",encoding="latin1")
        content = filestream.read()
        tokens = content.split()
        wordFeatures = dict(Counter(tokens))

        #distinctWords = distinctWords.union(set(wordFeatures.keys())) 
    except:
        print("Could not process file {0}".format(f))
    finally:
        filestream.close()
    return wordFeatures

def trainPerceptron(labelFilePathTuples,fileFeatureDict,distinctWords,maxIteration=20):
    """
    For Ham, set Label = -1, for Spam, set Label = +1

    return (weights,b)
    """
    weights = {w:0 for w in distinctWords}
    b = 0
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
                if(word in weights):
                    wordWeight = weights[word]
                else:
                    wordWeight = 0
                
                alpha += wordWeight*wordCount
            #alpha is a prediction result
            alpha += b
            if(trueLabel * alpha <= 0):
                for word in wordCounts.keys():
                    if(word in weights):
                        weights[word] = weights[word] + trueLabel*wordCounts[word]
                b = b + trueLabel

    return (weights,b)
            
#usage: per_learn.py train_folder_path [downsamplingRatio]
#Example:
#$ python3 per_learn.py /home/vitidn/mydata/repo_git/CSCI544/Assignment1/data/train
#$ python3 per_learn.py /home/vitidn/mydata/repo_git/CSCI544/Assignment1/data/train 0.1
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

        ham_files = random.sample(ham_files,numTrainFiles)
        spam_files = random.sample(spam_files,numTrainFiles)

    hamCount = len(ham_files)
    spamCount = len(spam_files)
    
    spam_files = [("spam",filepath) for filepath in spam_files]
    ham_files = [("ham",filepath) for filepath in ham_files]
    labelFilePathTuples = spam_files + ham_files
    distinctWords = set()
    #read features of all files(cached) + read distinct words(distinctWords)
    print("start reading files...")
    fileFeatureDict = readAllWordFeatures(labelFilePathTuples)
    print("finish reading files...")
    parameters = trainPerceptron(labelFilePathTuples,fileFeatureDict,distinctWords)

    with open('per_model.txt','w') as writefile:
        json.dump(parameters,writefile)
