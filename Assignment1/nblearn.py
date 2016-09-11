import sys
import os
from os.path import isdir, join, isfile
import random
import json

ham_files = []
spam_files = []

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

rootFolder = sys.argv[1]
rIndex = rootFolder.rfind("/")
readFolder(rootFolder[0:rIndex],rootFolder[rIndex+1:])

#number of all files
numFiles = len(ham_files) + len(spam_files)
#select only 10% to train, Pick half between Spam and Ham
numTrainFiles = int(round(0.1*numFiles*0.5))

ham_files = random.sample(ham_files,numTrainFiles)
spam_files = random.sample(spam_files,numTrainFiles)

hamDict = {}
spamDict = {}

def trainClass(dictionary,filepaths):
    for f in filepaths:
        filestream = open(f,"r",encoding="latin1")
        content = filestream.read()
        tokens = content.split()
        for token in tokens:
            if(token in dictionary):
                dictionary[token] = dictionary[token] + 1
            else:
                dictionary[token] = 1

trainClass(hamDict,ham_files)
trainClass(spamDict,spam_files)

#save the model's parameters
modelParameters = {
    "ham":hamDict,
    "spam":spamDict
}

with open('nbmodel.txt','w') as writefile:
    json.dump(modelParameters,writefile)
