import sys
import os
from os.path import isdir, join, isfile
import random
import json

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

def trainClass(dictionary,filepaths):
    for f in filepaths:
        try:
            filestream = open(f,"r",encoding="latin1")
            content = filestream.read()
            tokens = content.split()
            for token in tokens:
                if(token in dictionary):
                    dictionary[token] = dictionary[token] + 1
                else:
                    dictionary[token] = 1
        except:
            print("Could not process file {0}".format(f))
        finally:
            filestream.close()

#usage: nblearn.py train_folder_path [downsamplingRatio]
#Example:
#$ python3 nblearn.py /home/vitidn/mydata/repo_git/CSCI544/Assignment1/data/train
#$ python3 nblearn.py /home/vitidn/mydata/repo_git/CSCI544/Assignment1/data/train 0.1
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

    hamDict = {}
    spamDict = {}

    trainClass(hamDict,ham_files)
    trainClass(spamDict,spam_files)

    #save the model's parameters
    modelParameters = {
        "ham":hamDict,
        "spam":spamDict,
        "hamCount":hamCount,
        "spamCount":spamCount
    }

    with open('nbmodel.txt','w') as writefile:
        json.dump(modelParameters,writefile)
