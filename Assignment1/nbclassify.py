import sys
import os
from os.path import isdir, join, isfile
import json
import math

def readFolder(folderPath,folderName):
    global all_files
    #read .txt files
    fs = [join(folderPath,folderName,f) for f in os.listdir(join(folderPath,folderName)) if isfile(join(folderPath,folderName,f)) and f[f.rfind(".")+1:] == "txt"]
    all_files = all_files + fs
    #then read all sub folders
    dirs = [d for d in os.listdir(join(folderPath,folderName)) if isdir(join(folderPath,folderName,d))]
    for d in dirs:
        readFolder(join(folderPath,folderName),d)

def calculateProb(classProb,dictionary,anotherDictionary,wordcount,vocabSize,content):
    tokens = content.split()
    #do add-one smoothing only if the word appeared in at least one dictionary
    counts = [dictionary[token] if token in dictionary else 0 if token in anotherDictionary else -1 for token in tokens]
    counts = [c for c in counts if c != -1]
    
    logodds = [ math.log((c+1)/(wordcount + vocabSize)) for c in counts]
    sumLogOdds = sum(logodds) + math.log(classProb)
    return sumLogOdds

#usage: nbclassify.py test_folder_path
#Example:
#$ python3 nbclassify.py /home/vitidn/mydata/repo_git/CSCI544/Assignment1/data/dev
if __name__ == "__main__":
    #read the model's parameter
    with open('nbmodel.txt') as parameterfile:
        parameters = json.load(parameterfile)

    hamDict = parameters["ham"]
    spamDict = parameters["spam"]

    hamCount = parameters["hamCount"]
    spamCount = parameters["spamCount"]

    distWords = list(hamDict.keys()) + list(spamDict.keys())
    distWords = list(set(distWords))
    vocabSize = len(distWords)

    hamWordCount = sum(list(hamDict.values()))
    spamWordCount = sum(list(spamDict.values()))  

    #files to be predicted
    all_files = []

    rootFolder = sys.argv[1]
    rIndex = rootFolder.rfind("/")
    #read all files to predict into all_files
    readFolder(rootFolder[0:rIndex],rootFolder[rIndex+1:])
    predictedLabels = []

    #do the prediction
    hamClassProb = hamCount / (hamCount + spamCount)
    spamClassProb = spamCount / (hamCount + spamCount)
    for f in all_files:
        try:
            filestream = open(f,"r",encoding="latin1")
            content = filestream.read()
            probHam = calculateProb(hamClassProb,hamDict,spamDict,hamWordCount,vocabSize,content)
            probSpam = calculateProb(spamClassProb,spamDict,hamDict,spamWordCount,vocabSize,content)
            if(probHam > probSpam):
                predictedLabels = predictedLabels + ["ham"]
            else:
                predictedLabels = predictedLabels + ["spam"]
        except :
            print("Could not process file {0}".format(f))
        finally:
            filestream.close()

    #write out the result
    writeContent = ""
    for result in zip(all_files,predictedLabels):
        writeContent = writeContent + "{0} {1}".format(result[1],result[0]) + "\n"

    outputfile = open("nboutput.txt","w")
    outputfile.write(writeContent)
    outputfile.close()
