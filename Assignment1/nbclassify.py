import sys
import os
from os.path import isdir, join, isfile
import json
import math

with open('nbmodel.txt') as parameterfile:
    parameters = json.load(parameterfile)

hamDict = parameters["ham"]
spamDict = parameters["spam"]

distWords = list(hamDict.keys()) + list(spamDict.keys())
distWords = list(set(distWords))
vocabSize = len(distWords)

hamWordCount = sum(list(hamDict.values()))
spamWordCount = sum(list(spamDict.values()))  

all_files = []
def readFolder(folderPath,folderName):
    global all_files
    #read .txt files
    fs = [join(folderPath,folderName,f) for f in os.listdir(join(folderPath,folderName)) if isfile(join(folderPath,folderName,f)) and f[f.rfind(".")+1:] == "txt"]
    all_files = all_files + fs
    #then read all sub folders
    dirs = [d for d in os.listdir(join(folderPath,folderName)) if isdir(join(folderPath,folderName,d))]
    for d in dirs:
        readFolder(join(folderPath,folderName),d)

def calculateProb(dictionary,wordcount,vocabSize,content):
    tokens = content.split()
    counts = [dictionary[token] if token in dictionary else 0 for token in tokens]
    logodds = [ math.log((c+1)/(wordcount + vocabSize)) for c in counts]
    sumLogOdds = sum(logodds)
    return sumLogOdds

rootFolder = sys.argv[1]
rIndex = rootFolder.rfind("/")
#read all files to predict into all_files
readFolder(rootFolder[0:rIndex],rootFolder[rIndex+1:])
predictedLabels = []

for f in all_files:
    try:
        filestream = open(f,"r",encoding="latin1")
        content = filestream.read()
        probHam = calculateProb(hamDict,hamWordCount,vocabSize,content)
        probSpam = calculateProb(spamDict,spamWordCount,vocabSize,content)
        if(probHam > probSpam):
            predictedLabels = predictedLabels + ["ham"]
        else:
            predictedLabels = predictedLabels + ["spam"]
    except :
        print("Could not process file {0}".format(f))
    finally:
        filestream.close()

writeContent = ""
for result in zip(all_files,predictedLabels):
    writeContent = writeContent + "{0} {1}".format(result[1],result[0]) + "\n"

outputfile = open("nboutput.txt","w")
outputfile.write(writeContent)
outputfile.close()
