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

def predict(weights,b,filepath,fileFeatureDict):

    wordCounts = fileFeatureDict[filepath]

    alpha = 0    
    for word, wordCount in wordCounts.items():
        if(word in weights):
            wordWeight = weights[word]
        else:
            wordWeight = 0
                
        alpha += wordWeight*wordCount
    prediction = alpha + b   

    return "spam" if prediction > 0 else "ham"

def readAllWordFeatures(filelpaths):
    fileFeatureDict = {}

    for f in filelpaths:
        wordFeatures = readWordFeatures(f)
        fileFeatureDict[f] = wordFeatures
    return fileFeatureDict

def readWordFeatures(filepath):
    wordFeatures = {}

    try:
        filestream = open(filepath,"r",encoding="latin1")
        content = filestream.read()
        tokens = content.split()
        for token in tokens:
            if(token in wordFeatures):
                wordFeatures[token] = wordFeatures[token] + 1
            else:
                wordFeatures[token] = 1 
    except:
        print("Could not process file {0}".format(f))
    finally:
        filestream.close()
    return wordFeatures

#usage: per_classify.py test_folder_path output_filename
#Example:
#$ python3 per_classify.py /home/vitidn/mydata/repo_git/CSCI544/Assignment1/data/dev predict_result.txt
if __name__ == "__main__":

    #read the model's parameter
    with open('per_model.txt') as parameterfile:
        parameters = json.load(parameterfile)

    weights = parameters[0]
    b = parameters[1]

    #files to be predicted
    all_files = []

    rootFolder = sys.argv[1]
    rIndex = rootFolder.rfind("/")
    #read all files to predict into all_files
    readFolder(rootFolder[0:rIndex],rootFolder[rIndex+1:])
    #cache all file's features
    fileFeatureDict =readAllWordFeatures(all_files)
    predictedLabels = []

    #do the prediction
    for f in all_files:
        try:
            predictedLabel = predict(weights,b,f,fileFeatureDict)
            predictedLabels = predictedLabels + [predictedLabel]
        except :
            print("Could not process file {0}".format(f))
            predictedLabels = predictedLabels + ["N/A"]

    #write out the result

    writeFilePath = sys.argv[2]
    
    writeContent = ""
    for result in zip(all_files,predictedLabels):
        writeContent = writeContent + "{0} {1}".format(result[1],result[0]) + "\n"

    outputfile = open(writeFilePath,"w", encoding="latin1")
    outputfile.write(writeContent)
    outputfile.close()
