import hw3_corpus_tool as taTool
import pycrfsuite
import sys
import glob
import os

def generateFeatures(dialog,isChangeSpeaker,isFirstUtterance):
    feature = [
            "isChangeSpeaker={}".format(isChangeSpeaker),
            "isFirstUtterance={}".format(isFirstUtterance)
        ]
    if(dialog.pos):
        for index,posTag in enumerate(dialog.pos):
            feature = feature + [
                "token.{}={}".format(index,posTag.token),
                "pos.{}={}".format(index,posTag.pos)
            ]
    else:
        feature = feature + [
            "token.0={}".format("UNDEFINED"),
            "pos.0={}".format("UNDEFINED")
        ]
    return(feature)

def extractFeaturesAndLabels(inputFolder):
    dialogueCorpus = taTool.get_data(inputFolder)

    dialogCorpusFeature = []
    dialogCorpusLabel = []

    for dialogSet in dialogueCorpus:
        dialogSetFeature = []
        dialogSetLabel = []
        previousSpeaker = None
        currentSpeaker = None
        for index,dialog in enumerate(dialogSet):
            actTag = dialog.act_tag
            currentSpeaker = dialog.speaker

            feature = generateFeatures(dialog, (currentSpeaker == previousSpeaker), index == 0)

            previousSpeaker = currentSpeaker

            dialogSetFeature = dialogSetFeature + [feature]
            dialogSetLabel = dialogSetLabel + [actTag]

        dialogCorpusFeature = dialogCorpusFeature + [dialogSetFeature]
        dialogCorpusLabel = dialogCorpusLabel + [dialogSetLabel]
    return(dialogCorpusFeature,dialogCorpusLabel)

if __name__ == "__main__":
    '''
    usage: python3 baseline_crf.py ./tmp/train ./tmp/test ./tmp/result.txt

    '''
    inputFolder = sys.argv[1]
    testFolder = sys.argv[2]
    outputFile = sys.argv[3]

    dialogCorpusFeature, dialogCorpusLabel = extractFeaturesAndLabels(inputFolder)

    #train CRF
    crfModel = pycrfsuite.Trainer(verbose=False)
    for xSeq,ySeq in zip(dialogCorpusFeature,dialogCorpusLabel):
        crfModel.append(xSeq,ySeq)

    crfModel.train('crfModel.crfsuite')

    #tag test data
    crfTagger = pycrfsuite.Tagger()
    crfTagger.open('crfModel.crfsuite')

    dialogCorpusFeature, dialogCorpusLabel = extractFeaturesAndLabels(testFolder)
    testFileNames = sorted(glob.glob(os.path.join(testFolder, "*.csv")))

    writeContent = ""
    for index,testFileName in enumerate(testFileNames):
        writeContent += testFileName.split("/")[-1] + "\n"
        writeContent += "\n".join(crfTagger.tag(dialogCorpusFeature[index]))
        writeContent += "\n\n"

    writer = open(outputFile, "w", encoding="latin1")
    writer.write(writeContent)
    writer.close()
