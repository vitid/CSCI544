import hw3_corpus_tool as taTool
import pycrfsuite
import sys
import glob
import os
import math
from random import shuffle
import operator

# def generateFeatures(dialog,isChangeSpeaker,isFirstUtterance,positionConversation,numContinueSpeak,nextConversationLength,isLastUtterance):
def generateFeatures(dialog, isChangeSpeaker, isFirstUtterance):
    feature = [
            "isChangeSpeaker={}".format(isChangeSpeaker),
            "isFirstUtterance={}".format(isFirstUtterance),

            # "isLastUtterance={}".format(isLastUtterance),
            # "positionConversation={}".format(positionConversation),
            # "numContinueSpeak={}".format(numContinueSpeak>0)
        ]

    # numWords = len(dialog.pos) if dialog.pos else 0
    # numWords = "large" if numWords > 10 else "medium" if numWords > 5 else "small"

    # lastWord = "" if not (dialog.pos) else dialog.pos[-1].token.lower()
    lastWord = "" if len(dialog.text)<1 else dialog.text[-1]
    last2Word = lastWord if len(dialog.text)<2 else dialog.text[-2:]
    last3Word = last2Word if len(dialog.text)<3 else dialog.text[-3:]

    feature += [
        # "isLaughted={}".format("<laughter>" in dialog.text.lower()),
        # "isInhale={}".format("<inhaling>" in dialog.text.lower()),
        # "isGasp={}".format("<gasp>" in dialog.text.lower()),
        # "numWords={}".format(numWords),
        # "nextConversationLength={}".format(nextConversationLength),

        "isEndWithHyphen={}".format(dialog.text.endswith("-/") or dialog.text.endswith("- /")),
        "isEndWithSlash={}".format(dialog.text.endswith(" /")),
        lastWord,
        last2Word,
        last3Word
    ]
    # for key in countDict:
    #     feature += ["count_{}={}".format(key,countDict[key])]
    sentenceFeature = []
    # countDict = {"UNDEFINE":0}
    if(dialog.pos):
        for index,posTag in enumerate(dialog.pos):
            if posTag.pos in [",","."]:
                continue
            sentenceFeature = sentenceFeature + [
                #"token[{}]={}".format(index,posTag.token.lower()),
                #"pos[{}]={}".format(index,posTag.pos),
                # "token_pos.{}={}/{}".format(index, posTag.token,posTag.pos),
                posTag.token.lower(),posTag.pos.lower()
                # posTag.token.lower(),
                # posTag.pos[:2] if len(posTag.pos) > 2 else posTag.pos
                # "{}/{}".format( posTag.token.lower(), posTag.pos)
            ]
            # if posTag.pos in countDict:
            #     countDict[posTag.pos] += 1
            # else:
            #     countDict[posTag.pos] = 1
    else:
        sentenceFeature = sentenceFeature + [
            # "token[{}]={}".format(0, dialog.text),
            # "pos[{}]={}".format(0, dialog.text)
            # "token_pos.{}={}/{}".format(0, dialog.text, dialog.text),
            dialog.text.lower(),dialog.text.lower()
            # "{}/{}".format(dialog.text, dialog.text)
        ]

    # feature = feature + [
        #POS with highest frequency
        # sorted(countDict.items(),key=operator.itemgetter(1),reverse=True)[0][0],
    # ]
    return(feature + sentenceFeature)

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

            # numContinueSpeak = 0
            # while (index + numContinueSpeak + 1) < len(dialogSet) and dialogSet[index + numContinueSpeak + 1].speaker == currentSpeaker:
            #     numContinueSpeak += 1
            # nextConversationLength = 0
            # if (index + 1) < len(dialogSet):
            #     if dialogSet[index+1].pos:
            #         nextConversationLength = len(dialogSet[index+1].pos)
            # feature = generateFeatures(dialog, (currentSpeaker == previousSpeaker), index == 0,index,numContinueSpeak,nextConversationLength,index == (len(dialogSet)-1))
            #
            feature = generateFeatures(dialog, (currentSpeaker == previousSpeaker), index == 0)

            previousSpeaker = currentSpeaker

            dialogSetFeature = dialogSetFeature + [feature]
            dialogSetLabel = dialogSetLabel + [actTag]

        dialogCorpusFeature = dialogCorpusFeature + [dialogSetFeature]
        dialogCorpusLabel = dialogCorpusLabel + [dialogSetLabel]
    return(dialogCorpusFeature,dialogCorpusLabel)

if __name__ == "__main__":
    '''
    usage: python3 advanced_crf.py ./tmp/train ./tmp/test ./tmp/result.txt [k_fold_cross_validation]
    '''
    inputFolder = sys.argv[1]
    testFolder = sys.argv[2]
    outputFile = sys.argv[3]

    kFoldCrossValidation = -1
    if len(sys.argv) > 4:
        kFoldCrossValidation = int(sys.argv[4])

    dialogCorpusFeature, dialogCorpusLabel = extractFeaturesAndLabels(inputFolder)

    #if we want to train the whole data and predict the test data as required
    if(kFoldCrossValidation == -1):
        #train CRF
        crfModel = pycrfsuite.Trainer(verbose=False)
        for xSeq,ySeq in zip(dialogCorpusFeature,dialogCorpusLabel):
            crfModel.append(xSeq,ySeq)

        crfModel.set_params({
            'c1': 1.0,  # L1 penalty
            'c2': 1e-3,  # L2 penalty
            'max_iterations': 100
        })

        print("begin training...")
        crfModel.train('crfModel.crfsuite')
        print("finish training...")

        #tag test data
        crfTagger = pycrfsuite.Tagger()
        crfTagger.open('crfModel.crfsuite')

        dialogCorpusFeature, dialogCorpusLabel = extractFeaturesAndLabels(testFolder)
        testFileNames = sorted(glob.glob(os.path.join(testFolder, "*.csv")))

        writeContent = ""
        for index,testFileName in enumerate(testFileNames):
            writeContent += "Filename=" + "\"" + testFileName.split("/")[-1] + "\"" + "\n"
            writeContent += "\n".join(crfTagger.tag(dialogCorpusFeature[index]))
            writeContent += "\n\n"

        writer = open(outputFile, "w", encoding="latin1")
        writer.write(writeContent)
        writer.close()
    else:
        #here, we want to run k-fold cross validation
        foldLabels = list(range(kFoldCrossValidation)) * int(math.ceil((len(dialogCorpusFeature)+0.0) / kFoldCrossValidation))
        foldLabels = foldLabels[0:len(dialogCorpusFeature)]
        shuffle(foldLabels)

        accuracies = []
        for foldLabel in range(kFoldCrossValidation):
            trainDialogCorpusFeature = [corpus for corpus,label in zip(dialogCorpusFeature,foldLabels) if label != foldLabel]
            trainDialogCorpusLabel = [corpusLabel for corpusLabel, label in zip(dialogCorpusLabel, foldLabels) if label != foldLabel]

            testDialogCorpusFeature = [corpus for corpus, label in zip(dialogCorpusFeature, foldLabels) if label == foldLabel]
            testDialogCorpusLabel = [corpusLabel for corpusLabel, label in zip(dialogCorpusLabel, foldLabels) if label == foldLabel]

            # train CRF
            crfModel = pycrfsuite.Trainer(verbose=False)
            for xSeq, ySeq in zip(trainDialogCorpusFeature, trainDialogCorpusLabel):
                crfModel.append(xSeq, ySeq)

            crfModel.set_params({
                'c1': 1.0,  # L1 penalty
                'c2': 1e-3,  # L2 penalty
                'max_iterations': 100
            })

            print("begin training fold:{}...".format(foldLabel))
            crfModel.train('crfModel.cv.crfsuite')
            print("finish training...")

            # tag test data
            crfTagger = pycrfsuite.Tagger()
            crfTagger.open('crfModel.cv.crfsuite')

            predictDialogCorpusTags = [crfTagger.tag(d) for d in testDialogCorpusFeature]
            numPredict = 0
            numCorrect = 0
            for predictDialogCorpusTag,actualDialogCorpusTag in zip(predictDialogCorpusTags,testDialogCorpusLabel):
                for predictTag,actualTag in zip(predictDialogCorpusTag,actualDialogCorpusTag):
                    if predictTag == actualTag:
                        numCorrect += 1
                    numPredict += 1

            accuracy = ((numCorrect+0.0) / numPredict)
            accuracies.append(accuracy)
            #break
        print("Accuracies:{}".format(accuracies))
        print("Average accuracy:{}".format(sum(accuracies)/len(accuracies)))
