import sys
import os
import hw3_corpus_tool as taTool

if __name__ == "__main__":
    '''
    usage: python3 evaluate_model.py ./tmp/test ./tmp/result.txt
    '''
    testFolder = sys.argv[1]
    outputFile = sys.argv[2]

    filePath = ""
    predictTags = []
    predictList = []
    with open(outputFile,'r') as inputStream:
        for line in inputStream:
            if line.strip()[-4:] == ".csv":
                filePath = os.path.join(testFolder, line.strip())
                predictTags = []
            elif len(line.strip()) > 0:
                predictTags += [line.strip()]
            else:
                predictList += [(filePath,predictTags)]

    numPredict = 0
    numCorrect = 0

    for filePath,predictTags in predictList:
        dialogUtterances = taTool.get_utterances_from_filename(filePath)
        actualTags = [d.act_tag for d in dialogUtterances]
        for predictTag,actualTag in zip(predictTags,actualTags):
            if(predictTag == actualTag):
                numCorrect += 1
            numPredict += 1

    accuracy = ((numCorrect+0.0) / numPredict)
    print("accuracy:{}".format(accuracy))
