#usage: evaluate.py

if __name__ == "__main__":

    predictResults = []

    with open("nboutput.txt") as inputstream:
        for line in inputstream:
            whiteSpaceIndex = line.find(" ")
            predictLabel = line[0:whiteSpaceIndex]
            filePath = line[whiteSpaceIndex+1:]
            fileName = filePath[filePath.rfind("/")+1:]
            #all test files contain either "ham" or "spam" in their name
            if "ham" in fileName:
                actualLabel = "ham"
            else:
                actualLabel = "spam"

            predictResults = predictResults + [(predictLabel,actualLabel)]

    #construct confusion matrix
    #   **********      Predict HAM |   Predict SPAM
    #   Actual HAM          [0][0]          [0][1]
    #   Actual SPAM         [1][0]          [1][1]
    #-------------------------

    confusionMatrix = [[0,0],[0,0]]
    for result in predictResults:
        if(result[0] == result[1] == "ham"):
            confusionMatrix[0][0] = confusionMatrix[0][0] + 1
        elif(result[0] == result[1] == "spam"):
            confusionMatrix[1][1] = confusionMatrix[1][1] + 1
        elif(result[0] == "ham" and result[1] == "spam"):
            confusionMatrix[1][0] = confusionMatrix[1][0] + 1
        else:
            confusionMatrix[0][1] = confusionMatrix[0][1] + 1

    print(confusionMatrix)

    precisionHam = confusionMatrix[0][0] / (confusionMatrix[0][0] + confusionMatrix[1][0])
    recallHam = confusionMatrix[0][0] / (confusionMatrix[0][0] + confusionMatrix[0][1])
    fscoreHam = 2 * precisionHam * recallHam / (precisionHam + recallHam)
    print("[HAM]Precision:{0},Recall:{1},FScore:{2}".format(precisionHam,recallHam,fscoreHam))

    precisionSpam = confusionMatrix[1][1] / (confusionMatrix[1][1] + confusionMatrix[0][1])
    recallSpam = confusionMatrix[1][1] / (confusionMatrix[1][1] + confusionMatrix[1][0])
    fscoreSpam = 2 * precisionSpam * recallSpam / (precisionSpam + recallSpam)
    print("[SPAM]Precision:{0},Recall:{1},FScore:{2}".format(precisionSpam,recallSpam,fscoreSpam))
