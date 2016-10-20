import sys

#usage: python3 evaluate.py predict_result.txt
if __name__ == "__main__":

    filename = sys.argv[1]

    predictResults = []

    with open(filename) as inputstream:
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

    truePositive =  sum([1 for x in predictResults if x[0] == x[1] == "ham"])
    trueNegative =  sum([1 for x in predictResults if x[0] == x[1] == "spam"])
    falsePositive =  sum([1 for x in predictResults if x[0] == "ham" and x[1] == "spam"])
    falseNegative =  sum([1 for x in predictResults if x[0] == "spam" and x[1] == "ham"])

    precisionHam = truePositive / (truePositive + falsePositive)
    recallHam = truePositive / (truePositive + falseNegative)
    fscoreHam = 2 * precisionHam * recallHam / (precisionHam + recallHam)
    print("[HAM]Precision:" + str(precisionHam) + ",Recall:" + str(recallHam) + ",FScore:" + str(fscoreHam))

    precisionSpam = trueNegative / (trueNegative + falseNegative)
    recallSpam = trueNegative / (trueNegative + falsePositive)
    fscoreSpam = 2 * precisionSpam * recallSpam / (precisionSpam + recallSpam)
    print("[SPAM]Precision:" + str(precisionSpam) + ",Recall:" + str(recallSpam) + ",FScore:" + str(fscoreSpam))
