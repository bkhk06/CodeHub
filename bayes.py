from numpy import *
import numpy as np
def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]  # 1 is abusive, 0 not
    return postingList, classVec


def createVocabList(dataSet):
    vocabSet = set([])  # create empty set
    for document in dataSet:
        vocabSet = vocabSet | set(document)  # union of the two sets
    return list(vocabSet)


def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)

    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word: %s is not in my Vocabulary!" % word)
    return returnVec

def trainNB0(trainMatrix,trainCategrory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategrory)/float(numTrainDocs)
    p0Num = ones(numWords);p1Num = ones(numWords)
    p0Denom = 2.0; p1Denom = 2.0
    for i in range(numTrainDocs):
        if trainCategrory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom +=sum(trainMatrix[i])

    p1Vect = log(p1Num/p1Denom)
    p0Vect = log(p0Num/p0Denom)

    return p0Vect,p1Vect,pAbusive

def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
    p1 = sum(vec2Classify*p1Vec)+log(pClass1)
    p0 = sum(vec2Classify*p0Vec)+log(1.0-pClass1)

    if p1>p0:
        return 1
    else:
        return 0

def testingNB():
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V, p1V, pAb = trainNB0(array(trainMat), array(listClasses))

    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))

    testEntry = ['stupid', 'cute']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))

def bagOfWords2VecMN(vocabList,inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] +=1
    return returnVec

def textPasrse(bigString):
    import re
    listOfTokens  = re.split(r'\W*',bigString)
    print("\n------------**************----------------\n")
    print([tok.lower() for tok in listOfTokens if len(tok) >2])
    return [tok.lower() for tok in listOfTokens if len(tok) >2]

def spamTest():
    docList = [];classList = [];fullText = []
    for i in range(1,26):
        fr_spam = open('email/spam/%d.txt' % i,encoding="ISO-8859-1").read()
        #print(fr_spam)
        wordList = textPasrse(fr_spam)
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        fr_ham = open('email/ham/%d.txt' % i,encoding="ISO-8859-1").read()
        #print(fr_ham)
        wordList = textPasrse(fr_ham)
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
        vocabList = createVocabList(docList)
    trainingSet = list(range(50));testSet=[]
    for i in range(10):
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = [];trainingClasses = []
    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList,docList[docIndex]))
        trainingClasses.append(classList[docIndex])
    p0V,p1V,pSpam = trainNB0(array(trainMat),array(trainingClasses))
    errorCount = 0
    for docIndex in testSet:  # classify the remaining items
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
            print("classification error", docList[docIndex])
    print ('the error rate is: ', float(errorCount) / len(testSet))
    # return vocabList,fullText
if __name__ == "__main__":
    import bayes

    listOPosts, listClasses = bayes.loadDataSet()
    print(listOPosts)
    print(listClasses)

    myVocabList = bayes.createVocabList(listOPosts)
    print("\nmyVocabList:\n",myVocabList)

    print("\nlistOPosts[0]:",listOPosts[0],"\n",bayes.setOfWords2Vec(myVocabList,listOPosts[0]),"\n")
    print()
    print("\nlistOPosts[3]:",listOPosts[3],"\n",bayes.setOfWords2Vec(myVocabList,listOPosts[3]))

    print("\n---***********-----Train------***********--\n")
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(bayes.setOfWords2Vec(myVocabList,postinDoc))
    import os

    p0V,p1V,pAb = bayes.trainNB0(trainMat,listClasses)

    print("pAb: ",pAb,"\np0V:\n",p0V,"\np1V:\n",p1V)

    bayes.testingNB()

    bayes.spamTest()
