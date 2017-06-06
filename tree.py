from math import log
import operator

def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1,'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [1, 1, 'no']]
    dataSet1 = [[1, 1, 0, 'yes'],
               [1, 1, 1, 'yes'],
               [1, 0, 1, 'no'],
               [0, 1, 0, 'no'],
               [1, 1, 1, 'no']]
    labels = ['no surfacing','flippers']
    #change to discrete values
    return dataSet, labels

def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel]=0
        labelCounts[currentLabel] +=1
    shannonEnt=0.0

    for key in labelCounts:
        prob=float(labelCounts[key])/numEntries
        shannonEnt -=prob * log(prob,2)
    return shannonEnt

def splitDataSet(dataSet,axis,value):
    retDataSet=[]#temp dataset
    for featVec in dataSet:
        if featVec[axis]==value:
            reducedFeatVec=featVec[:axis]#generate pre-half
            #print(reducedFeatVec)
            reducedFeatVec.extend(featVec[axis+1:])#generate post-half sub-set
            #print(reducedFeatVec)
            retDataSet.append(reducedFeatVec)#combine subset as whole set
            #print(retDataSet)
    return retDataSet

def chooseBestFeatureToSplit(dataset):
    numFeatures=len(dataset[0])-1#except the last one as label, the others are features
    baseEntropy=calcShannonEnt(dataset)
    bestInfoGain=0.0;bestFeature=-1
    for i in range(numFeatures):
        featList=[example[i] for example in dataset]
        uniqueVals=set(featList)
        #print(uniqueVals)
        newEntropy=0.0
        for value in uniqueVals:
            subDataSet=splitDataSet(dataset,i,value)
            prob=len(subDataSet)/float(len(dataset))
            #print(value,prob)
            newEntropy +=prob*calcShannonEnt(subDataSet)
        infoGain=baseEntropy-newEntropy
        if(infoGain>bestInfoGain):
            bestInfoGain=infoGain
            bestFeature=i
    return bestFeature

def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys():classCount[vote]=0
        classCount[vote] +=1
    sortedClassCount=sorted(classCount.items(),
                            key=operator.itemgetter(1),reverse=True)
    return sortedClassCount [0][0]

def createTree(dataSet,labels):
    classList=[example[-1] for example in dataSet]
    if classList.count(classList[0])==len(classList):
        return classList[0]#stop the splitting when class are same
    if len(dataSet[0])==1:
        return majorityCnt(classList)
    bestFeat=chooseBestFeatureToSplit(dataSet)
    bestFeatLabel=labels[bestFeat]
    myTree={bestFeatLabel:{}}
    del (labels[bestFeat])
    featValues=[example[bestFeat] for example in dataSet]
    uniqueVals=set(featValues)
    for value in uniqueVals:
        subLabels=labels[:]
        myTree[bestFeatLabel][value]=createTree(splitDataSet\
                                                    (dataSet,bestFeat,value),subLabels)
    return myTree


if __name__ == "__main__":
    import trees
    myDat,labels=trees.createDataSet()
    print(myDat,'\n',labels)
    print("The shannonEnt is :",trees.calcShannonEnt(myDat))

    print("\nsplitDataSet(myDat,1,1):\n",trees.splitDataSet(myDat,1,1))
    #print("\nsplitDataSet(myDat,0,0):\n",trees.splitDataSet(myDat,0,0))

    print("\nchooseBestFeatureToSplit: ",trees.chooseBestFeatureToSplit(myDat))

    myTree=createTree(myDat,labels)
    print("\nmyTree:\n",myTree)
