from numpy import *

def loadDataSet(filename):
    dataMat = []
    fr = open(filename)
    #print(fr)
    for line in fr.readlines():
        curLine = line.split('\t')
        fltLine = map(float,curLine)
        fltLine=list(fltLine)##python 2 and 3 difference
        dataMat.append(fltLine)
    return dataMat


# def loadDataSet(filename):
#     numFeat = len(open(filename).readline().split('\t'))#-1
#     dataMat = [];#labelMat = []
#     fr = open(filename)
#     #print("fr\n: ",fr.readlines())
#     for line in fr.readlines():
#         lineArr = []
#         curLine = line.strip().split('\t')
#         #print("numFeat: ",numFeat)
#         for i in range(numFeat):
#             lineArr.append(float(curLine[i]))
#         fltLine = map(float, lineArr)
#         dataMat.append(fltLine)
#         #print(dataMat)
#         #labelMat.append(float(curLine[-1]))
#     return dataMat#,labelMat

def binSplitDataSet(dataSet,feature,value):
    #dataSet = array(dataSet)
    mat0 = dataSet[nonzero(dataSet[:,feature] > value)[0],:][0]
    mat1 = dataSet[nonzero(dataSet[:,feature] <= value)[0], :][0]
    return mat0,mat1

def regLeaf(dataSet):#returns the value used for each leaf
    return mean(dataSet[:,-1])

def regErr(dataSet):
    return var(dataSet[:,-1]) * shape(dataSet)[0]

def linearSolve(dataSet):   #helper function used in two places
    m,n = shape(dataSet)
    X = mat(ones((m,n))); Y = mat(ones((m,1)))#create a copy of data with 1 in 0th postion
    X[:,1:n] = dataSet[:,0:n-1]; Y = dataSet[:,-1]#and strip out Y
    xTx = X.T*X
    if linalg.det(xTx) == 0.0:
        raise NameError('This matrix is singular, cannot do inverse,\n\
        try increasing the second value of ops')
    ws = xTx.I * (X.T * Y)
    return ws,X,Y

def modelLeaf(dataSet):#create linear model and return coeficients
    ws,X,Y = linearSolve(dataSet)
    return ws

def modelErr(dataSet):
    ws,X,Y = linearSolve(dataSet)
    yHat = X * ws
    return sum(power(Y - yHat,2))

def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):
    tolS = ops[0]; tolN = ops[1]
    #if all the target variables are the same value: quit and return value
    if len(set(dataSet[:,-1].T.tolist()[0])) == 1: #exit cond 1
        return None, leafType(dataSet)
    else:
        m,n = shape(dataSet)
        #print("m= ",m,"n= ",n)
        #the choice of the best feature is driven by Reduction in RSS error from mean
        S = errType(dataSet)
        bestS = inf; bestIndex = 0; bestValue = 0
        for featIndex in range(n-1):
            #print("featIndex: ",featIndex)
            #print("(dataSet[:,featIndex].T.A.tolist())[0]: ",(dataSet[:,featIndex].T.A.tolist())[0])
            for splitVal in set((dataSet[:,featIndex].T.A.tolist())[0]):#set(dataSet[:,featIndex]):
                mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
                #print("\nsplitVal:",splitVal,"mat0: ",mat0,"mat1: ",mat1)
                if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):
                    continue
                newS = errType(mat0) + errType(mat1)
                if newS < bestS:
                    bestIndex = featIndex
                    bestValue = splitVal
                    bestS = newS
        #if the decrease (S-bestS) is less than a threshold don't do the split
        if (S - bestS) < tolS:
            return None, leafType(dataSet) #exit cond 2
        mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
        if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):  #exit cond 3
            return None, leafType(dataSet)
        return bestIndex,bestValue#returns the best feature to split on
                              #and the value used for that split

def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):#assume dataSet is NumPy Mat so we can array filtering
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)#choose the best split
    if feat == None: return val #if the splitting hit a stop condition return val
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree



if __name__ == "__main__":
    import regTrees
    # testMat = mat(eye(4))
    # print(testMat)
    # mat0,mat1 = binSplitDataSet(testMat,1,0.5)
    # print("\nmat0:/n",mat0)
    # print("mat1:/n",mat1)


    myDat = regTrees.loadDataSet('ex0.txt')
    #print(myDat)
    myDat = mat(myDat)
    #print(myDat)
    print "createTree for ex0:\n",regTrees.createTree(myDat)

    myDat = regTrees.loadDataSet('ex00.txt')
    # print(myDat)
    myDat = mat(myDat)
    # print(myDat)
    print "\ncreateTree for ex00:\n",regTrees.createTree(myDat)