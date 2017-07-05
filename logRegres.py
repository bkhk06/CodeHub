from numpy import *
import numpy as np
def loadDataSet():
    dataMat = [];labelMat=[]
    fr = open('testSet.txt')
    #print(fr)
    for line in fr.readlines():
        lineArr = line.strip().split()
        #print(lineArr)
        dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat

def sigmoid(inX):
    sigmoid_output=1.0/(1+np.exp(-inX))
    print("sigmoid_output:",sigmoid_output)
    return sigmoid_output

def gradAscent(dataMatIn,classLabels):
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).transpose()
    m,n = shape(dataMatrix)
    #print("m,n:",m,n)
    alpha = 0.001
    maxCycles = 500
    weights = ones((n,1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix*weights)
        error = (labelMat-h)
        # print("w:",shape(weights))
        # print("dataMatrix:",shape(dataMatrix))
        # print("error:",shape(error))
        weights = weights+alpha*dataMatrix.transpose()*error
    return weights

def stocGradAscent0(dataMatrix,classLabel):
    m,n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)
    print("w:",weights)
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i]*weights))
        print(h)
        error = classLabel[i] -h
        error_alpha = error*alpha
        print("error_alpha:",error_alpha)
        print("error:",error)
        print("dataMatrix:",dataMatrix[i])
        print("wegihts:",weights)
        print("Add w+data:",weights+dataMatrix[i])
        print("alpha:",error_alpha*dataMatrix[i])
        weights = weights + alpha*error*dataMatrix[i]
    return  weights

def plotBestFit(weights):
    import matplotlib.pyplot as plt
    dataMat,labelMat=loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i])== 1:
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x, y)
    plt.xlabel('X1'); plt.ylabel('X2');
    plt.show()

if __name__ == "__main__":
    import logRegres
    dataArr,labelMat = logRegres.loadDataSet()
    #weights=logRegres.gradAscent(dataArr,labelMat)
    weights = logRegres.stocGradAscent0(dataArr, labelMat)
    print(weights)
    #print(weights.getA())

    logRegres.plotBestFit(weights)

