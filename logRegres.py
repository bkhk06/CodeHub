# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 08:32:05 2017
@author: Liu-Da
"""

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
    sigmoid_output=1.0/(1+exp(-inX))
    #print("sigmoid_output:",sigmoid_output)
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
    weights = ones(n) # it is a array,w: [ 1.  1.  1.], not a list, list is like [1,2,3,4]

    for i in range(m):
        h = sigmoid(sum(dataMatrix[i]*weights))
        error = classLabel[i] -h
        weights = weights + alpha*error*array(dataMatrix[i])
        #it needs to convert list into array,cause you can multiply a list with an integer, but not a float
        #>>> [1] * 4
        #[1, 1, 1, 1]
        #but not by a float:
        #[1] * 4.0,  TypeError: can't multiply sequence by non-int of type 'float'
    return  weights

def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    m,n = shape(dataMatrix)
    weights = ones(n)   #initialize to all ones
    for j in range(numIter):
        dataIndex = list(range(m))# convert range type into list
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.0001    #apha decreases with iteration, does not
            randIndex = int(random.uniform(0,len(dataIndex)))#go to 0 because of the constant
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            #print("type of dataIndex[randIndex]:",dataIndex[randIndex])
            del(dataIndex[randIndex])
    return weights


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



    weights = logRegres.gradAscent(dataArr, labelMat)
    print("\nthe computed weight of gradAscent:\n",weights)
    print("type of weights:",type(weights))
    print("type of weights.getA:", type(weights.getA))#convert self from matrix to  ndarray
    logRegres.plotBestFit(weights.getA())

    weights = logRegres.stocGradAscent0(dataArr, labelMat)
    print("\ntype of weights from stocGradAscent0:", type(weights))
    print("the computed weight of stocGradAscent0:\n",weights)
    logRegres.plotBestFit(weights)

    weights = logRegres.stocGradAscent1(array(dataArr),labelMat)
    print("\ntype of weights from stocGradAscent1:", type(weights))
    print("the computed weight of stocGradAscent1:\n", weights)
    logRegres.plotBestFit(weights)
