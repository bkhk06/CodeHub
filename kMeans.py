# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 09:35:29 2017

@author: Liu-Da
"""

from numpy import *

def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = map(float,curLine)
        dataMat.append(list(fltLine))
    return dataMat

def distEclud(vecA,vecB):
    return sqrt(sum(power(vecA-vecB,2)))

def randCent(dataSet,k):
    n = shape(dataSet)[1]
    centroids = mat(zeros((k,n)))
    for j in range(n):
        minJ = min(dataSet[:,j])
        rangeJ = float(max(dataSet[:,j])-minJ)
        centroids[:,j] = minJ+rangeJ*random.rand(k,1)
    return centroids


if __name__ == "__main__":
    import kMeans
    dataMat = mat(kMeans.loadDataSet('testSet.txt'))
    #print("\ndataMat:\n", dataMat)
    #print("\n(dataMat[,:0]):\n", dataMat[:, 0])
    print("\nmin(dataMat[,:0]):",min(dataMat[:,0]))
    #print("\n(dataMat[,:1]):\n", dataMat[:, 1])
    print("\nmin(dataMat[,:1]):",min(dataMat[:,1]))

    print("\nrandCent(dataMat,2):\n",randCent(dataMat,2))
    print("\ndistEclud(dataMat[0],dataMat[1]:\n",distEclud(dataMat[0],dataMat[1]))

