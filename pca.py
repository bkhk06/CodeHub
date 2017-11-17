from numpy import *

def loadDataSet(fileName, delim='\t'):
    fr = open(fileName)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    datArr = [list(map(float,line)) for line in stringArr]#thansfer to list type
    return mat(datArr)


# def loadDataSet(filename):
#     dataMat = []
#     fr = open(filename)
#     #print(fr)
#     for line in fr.readlines():
#         curLine = line.split('\t')
#         fltLine = map(float,curLine)
#         fltLine=list(fltLine)##python 2 and 3 difference
#         dataMat.append(fltLine)
#     return dataMat

def pca(dataMat,topNfeat = 9999999):
    meanVals = mean(dataMat,axis=0)
    meanRemoved = dataMat - meanVals
    covMat = cov(meanRemoved, rowvar=0)
    eigVals, eigVects = linalg.eig(mat(covMat))
    print("eigVals:",eigVals, "\neigVects:",eigVects)
    eigInd = argsort(eigVals)
    eigInd = eigInd[:-(topNfeat+1):-1]
    redEigVects = eigVects[:,eigInd]
    lowDDataMat = meanRemoved * redEigVects
    reconMat = (lowDDataMat*redEigVects.T) + meanVals
    return lowDDataMat,reconMat

def replaceNanWithMean():
    dataMat = loadDataSet('secom.data',' ')
    numFeat = shape(dataMat)[1]
    for i in range(numFeat):
        meanVal = mean(dataMat[nonzero(-isnan(dataMat[:,i].A))[0],i])
        dataMat[nonzero(isnan(dataMat[:,i].A))[0],i] = meanVal
    return dataMat

if __name__ == "__main__":
    import pca_a
    dataMat = loadDataSet('testSet.txt')
    print(dataMat)
    lowDMat,reconMat = pca(dataMat,1)
    print("\nshape(lowDMat):\n",shape(lowDMat))
    #print("lowDMat:\n",lowDMat,"\nreconMat:\n",reconMat)

    import matplotlib
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dataMat[:, 0].flatten().A[0], dataMat[:, 1].flatten().A[0], marker='^', s=90)
    ax.scatter(reconMat[:, 0].flatten().A[0], reconMat[:, 1].flatten().A[0], marker='o', s=50, c='red')
    plt.show()

    ##########################################################################
    print("#######################################################")
    dataMat = loadDataSet('testSet3.txt')
    print(dataMat)
    lowDMat, reconMat = pca(dataMat, 3)
    print("\nshape(lowDMat):\n", shape(lowDMat))
    # print("lowDMat:\n",lowDMat,"\nreconMat:\n",reconMat)

    import matplotlib
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dataMat[:, 0].flatten().A[0], dataMat[:, 1].flatten().A[0], marker='^', s=90)
    ax.scatter(reconMat[:, 0].flatten().A[0], reconMat[:, 1].flatten().A[0], marker='o', s=50, c='red')
    plt.show()
    ####################################################################

    dataMat = pca_a.replaceNanWithMean()
    meanVals = mean(dataMat,axis=0)
    meanRemoved = dataMat - meanVals
    covMat = cov(meanRemoved,rowvar=0)

    eigVals,eigVects = linalg.eig(mat(covMat))

    print(eigVals)
