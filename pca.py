from numpy import *
#
# def loadDataSet(fileName, delim='\t'):
#     fr = open(fileName)
#     stringArr = [line.strip().split(delim) for line in fr.readlines()]
#     datArr = [map(float,line) for line in stringArr]
#     return mat(datArr)


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

if __name__ == "__main__":
    import pca_a
    dataMat = loadDataSet('testSet.txt')
    print(dataMat)
    lowDMat,reconMat = pca(dataMat,1)
    print("\nshape(lowDMat):\n",shape(lowDMat))
    #print("lowDMat:\n",lowDMat,"\nreconMat:\n",reconMat)

