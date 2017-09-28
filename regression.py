from numpy import *

def loadDataSet(filename):
    numFeat = len(open(filename).readline().split('\t'))-1
    dataMat = [];labelMat = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

def standRegres(xArr,yArr):
    xMat = mat(xArr);yMat = mat(yArr).T
    xTx=xMat.T*xMat
    if linalg.det(xTx) == 0.0:
        print("This matrix is sigular,canot do inverse")
        return
    ws = xTx.I*(xMat.T*yMat)
    return ws

if __name__ == "__main__":
    import regression
    from numpy import *
    xArr,yArr = regression.loadDataSet('ex0.txt')
    print(xArr[0:2])

    ws = regression.standRegres(xArr,yArr)
    print(ws)

    xMat = mat(xArr)
    yMat = mat(yArr)
    yHat = xMat*ws

    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xMat[:,1].flatten().A[0],yMat.T[:,0].flatten().A[0])

    xCopy = xMat.copy()
    #print(xCopy)
    xCopy.sort(0)
    #print(xCopy[:,1])
    yCopy = yHat.copy()
    yCopy.sort(0)
    #print("yHat:",yHat)
    ax.plot(xCopy[:,1],yCopy)
    plt.show()
