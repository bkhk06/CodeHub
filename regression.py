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

def lwlr(testPoint,xArr,yArr,k=1.0):
    xMat =mat(xArr);yMat = mat(yArr).T
    m = shape(xMat)[0]

    weights = mat(eye((m)))
    for j in range(m):
        diffMat = testPoint - xMat[j,:]
        weights[j,j] = exp(diffMat*diffMat.T/(-2.0*k**2))
    xTx = xMat.T*(weights*xMat)

    if linalg.det(xTx) == 0.0:
        print("This matrix is singular,cannot do inverse!")
        return
    ws = xTx.I*(xMat.T*(weights*yMat))
    return testPoint*ws

def lwlrTest(testArr,aArr,yArr,k=1.0):
    m = shape(testArr)[0]
    yHat = zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i],xArr,yArr,k)
    return yHat




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

    #print("yArr[0]: ",yArr[0])
    #print(regression.lwlr(xArr[0],xArr,yArr,1.0))


    # m_xArr = shape(xArr)[0]
    # for i in range(m_xArr):
    #     #print("xArr: ",xArr[i])
    #     yHat[i] = lwlr(xArr[i],xArr,yArr,0.01)
    #     #print("yHat: ",yHat[i])
    # #yHat = mat(yHat)
    # print("yHat: ", shape(mat(yHat)))
    #
    # print("xArr[10] ",xArr[10])
    # #yHat = lwlr(xArr,xArr,yArr,0.003)

    #print(xArr)
    xMat = mat(xArr)
    srtInd = xMat[:,1].argsort(0)
    xSort = xMat[srtInd][:,0,:]
    yHat = lwlrTest(xArr,xArr,yArr,0.003)
    #print(yHat)
    #yHat.sort(0)

    import matplotlib.pyplot as plt2

    fig2 = plt2.figure()
    ax2 = fig2.add_subplot(111)

    #print("srtInd",srtInd,"yHat: ",yHat[srtInd])
    print(shape(xSort[:,1]))
    print(shape(yHat[srtInd]))
    ax2.plot(xSort[:,1],yHat[srtInd])
    ax2.scatter(xMat[:,1].flatten().A[0],mat(yArr).T.flatten().A[0],s=2,c='red')
    plt2.show()

