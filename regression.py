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
    xMat =mat(xArr);yMat = mat(yArr).T;#testPoint = mat(testPoint0)
    m = shape(xMat)[0]

    weights = mat(eye((m)))
    for j in range(m):
        #print(j," shape of testPoint: ",shape(testPoint),"testPoint",testPoint)
        #print("shape of xMat_j: ",shape(xMat[j,:]),"xMat_j: ",xMat[j,:])
        diffMat = testPoint - xMat[j,:]
        #print("diffMat: ",diffMat)
        weights[j,j] = exp(diffMat*diffMat.T/(-2.0*k**2))
    xTx = xMat.T*(weights*xMat)

    if linalg.det(xTx) == 0.0:
        print("This matrix is singular,cannot do inverse!")
        return
    ws = xTx.I*(xMat.T*(weights*yMat))
    return testPoint*ws

def lwlrTest(testArr,xArr,yArr,k=1.0):
    m = shape(testArr)[0]
    yHat = zeros(m)
    for i in range(m):
        #print(i," lwlrtest")
        yHat[i] = lwlr(testArr[i],xArr,yArr,k)
    return yHat

def rssError(yArr,yHatArr):
    return ((yArr-yHatArr)**2).sum()

def ridgeregres(xMat,yMat,lam=0.2):
    xTx = xMat.T*xMat
    demon = xTx + eye(shape(xMat)[1])*lam
    if linalg.det(demon) == 0.0:
        print("This matrix is singular,can't do inverse!")
        return
    ws = demon.I*(xMat.T*yMat)
    return ws

def ridgeTest(xArr,yArr):
    xMat = mat(xArr);yMat = mat(yArr).T
    yMean = mean(yMat,0)
    yMat = yMat - yMean
    xMeans = mean(xMat,0)
    xVar = var(xMat,0)
    xMat = (xMat-xMeans)/xVar
    numTestPts = 30
    wMat = zeros((numTestPts,shape(xMat)[1]))
    for i in range(numTestPts):
        ws = ridgeregres(xMat,yMat,exp(i-10))
        wMat[i,:] = ws.T
    return wMat

def regularize(xMat):  # regularize by columns
    inMat = xMat.copy()
    inMeans = mean(inMat, 0)  # calc mean then subtract it off
    inVar = var(inMat, 0)  # calc variance of Xi then divide by it
    inMat = (inMat - inMeans) / inVar
    return inMat

def stageWise(xArr, yArr, eps=0.01, numIt=100):
    xMat = mat(xArr);
    yMat = mat(yArr).T
    yMean = mean(yMat, 0)
    yMat = yMat - yMean  # can also regularize ys but will get smaller coef
    xMat = regularize(xMat)
    m, n = shape(xMat)
    returnMat = zeros((numIt, n))  # testing code remove
    ws = zeros((n, 1));
    wsTest = ws.copy();
    wsMax = ws.copy()
    for i in range(numIt):
          print
          ws.T
          lowestError = inf;
          for j in range(n):
              for sign in [-1, 1]:
                  wsTest = ws.copy()
                  wsTest[j] += eps * sign
                  yTest = xMat * wsTest
                  rssE = rssError(yMat.A, yTest.A)
                  if rssE < lowestError:
                      lowestError = rssE
                      wsMax = wsTest
          ws = wsMax.copy()
          returnMat[i, :] = ws.T
    return returnMat


if __name__ == "__main__":
    import regression
    # from numpy import *
    # xArr,yArr = regression.loadDataSet('ex0.txt')
    # print(xArr[0:2])
    #
    # ws = regression.standRegres(xArr,yArr)
    # print(ws)
    #
    # xMat = mat(xArr)
    # yMat = mat(yArr)
    # yHat = xMat*ws
    #
    # import matplotlib.pyplot as plt
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.scatter(xMat[:,1].flatten().A[0],yMat.T[:,0].flatten().A[0])
    #
    # xCopy = xMat.copy()
    # #print(xCopy)
    # xCopy.sort(0)
    # #print(xCopy[:,1])
    # yCopy = yHat.copy()
    # yCopy.sort(0)
    # #print("yHat:",yHat)
    # ax.plot(xCopy[:,1],yCopy)
    # plt.show()
    #
    #
    #
    # #
    # #print("yArr[0]: ",yArr[0])
    # #print(regression.lwlr(xArr[0],xArr,yArr,1.0))
    #
    # ##print("xArr[10] ",xArr[10])
    # #yHat = lwlr(xArr,xArr,yArr,0.003)
    #
    # #print(xArr)
    # xMat = mat(xArr)
    # #print(xMat[:,0:2])
    # srtInd = xMat[:,1].argsort(0)
    # xSort = xMat[srtInd][:,0,:]
    # yHat = lwlrTest(xArr,xArr,yArr,0.003)
    # #print(yHat)
    # #yHat.sort(0)#
    #
    # import matplotlib.pyplot as plt2
    #
    # fig2 = plt2.figure()
    # ax2 = fig2.add_subplot(111)
    #
    # #print("srtInd",srtInd,"yHat: ",yHat[srtInd])
    # print(shape(xSort[:,1]))
    # print(shape(yHat[srtInd]))
    # ax2.plot(xSort[:,1],yHat[srtInd])
    # #ax2.plot(xSort[:,1],yHat)
    # ax2.scatter(xMat[:,1].flatten().A[0],mat(yArr).T.flatten().A[0],s=2,c='red')
    # plt2.show()

    #
    # #predict the abalone's age， that need to comment above codes firstly
    # abX, abY = loadDataSet('abalone.txt')
    #
    # yHat01 = lwlrTest(mat(abX[0:99]),abX[0:99],abY[0:99],0.1)
    # yHat1 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 1)
    # yHat10 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 10)
    #
    # print("rssError(abY[0:99],yHat01.T: ",rssError(abY[0:99],yHat01.T))
    # print("rssError(abY[0:99],yHat1.T: ", rssError(abY[0:99], yHat1.T))
    # print("rssError(abY[0:99],yHat10.T: ", rssError(abY[0:99], yHat10.T))
    # print("######################################")
    # yHat01 = lwlrTest(mat(abX[100:199]), abX[0:99], abY[0:99], 0.1)
    # yHat1 = lwlrTest(abX[100:199], abX[0:99], abY[0:99], 1)
    # yHat10 = lwlrTest(abX[100:199], abX[0:99], abY[0:99], 10)
    #
    # print("rssError(abY[0:99],yHat01.T: ", rssError(abY[100:199], yHat01.T))
    # print("rssError(abY[0:99],yHat1.T: ", rssError(abY[100:199], yHat1.T))
    # print("rssError(abY[0:99],yHat10.T: ", rssError(abY[100:199], yHat10.T))
    #
    # print("##################################$$$$$$")
    # ws = standRegres(abX[0:99],abY[0:99])
    # yHat = mat(abX[100:199])*ws
    # print("rssError(abY[100:199],yHat.T.A: ",rssError(abY[100:199],yHat.T.A))

    # # predict the abalone's age with ridgeRegres， that need to comment above codes firstly
    # abX,abY = loadDataSet('abalone.txt')
    # ridgeWeights =  ridgeTest(abX,abY)
    # print("ridgeWeights: \n",ridgeWeights)
    # print("shape of ridgeWeights ",shape(ridgeWeights))
    # import matplotlib.pyplot as plt3
    # fig3 = plt3.figure()
    # ax3 = fig3.add_subplot(111)
    # ax3.plot(ridgeWeights)
    # plt3.show()

    # #StageWise
    # xArr,yArr = loadDataSet('abalone.txt')
    # print(stageWise(xArr,yArr,0.01,200))
    # print("###################")
    # print(stageWise(xArr,yArr,0.001,5000))
    # print("$$$$$$$$$$$$$$$$$$$")
    # xMat = mat(xArr);yMat = mat(yArr).T
    # xMat = regularize(xMat)
    # yM = mean(yMat,0)
    # yMat = yMat - yM
    # weights = standRegres(xMat,yMat.T)
    # print(weights.T)




