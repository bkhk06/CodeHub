def loadDataSet():
    dataSet = [[1,3,4],[2,3,5],[1,2,3,5],[2,5]]
    return dataSet

def createC1(dataSet):
    C1 = []
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                C1.append([item])
    C1.sort()
    return list(map(frozenset,C1))

def scanD(D,Ck,minSupport):
    ssCnt = {}
    for tid in D:
        for can in Ck:
            if can.issubset(tid):
                if can not in ssCnt:
                    ssCnt[can]=1
                else:
                    ssCnt[can]+=1
        numItems = float(len(D))
        retList = []
        supportData = {}
        for key in ssCnt:
            support = ssCnt[key]/numItems
            if support>=minSupport:
                retList.insert(0,key)
            supportData[key] =support
    return retList,supportData

if __name__ == "__main__":
    import apriori
    dataSet = loadDataSet()
    print("dataSet",dataSet)
    C1 = createC1(dataSet)
    print("\nC1:\n",C1)
    D= list(map(set,dataSet))
    print("\nD:\n",D)
    L1,supportData0 = scanD(D,C1,0.5)
    print("\nL1:\n",L1,"\n\nsupportData0:\n",supportData0)




