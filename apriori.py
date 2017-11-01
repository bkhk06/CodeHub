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

def aprioriGen(Lk,k):#create CK
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i+1,lenLk):
            L1 = list(Lk[i])[:k-2];L2 = list(Lk[j])[:k-2]
            L1.sort();L2.sort()
            if L1==L2:
                retList.append(Lk[i]|Lk[j])
    return retList

def apriori(dataSet,minSupport = 0.5):
    C1 = createC1(dataSet)
    D = list(map(set,dataSet))
    L1,supportData = scanD(D,C1,minSupport)
    L = [L1]
    k = 2
    while(len(L[k-2])>0):
        Ck = aprioriGen(L[k-2],k)
        Lk,supK = scanD(D,Ck,minSupport)
        supportData.update(supK)
        L.append(Lk)
        k +=1
    return L,supportData

def generalRules(L,supportData,minConf=0.7):
    bigRuleList = []
    for i in range(1,len(L)):
        for freqSet in L[i]:
            H1 = [frozenset([item]) for item in freqSet]
            if (i > 1):
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else:
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)
    return bigRuleList

def calcConf(freqSet, H, supportData, brl, minConf=0.7):
    prunedH = []  # create new list to return
    for conseq in H:
        conf = supportData[freqSet] / supportData[freqSet - conseq]  # calc confidence
        if conf >= minConf:
            print(freqSet - conseq, '-->', conseq, 'conf:', conf)
            brl.append((freqSet - conseq, conseq, conf))
            prunedH.append(conseq)
    return prunedH


def rulesFromConseq(freqSet, H, supportData, brl, minConf=0.7):
    m = len(H[0])
    if (len(freqSet) > (m + 1)):  # try further merging
        Hmp1 = aprioriGen(H, m + 1)  # create Hm+1 new candidates
        Hmp1 = calcConf(freqSet, Hmp1, supportData, brl, minConf)
        if (len(Hmp1) > 1):  # need at least two sets to merge
            rulesFromConseq(freqSet, Hmp1, supportData, brl, minConf)

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

    L,supportData = apriori.apriori(dataSet)
    print("\nL:\n",L)
    print("\nL[0]:\n", L[0])
    print("\nL[1]:\n", L[1])
    print("\nL[2]:\n", L[2])
    print("\naprioriGen(L[0],2):\n",aprioriGen(L[0],2))
    L,supportData = apriori.apriori(dataSet,minSupport=0.7)
    print("\nL of apriori.apriori(dataSet,minSupport=0.7:\n",L)

    print("\n#########generalRules(L, suppData, minConf=0.7)######\n")
    L,suppData = apriori.apriori(dataSet,minSupport = 0.5)
    rules = apriori.generalRules(L,suppData,minConf=0.7)
    print("\nrules:\n",rules)

    print("\n#########generalRules(L, suppData, minConf=0.5)######\n")
    L, suppData = apriori.apriori(dataSet, minSupport=0.5)
    rules = apriori.generalRules(L, suppData, minConf=0.5)
    print("\nrules:\n", rules)

    # print("\n#########votesmart######\n")
    # from votesmart import votesmart
    # votesmart.apikey = '49024thereoncewasamanfromnantucket94040'
    # bills = votesmart.votes.getBillsByStateRecent()
    # for bill in bills:
    #     print(bill.title,bill.billId)

    print("\nMush######\n")
    mushDatSet = [line.split() for line in open('mushroom.dat').readlines()]
    L,suppData = apriori.apriori(mushDatSet,minSupport=0.3)
    print(L[3])
    for item in L[3]:
         if item.intersection('2'): print(item)

