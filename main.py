import load
import gm

date=list(range(2011,2019))

def getV(sub):
    result=[]
    for i in range(1,9):
        val=float(load.getCell(i,sub))
        result.append(val)
    return result

g = gm.Gray_model()

def calu(sub):
    v=getV(sub)
    g.fit(date,v)
    result=g.predict(2021)
    for i in range(11):
        print(result[i])
    print(g.score(v,result[:8],date))

for i in range(1,5):
    print('第',i,'列')
    calu(i)