import pandas as pd
import numpy as np
import talib as tb
dbg ={}
class IND:
    req = 'c'
    def __init__(self,**args):
        pass
    def __call__(self,data,**args):
        raise Exception('NIN')

class SMA(IND):
    def __init__(self,N=10,**args):
        IND.__init__(self)
        self.N = int(N)
    def __call__(self,data,**args):
        return data.rolling(window = self.N, center = False).mean()

class WMA(IND):
    def __init__(self,N,**args):
        IND.__init__(self)
        self.N = int(N)
        self.w = np.array(range(1,self.N+1))/sum(range(self.N+1))
    def __call__(self,data,**args):
        return data.rolling(window = self.N,center = False).apply(lambda x:np.dot(x,self.w))

class HMA(IND):
    def __init__(self,N,**args):
        IND.__init__(self)
        self.N = int(N)
        self.WMAHN = WMA(int(self.N/2))
        self.WMAN = WMA(self.N)
        self.WMASN = WMA(int(np.sqrt(self.N)))
    def __call__(self,data,**args):
        return self.WMASN(2*self.WMAHN(data)-self.WMAN(data))

class EMA(IND):
    def __init__(self,alpha,**args):
        IND.__init__(self)
        self.alpha = float(alpha)
    def __call__(self,data,**args):
        return data.ewm(alpha = self.alpha).mean()

class KAMA(IND):
    def __init__(self,N=30,F = 5,S = 30,**args):
        IND.__init__(self)
        self.N = int(N)
        self.F = int(F)
        self.S = int(S)
    def __call__(self,data,**args):
        if type(data) is pd.DataFrame:
            data = data[data.columns[0]]
        U = data.rolling(window = self.N,center = False).apply(lambda x:abs(x[0] - x[-1]))
        D = data.rolling(window = self.N,center = False).apply(lambda x:sum([abs(x[i]-x[i-1]) for i in range(1,self.N)]))
        E = U/D
        sc = E * (self.F/(self.F+1) - self.S/(self.S+1)) + self.F/(self.S+1)
        c = sc*sc
        AMA = data*0
        AMA.iloc[0] = data.iloc[0]
        for i in range(1,len(AMA)):
            AMA.iloc[i] = AMA.iloc[i-1] + c.iloc[i] * (data.iloc[i]-AMA.iloc[i-1])
            dbg['dbgx'] = AMA
            if np.isnan(AMA.iloc[i]):
                AMA.iloc[i] = data.iloc[i]
        return AMA

corr = lambda x,y:np.corrcoef(x,y)[0,1]

class SIMD(IND):
    def __init__(self,w = None,**args):
        IND.__init__(self)
        self.w = w
        self.N = len(w)
    def __call__(self,data,**args):
        return data.rolling(window = self.N,center = False).apply(lambda x:corr(x,self.w))

class LINSIM(SIMD):
    def __init__(self,N,**args):
        SIMD.__init__(self,range(1,int(N)+1))

class EXPSIM(SIMD):
    def __init__(self,N,tau = 1,draft = 0,**args):
        SIMD.__init__(self,np.array( [np.exp(tau*x/int(N)+draft) for x in range(int(N))]))

class AMA(IND):
    def __init__(self,N = 60,p = 3.3,F = 5,**args):
        IND.__init__(self)
        self.N = int(N)
        self.p = p
        self.F = int(F)
    def __call__(self,data,**args):
        if type(data) is pd.DataFrame:
            data = data[data.columns[0]]
        S = data.rolling(window = self.N,center = False).std()
        M = data.rolling(window = self.N,center = False).mean()
        D = S/M
        s = ((self.N - self.F) * self.p * D+self.F).fillna(0).apply(lambda x:int(x))

        res = data * 1
        for i in range(len(data)):
            res.iloc[i] = data.iloc[max(i-s.iloc[i],0):i].mean()
        return res

class OBV(IND):
    req = 'cv'
    def __init__(self,**args):
        IND.__init__(self,**args)
    def __call__(self,data,**args):
        c = data[data.columns[0]]
        v = data[data.columns[1]]
        return tb.OBV(c,v)

class AD(IND):
    req = 'chlv'
    def __init__(self,**args):
        IND.__init__(self,**args)
    def __call__(self,data,**args):
        c,h,l,v = data 
        c = data[c]
        h = data[h]
        l = data[l]
        v = data[v]
        return tb.AD(h,l,c,v)

class BOLL(IND):
    def __init__(self,N=40,a=1.7,**args):
        IND.__init__(self,**args)
        self.N = N
        self.a = a 
    def __call__(self,data,**args):
        if type(data) is pd.DataFrame:
            data = data[data.columns[0]]
        res = pd.DataFrame()
        m = data.rolling(window = self.N,center = False).mean()
        s = data.rolling(window = self.N,center = False).std()
        res['u'] = m+s*self.a
        res['d'] = m-s*self.a
        return res


def SMOOTH(ind,method = SMA,Sarg = {'N' : 5,'bin' : False},**args):
    class SMO(ind):
        def __init__(self,**args):
            ind.__init__(self,**args)
        def __call__(self,**args):
            argsZ = {x:args[x] for x in args.keys() if x!= 'data'}
            res = method(**Sarg)(data = ind.__call__(self,**args),**argsZ).fillna(0)
            if Sarg['bin']:
                res = res.apply(lambda x:round(x))
            return res
    return SMO



