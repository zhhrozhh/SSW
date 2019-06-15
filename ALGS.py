
import pandas as pd
import numpy as np
from importlib import reload
import IND
import talib

GIMP = {
    'IND':IND
}


def dbg_udt():
    GIMP['IND'] = reload(GIMP['IND'])
    try:
        GIMP['IND'].dbg_udt()
    except:
        pass 


class ALG:
    def __init__(self,**args):
        try:
            self.C = args['C']
        except:
            self.C = None
        try:
            self.O = args['O']
        except:
            self.O = None
        try:
            self.H = args['H']
        except:
            self.H = None
        try:
            self.L = args['L']
        except:
            self.L = None
        try:
            self.V = args['V']
        except:
            self.V = None
    def __call__(self,**args):
        raise Exception()
    def ret(self,bch = None,drift = 2,**args):
        if not bch:
            bch = self.C

        rts = ((bch - bch.shift(1))/bch.shift(1)).fillna(0)
        dcs = self.__call__(**args).shift(drift).fillna(0)
        res = rts*0
        res.iloc[0] = 1
        for i in range(1,len(rts)):
            res.iloc[i] = 1+dcs.iloc[i]*rts[i]
        return pd.DataFrame({'bch':rts+1,'alg':res})

    def am_ret(self,bch = None,drift = 2,**args):
        d = self.ret(bch,drift,**args)
        for x in d.columns:
            d[x] = d[x].cumsum()/(pd.Series(range(1,len(d)+1),index = d.index))
        return d

    def gm_ret(self,bch = None,drift = 2,**args):
        d = self.ret(bch,drift,**args)
        for x in d.columns:
            d[x] = d[x].cumprod()**(1/pd.Series(range(1,len(d)+1),index = d.index))
        return d
    def c_ret(self,bch = None,drift = 2,**args):
        return self.ret(bch,drift,**args).cumprod()





class UD_ALG(ALG):
    def __init__(self,**args):
        ALG.__init__(self,**args)
        try:
            self.idc = args['idc']
        except Exception as e:
            self.idc = None

    def __call__(self,bias = 0,**args):
        b1 = self.idc > (self.idc.shift(1)+bias)
        s1 = self.idc < (self.idc.shift(1)+bias)
        bp = b1 & ~b1.shift(1).fillna(True)
        sp = s1 & ~s1.shift(1).fillna(True)
        res = self.idc*0
        st = 0
        for i in range(len(res)):
            if bp.iloc[i]:
                st = 1
            if sp.iloc[i]:
                st = 0
            res.iloc[i] = st
        return res

class CROS_ALG(UD_ALG):
    def __init__(self,**args):
        try:
            self.F = args['F']
        except:
            self.F = None
        try:
            self.S = args['S']
        except:
            self.S = None
        D = self.F-self.S
        G = D.cumsum()
        UD_ALG.__init__(self,idc = G,**args)

class T_CROS_ALG(UD_ALG):
    def __init__(self,**args):
        try:
            self.T = args['T']
        except:
            self.T = None
        try:
            self.S = args['S']
        except:
            self.S = None
        try:
            self.F = args['F']
        except:
            self.F = None

        TT = (self.T<self.S)&(self.T<self.F)
        D = self.F-self.S
        G = D.cumsum()
        UD_ALG.__init__(self,idc = G*TT,**args)

class VOT_ALG(ALG):
    def __init__(self,*voters,**args):
        ALG.__init__(self,**args)
        self.voters = voters
        self.X = pd.DataFrame([v() for v in self.voters])
    def __call__(self,weights = None,**args):
        if not weights:
            weights = np.ones(len(self.voters))
        return self.X.apply((lambda x:np.dot(weights,x)/sum(weights)))

class HW_VOTER(VOT_ALG):
    def __init__(self,*voters,**args):
        VOT_ALG.__init__(self,*voters,**args)
        self.Y = pd.DataFrame([v.am_ret()['alg'] for v in self.voters]).T.rank(axis = 1,method = 'first') == len(self.voters)

    def __call__(self,**args):
        return pd.Series([np.dot(self.X.T.iloc[i],self.Y.iloc[i])/self.Y.iloc[i].sum() for i in range(len(self.X.T.index))] ,index = self.X.T.index)

class HW_HMA_VOTER(HW_VOTER):
    def __init__(self,C,F=5,S=120,D=4,**args):
        voters = [HMA_ALG(C,i) for i in range(F,S,D)]
        HW_VOTER.__init__(self,*voters,C=C)

        
class HMA_ALG(UD_ALG):
    def __init__(self,C,N=40,**args):
        UD_ALG.__init__(self,C=C,idc = GIMP['IND'].HMA(N)(C))



class HMA_VOTER(VOT_ALG):
    def __init__(self,C,F=5,S=120,D=4,**args):
        voters = [ HMA_ALG(C,i) for i in range(F,S,D)]
        VOT_ALG.__init__(self,*voters,C=C)

class SIM_ALG(ALG):
    def __init__(self,C,N=15,**args):
        ALG.__init__(self,C=C)
        self.ad = GIMP['IND'].EXPSIM(N,**args)
    def __call__(self,**args):
        return self.ad(self.C).apply(lambda x:max(x,0))

class SIM_VOTER(VOT_ALG):
    def __init__(self,C,F=5,S=120,D=4,**args):
        voters = [SIM_ALG(C,i,**args) for i in range(F,S,D)]
        VOT_ALG.__init__(self,*voters,C=C)

class D_SIM(ALG):
    def __init__(self,C,U = 21,D = 5,**args):
        ALG.__init__(self,C=C)
        self.ut = IND.EXPSIM(U,**args)(self.C).apply(lambda x:max(x,0))
        self.dt = IND.EXPSIM(D,tau=-0.3,**args)(self.C).apply(lambda x:max(x,0))
    def __call__(self,**args):

        return (self.ut-1.5*self.dt).apply(lambda x:max(x,0))

from PAT import PATS



def AID(alg,*pats,**args):
    N = 5
    if 'N' in args:
        N = args['N']
    ddec = 'exp'
    if 'ddec' in args:
        ddec = args['ddec']
    dinc = 'exp'
    if dinc in args:
        dinc = args['dinc']

    assert type(ddec) in (float,int,str)
    if type(ddec) is str:
        assert ddec[:3] in ['exp','lin','div','min']
    else:
        assert((ddec<=1)and(ddec>=0))

    assert type(dinc) in (float,int,str)
    if type(dinc) is str:
        assert dinc[:3] in ['exp','lin','add']
    else:
        assert((ddec<=1)and(ddec>=0))








