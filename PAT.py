import talib as tb
import numpy as np
import pandas as pd 
import IND
import ALGS
from scipy import special


class PAT:
    req = 'cohlv'
    ds = 3
    def __init__(self,**args):
        self.C,self.O,self.H,self.L,self.V = None,None,None,None,None
        try:
            self.penetration = args['penetration']
        except:
            self.penetration = 0.3
        if 'C' in args:
            self.C = args['C']
        if 'O' in args:
            self.O = args['O']
        if 'L' in args:
            self.L = args['L']
        if 'H' in args:
            self.H = args['H'] 
        if 'V' in args:
            self.V = args['V']
    def __call__(self,**args):
        pass
    def data_packer(self):
        return pd.DataFrame({'Adj_Close':self.C,'Adj_Open':self.O,'Adj_Volume':self.V,'Adj_High':self.H,'Adj_Low':self.L})
pss = ['CDLABANDONEDBABY','CDLDARKCLOUDCOVER','CDLEVENINGDOJISTAR','CDLEVENINGSTAR','CDLMATHOLD','CDLMORNINGDOJISTAR','CDLMORNINGSTAR']

cmd = """
class PAT_{x}(PAT):
    def __init__(self,**args):
        PAT.__init__(self,**args)
    def __call__(self,**args):
        return tb.CDL{x}(self.O,self.H,self.L,self.C{y})/100
"""


PATS = []
dirs = vars(tb)
for attr in dirs.keys():
    if attr.startswith('CDL'):
        exec(cmd.format(x = attr[3:],y=',self.penetration' if attr in pss else ''))
        PATS.append(eval('PAT_{}'.format(attr[3:]) ))


PAT_BREAKAWAY.ds = 5
PAT_LADDERBOTTOM.ds = 5
PAT_MATHOLD.ds = 5
PAT_RISEFALL3METHODS.ds = 5
PAT_XSIDEGAP3METHODS.ds = 5

PAT_CONCEALBABYSWALL.ds = 4

PAT_GAPSIDESIDEWHITE.ds = 2
PAT_DARKCLOUDCOVER.ds = 2
PAT_COUNTERATTACK.ds = 2
PAT_HARAMI.ds = 2
PAT_HARAMICROSS.ds = 2
PAT_BELTHOLD.ds = 2
PAT_HOMINGPIGEON.ds = 2
PAT_INNECK.ds = 2
PAT_KICKING.ds = 2
PAT_KICKINGBYLENGTH.ds = 2
PAT_MATCHINGLOW.ds = 2
PAT_ONNECK.ds = 2
PAT_PIERCING.ds = 2
PAT_SEPARATINGLINES.ds = 2

PAT_DOJI.ds = 1
PAT_DOJISTAR.ds = 1
PAT_DRAGONFLYDOJI.ds = 1
PAT_CLOSINGMARUBOZU.ds = 1 # trend rem
PAT_ENGULFING.ds = 1
PAT_GRAVESTONEDOJI.ds = 1 # bot rev
PAT_HAMMER.ds = 1
PAT_HANGINGMAN.ds = 1
PAT_INVERTEDHAMMER.ds = 1
PAT_LONGLEGGEDDOJI.ds = 1
PAT_LONGLINE.ds = 1
PAT_MARUBOZU.ds = 1
PAT_RICKSHAWMAN.ds = 1
PAT_SHOOTINGSTAR.ds = 1
PAT_SHORTLINE.ds = 1
PAT_SPINNINGTOP.ds = 1
PAT_TAKURI.ds = 1

corr = lambda x,y:np.corrcoef(x,y)[0,1]

def S_PAT_SEEKER(pat,data,thold = 0.7,**args):
    datax = data.copy()
    patx = pat.copy()
    datax['HL'] = datax.Adj_High - datax.Adj_Low
    datax['HC'] = datax.Adj_High - datax.Adj_Close
    datax['HO'] = datax.Adj_High - datax.Adj_Open
    datax['OC'] = datax.Adj_Open - datax.Adj_Close
    datax['CL'] = datax.Adj_Close - datax.Adj_Low
    datax['OL'] = datax.Adj_Open - datax.Adj_Low
    datax['V'] = datax.Adj_Volume
    patx['HL'] = patx.Adj_High - patx.Adj_Low
    patx['HC'] = patx.Adj_High - patx.Adj_Close
    patx['HO'] = patx.Adj_High - patx.Adj_Open
    patx['OC'] = patx.Adj_Open - patx.Adj_Close
    patx['CL'] = patx.Adj_Close - patx.Adj_Low
    patx['OL'] = patx.Adj_Open - patx.Adj_Low
    patx['V'] = pat.Adj_Volume
    
    return PAT_SEEKER(patx[['HL','HC','HO','OC','CL','OL','V']],datax[['HL','HC','HO','OC','CL','OL','V']],thold = thold,**args)

def PAT_SEEKER(pat,data,thold = 0.7,**args):

    paras = list(pat.columns)
    for p in paras:
        assert p in data

    res = data*0
    window = len(pat)
    L = len(data)
    assert L>window
    for p in paras:
        res[p] = data[p].rolling(window = window,center = False).apply(lambda x:corr(x,pat[p]))
    res = res.applymap(lambda x:max(0,x))
    res_ = (res>=thold).T.apply(np.prod)

    return res,res_

def PAT_SAVE(pat,name):
    pat.to_csv('SPATS/'+name)

def PAT_LOAD(name):
    pat = pd.DataFrame.from_csv('SPATS/'+name)
    class res(PAT):
        ds = len(pat)
        def __init__(self,thold = 0.7,**args):
            PAT.__init__(self,**args)
            try:
                self.data = args['data']
            except:
                self.data = self.data_packer()
            self.thold = thold
        def __call__(self):
            _,r = PAT_SEEKER(pat,self.data,thold = self.thold)
            z = str([str(x) for x in r[r!=0].index])
            print('{} parttens of {} found,{}'.format(r.sum(),name,z ))
            return r
    return res



def ut(x,a,b):
    return 0.5*special.erf((2*a+2*b-4*x)/(a-b) ) + 1.5

def dt(x,a,b):
    return 0.5*special.erf( -(2*a+2*b-4*x)/(a-b) ) + 1.5

def uh(x,a,b):
    return 2 - special.erf((2*a+2*b-4*x)/(a-b))**2

def dh(x,a,b):
    return special.erf((2*a+2*b-4*x)/(a-b))**2



