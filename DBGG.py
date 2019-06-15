import numpy as np
import inspect
import sys
import matplotlib.pyplot as plt
import pandas as pd
import uuid
from importlib import reload

import IND
import ALGS


GIMP = {
    'IND':IND,
    'ALGS':ALGS
}
def udt():
    GIMP['IND'] = reload(IND)
    GIMP['ALGS'] = reload(ALGS)



class PLOTER:
    def __init__(self,p,reg=[],**param):
        self.pp = p
        self.param = param
        self.pparam = {}
        self.uuid = uuid.uuid4()
        self.reg = reg
        try:
            self.pparam['color'] = param['color']
        except:
            pass
        try:
            self.pparam['alpha'] = param['alpha']
        except:
            pass
        try:
            self.pparam['width'] = param['width']
        except:
            pass
        reg.append(self)
    def plot(self,idx):
        if self.param['ana']:
            wh = self.c_ret(**self.param)
            self.param['pmain'].plot(range(len(idx)),wh['bch'].loc[idx],color = 'black')
            self.param['pmain'].plot(range(len(idx)),wh['alg'].loc[idx],**self.pparam)

        self.pp.plot(range(len(idx)),self.__call__(**self.param).loc[idx],**self.pparam)
    def remove(self):
        for i in range(len(self.reg)):
            if self.reg[i].uuid == self.uuid:
                self.reg.pop(i)
                break




def dbgg(G_var,**args):
    p_main = G_var['p_main']
    p_side = G_var['p_side']
    reg = G_var['G_ind_M']
    DATAF = G_var['DATAF']
    udt()

    ds = [35,50,65,80,95]

    class BOLLT:
        def __init__(self):
            u = pd.DataFrame()
            d = pd.DataFrame()
            self.c = []
            self.uuid = uuid.uuid4()
            for x in ds:
                ud = IND.BOLL(N=x,a=2)(DATAF['Adj_Close'])
                self.c.append(ud)
                u[x] = ud['u']
                d[x] = ud['d']
            self.ur = u.max(axis=1) - u.min(axis=1)
            self.dr = d.max(axis=1) - d.min(axis=1)
            reg.append(self)
        def plot(self,idx):
            for ud in self.c:
                u = ud['u'][idx]
                d = ud['d'][idx]
                p_main.fill_between(range(len(idx)),d,u,facecolor = 'blue',alpha = 0.1)
            p_side.plot(range(len(idx)),self.ur[idx],color = 'red')
            p_side.plot(range(len(idx)),self.dr[idx],color = 'black')
        def remove(self):
            for i in range(len(reg)):
                if reg[i].uuid == self.uuid:
                    reg.pop(i)




    BOLLT()



    plt.show(block = False)
