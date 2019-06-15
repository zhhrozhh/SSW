import matplotlib
matplotlib.use('tkagg')
import quandl
import sys
import os
import pandas as pd
import numpy as np
import uuid
import matplotlib.pyplot as plt
from matplotlib.widgets import Button,TextBox,CheckButtons,Slider,Cursor
from mpl_finance import candlestick2_ohlc
from matplotlib.dates import date2num,num2date
from matplotlib.ticker import Formatter
from pandas.core.window import _flex_binary_moment, _Rolling_and_Expanding
from matplotlib.patches import Circle,Patch
from math import sqrt
import inspect
import time
import talib
from importlib import reload
import DBGG
import ALGS
import IND
import PAT
import re
import tools

def load_pat_db():
    with open('scodes','r') as fp:
        G_var['PAT_DB']['scodes'] = eval(fp.read())
    print('LOADING',end = '')
    sys.stdout.flush()
    for scode in G_var['PAT_DB']['scodes']:
        #G_var['PAT_DB'][scode] = quandl.get('EOD/'+scode)[['Adj_Open','Adj_High','Adj_Low','Adj_Close','Adj_Volume']]
        G_var['PAT_DB'][scode] = tools.get_data(scode)
        print('-',end = '')
        sys.stdout.flush()
    print('>DONE')

def pat_anas(val = None,s = False):
    pat = G_var['pat_s']
    A_PATS = []
    print('SEARCHING',end = '')
    sys.stdout.flush()

    for scode in G_var['PAT_DB']['scodes']:
        data = G_var['PAT_DB'][scode]
        if not s:
            _,r = PAT.PAT_SEEKER(pat,data)
        else:
            _,r = PAT.S_PAT_SEEKER(pat,data)
        for idx in r[r!=0].index:
            gp = (data.loc[idx:][:len(pat)+G_var['PAT_DB']['ANA_WINDOW']]).iloc[len(pat):]
            if len(gp) == G_var['PAT_DB']['ANA_WINDOW']:
                A_PATS.append(gp.reset_index(drop = True))
                print('-',end = '')
                sys.stdout.flush()
    print('>DONE')
    if not len(A_PATS):
        return 
    CP = [(x-x.mean())/x.std() for x in A_PATS]
    CP = [x-x.iloc[0] for x in CP]
    D_ = sum(CP)/len(CP)
    fn = plt.figure(figsize = [5,5])
    ax = fn.add_axes([0,0,1,1])
    
    for x in CP:
        ax.plot(x.Adj_Close,alpha = 0.1,color = 'red')
    ax.plot(D_.Adj_Close)
    plt.show(block = False)


def update_spats():
    G_var['spats'] = os.listdir('SPATS')

def camaps():
    cmap = {}
    amap = {}
    fcmap = {}
    famap = {}
    cc = w = pd.DataFrame({'r':[np.NAN for i in range(361)],'g':[np.NAN for i in range(361)],'b':[np.NAN for i in range(361)]},index = range(361))
    cc.loc[0] = [255,0,0]
    cc.loc[60] = [255,255,0]
    cc.loc[120] = [0,255,0]
    cc.loc[180] = [0,255,255]
    cc.loc[240] = [0,0,255]
    cc.loc[300] = [255,0,255]
    cc.loc[360] = [255,0,0]
    cc = cc.interpolate().apply(lambda x:round(x))
    C = sum(G_var['seledpat'])
    I = 0
    for i in range(len(PAT.PATS)):
        if G_var['seledpat'][i]:
            d = cc.loc[round(I*(360/C))]
            cmap[PAT.PATS[i]] = '{:02x}{:02x}{:02x}'.format(int(d['r']),int(d['g']),int(d['b']))
            amap[PAT.PATS[i]] = 0.4
            I += 1
    C = len(G_var['ping_spats'])
    for i in range(C):
        d = cc.loc[round(i*360/C)]
        fcmap[G_var['ping_spats'][i]] = '{:02x}{:02x}{:02x}'.format(int(d['r']),int(d['g']),int(d['b']))
        famap[G_var['ping_spats'][i]] = 0.4


    G_var['cmap'] = cmap
    G_var['amap'] = amap
    G_var['fcmap'] = fcmap
    G_var['famap'] = famap



def update_legend():
    lgd = []
    for i in range(len(G_var['seledpat'])):
        if G_var['seledpat'][i]:
            lgd.append(Patch(facecolor = '#'+G_var['cmap'][PAT.PATS[i]],alpha = G_var['amap'][PAT.PATS[i]],label = PAT.PATS[i].__name__[4:]))
    for spats in G_var['ping_spats']:
        lgd.append(Patch(facecolor = '#'+G_var['fcmap'][spats],alpha = G_var['famap'][spats],label = spats))        
    p_pat_lgd.legend(handles = lgd)
    plt.show(block = False)

def dbg_udt():
    G_var['DBGG'] = reload(G_var['DBGG'])
    G_var['ALGS'] = reload(G_var['ALGS'])
    G_var['IND'] = reload(G_var['IND'])


print(matplotlib.get_backend())

def lno():
    return inspect.currentframe().f_back.f_lineno


def WMAW(n):
    s = sum(range(n+1))
    res = []
    for i in range(1,n+1):
        res.append(float(i)/s)
    return np.array(res)
def WMA(s,n):
    w = WMAW(n)
    return s.rolling(window = n,center = False).apply(lambda x:np.dot(x,w))

def bs_balance(bx,sx,tl,st = False):
    tl[:] = 0
    tl[bx] = 1
    tl[sx] = -1
    if st:
        tl *= -1
    ins = 0
    for i in range(len(tl)):
        if tl[i] == 1:
            if ins == 1:
                tl[i] = 0
            ins = 1
        if tl[i] == -1:
            if ins != 1:
                tl[i] = 0
            ins = -1

    bx,sx = bx[tl == 1],sx[tl == -1]
    if len(bx)>len(sx):
        bx.pop(bx.index[-1])

    if st:
        return sx,bx
    return bx,sx



    


class NSF(Formatter):
    def __init__(self, dates, fmt='%Y-%m-%d'):
        self.dates = dates
        self.fmt = fmt
    def __call__(self, x, pos=0):
        ind = int(np.round(x))
        if ind >= len(self.dates) or ind < 0:
            return ''
        return num2date(self.dates[ind]).strftime(self.fmt)


class Window:
    def __init__(self,width,height,name,unique = True):
        if unique and G_wind_ctrl[name]:
            raise Exception('unique window already opened')
        self.window_key = window_keygen()
        if self.window_key == -1:
            raise Exception('max window num achieved')
        self.fig = plt.figure(figsize = [width,height])
        self.fig.canvas.set_window_title(name)
        G_wind_reg[self.window_key] = self.fig
        if unique:
            G_wind_ctrl[name] = 1
        def close_h(event):
            if unique:
                G_wind_ctrl[name] = 0
            G_wind_reg.pop(self.window_key)
        self.fig.canvas.mpl_connect('close_event',close_h)


class Api_Setting_Window(Window):
    def __init__(self):
        Window.__init__(self,3,1.2,'set_api')
        self.inp_a = self.fig.add_axes([0.1,1-0.25,0.8,0.2])
        self.inp = TextBox(self.inp_a,'key')
        self.c_rem_a = self.fig.add_axes([0.1,1-0.25-0.29-0.04,0.3,0.25])
        self.c_rem = CheckButtons(self.c_rem_a,['save key'],[True])
        self.b_sub_a = self.fig.add_axes([0.1+0.3+0.05,1-0.25-0.29-0.04,0.3,0.25])
        self.b_sub = Button(self.b_sub_a,'submit')
        def sub(val):
            quandl.ApiConfig.api_key = self.inp.text
            if c_rem.get_status()[0]:
                with open('KeyR','w') as f:
                    f.write(self.inp.text)
                plt.close(self.fig)
        self.b_sub.on_clicked(sub)
        self.fig.show()
        plt.show(block = False)


class Scode_Setting_Window(Window):
    def __init__(self):
        Window.__init__(self,2.2,1.2,'set_scode')
        self.inp_a = self.fig.add_axes([0.1+0.15,1-0.25,0.6,0.2])
        self.inp = TextBox(self.inp_a,'symbol')
        self.b_sub_a = self.fig.add_axes([0.1,1-0.25-0.2-0.01,0.34,0.16])
        self.b_sub = Button(self.b_sub_a,'submit')

        def set_data(val):
            G_var['scode'] = self.inp.text
            #G_var['DATAF'] = quandl.get("EOD/"+self.inp.text)[['Adj_Open','Adj_Close','Adj_High','Adj_Low','Adj_Volume']]
            G_var['DATAF'] = tools.get_data(self.inp.text)
            G_var['beg_d'] = G_var['DATAF'].index[0]
            zoom_setter.valmax = len(G_var['DATAF'].index)
            zoom_setter.ax.set_xlim(zoom_setter.valmin,zoom_setter.valmax)
            zoom_setter.set_val(G_var['dds'] )
            beg_setter.valmax = len(G_var['DATAF'].index)
            beg_setter.ax.set_xlim(beg_setter.valmin,beg_setter.valmax)
            G_ind_M = []
            G_ind_S = []
            G_ind = {}

            plt.close(self.fig)
            update_plot()
        self.b_sub.on_clicked(set_data)
        self.inp.on_submit(set_data)
        self.fig.show()
        plt.show(block = False)


class SMA_Window(Window):
    def __init__(self,UWindow):
        Window.__init__(self,3,2,'SMA')
        G_dbg_reg[1] = self
        self.ds100_a = self.fig.add_axes([0.16,1-0.11,0.7,0.1])
        self.ds100 = Slider(self.ds100_a,'100N',0,5,valinit = 0)
        self.ds_a = self.fig.add_axes([0.16,1-0.11-0.1-0.01,0.7,0.1])
        self.ds = Slider(self.ds_a,'N',1,99,valinit = 3)
        self.cs_a = self.fig.add_axes([0.16,1-0.11-0.1-0.01-0.1-0.01,0.7,0.1])
        self.cs = TextBox(self.cs_a,'color',initial = 'CC0000')
        self.ws_a = self.fig.add_axes([0.16,1-0.11-0.1-0.01-0.1-0.01-0.1-0.01,0.7,0.1])
        self.ws = Slider(self.ws_a,'W',1,5,valinit = 1)
        self.add_a = self.fig.add_axes([0.36,1-0.11-0.1-0.01-0.1-0.01-0.1-0.01-0.1-0.01,0.2,0.1])
        self.add = Button(self.add_a,'add')
        self.UWindow = UWindow
        def add__(val):
            SMA(100*int(self.ds100.val)+int(self.ds.val),self.cs.text,int(self.ws.val))
            plt.close(self.fig)
            self.UWindow.update()
        self.add.on_clicked(add__)
        self.fig.show()
        plt.show(block = False)


class EMA_Window(Window):
    def __init__(self,UWindow):
        Window.__init__(self,3,2,'EMA')
        self.alpha_a = self.fig.add_axes([0.16,1-0.11,0.7,0.1])
        self.alpha = Slider(self.alpha_a,'alpha',0.0001,1,valinit = 0.0001)
        self.cs_a = self.fig.add_axes([0.16,1-0.11-0.1-0.01,0.7,0.1])
        self.cs = TextBox(self.cs_a,'color',initial = 'CC0000')
        self.ws_a = self.fig.add_axes([0.16,1-0.11-0.1-0.01-0.1-0.01,0.7,0.1])
        self.ws = Slider(self.ws_a,'W',1,5,valinit = 1)
        self.add_a = self.fig.add_axes([0.36,1-0.11-0.1-0.01-0.1-0.01-0.1-0.01,0.2,0.1])
        self.add = Button(self.add_a,'add')
        self.UWindow = UWindow
        def add__(val):
            EMA(float(self.alpha.val),self.cs.text,int(self.ws.val))
            plt.close(self.fig)
            self.UWindow.update()
        self.add.on_clicked(add__)
        self.fig.show()
        plt.show(block = False)


class Alligator_Window(Window):
    def __init__(self,UWindow):
        Window.__init__(self,3,2,'Alligator')
        self.jc_a = self.fig.add_axes([0.16,1-0.11,0.7,0.1])
        self.jc = TextBox(self.jc_a,'J Color',initial = 'CC0000')
        self.tc_a = self.fig.add_axes([0.16,1-0.11-0.01-0.1,0.7,0.1])
        self.tc = TextBox(self.tc_a,'T Color',initial = '00CC00')
        self.lc_a = self.fig.add_axes([0.16,1-0.11-0.01-0.1-0.01-0.1,0.7,0.1])
        self.lc = TextBox(self.lc_a,'L Color',initial = '0000CC')
        self.add_a = self.fig.add_axes([0.36,1-0.11-0.01-0.1-0.01-0.1-0.01-0.1,0.2,0.1])
        self.add = Button(self.add_a,'add')
        self.UWindow = UWindow
        def add__(val):
            Alligator(self.jc.text,self.tc.text,self.lc.text)
            plt.close(self.fig)
            self.UWindow.update()
        self.add.on_clicked(add__)
        self.fig.show()
        plt.show(block = False)


class Boll_Window(Window):
    def __init__(self,UWindow):
        Window.__init__(self,3,2,'Boll')
        self.ds_a = self.fig.add_axes([0.16,1-0.11,0.7,0.1])
        self.ds = Slider(self.ds_a,'N',1,100,valinit = 1)
        self.dc_a = self.fig.add_axes([0.16,1-0.11-0.01-0.1,0.7,0.1])
        self.dc = TextBox(self.dc_a,'color',initial = 'CC0000')
        self.xs_a = self.fig.add_axes([0.16,1-0.11-0.01-0.1-0.01-0.1,0.7,0.1])
        self.xs = Slider(self.xs_a,'alpha',0,1,valinit = 0.8)
        self.ns_a = self.fig.add_axes([0.16,1-0.11-0.01-0.1-0.01-0.1-0.01-0.1,0.7,0.1])
        self.ns = Slider(self.ns_a,'n',0,3,valinit = 1.5)
        self.add_a = self.fig.add_axes([0.36,1-0.11-0.01-0.1-0.01-0.1-0.01-0.1-0.01-0.1,0.2,0.1])
        self.add = Button(self.add_a,'add')
        self.UWindow = UWindow
        def add__(val):
            Boll(int(self.ds.val),self.dc.text,float(self.xs.val),float(self.ns.val))
            plt.close(self.fig)
            self.UWindow.update()
        self.add.on_clicked(add__)
        self.fig.show()
        plt.show(block = False)


class HMA_Window(Window):
    def __init__(self,UWindow):
        Window.__init__(self,3,2,'HMA')
        G_dbg_reg[1] = self
        self.ds100_a = self.fig.add_axes([0.16,1-0.11,0.7,0.1])
        self.ds100 = Slider(self.ds100_a,'100N',0,5,valinit = 2)
        self.ds_a = self.fig.add_axes([0.16,1-0.11-0.1-0.01,0.7,0.1])
        self.ds = Slider(self.ds_a,'N',1,99,valinit = 0)
        self.cs_a = self.fig.add_axes([0.16,1-0.11-0.1-0.01-0.1-0.01,0.7,0.1])
        self.cs = TextBox(self.cs_a,'color',initial = 'CC0000')
        self.ws_a = self.fig.add_axes([0.16,1-0.11-0.1-0.01-0.1-0.01-0.1-0.01,0.7,0.1])
        self.ws = Slider(self.ws_a,'W',1,5,valinit = 1)
        self.add_a = self.fig.add_axes([0.36,1-0.11-0.1-0.01-0.1-0.01-0.1-0.01-0.1-0.01,0.2,0.1])
        self.add = Button(self.add_a,'add')
        self.UWindow = UWindow
        def add__(val):
            HMA(100*int(self.ds100.val)+int(self.ds.val),self.cs.text,int(self.ws.val))
            plt.close(self.fig)
            self.UWindow.update()

        self.add.on_clicked(add__)
        self.fig.show()
        plt.show(block = False)


class Debug_Window(Window):
    def __init__(self):
        Window.__init__(self,5,0.3,'debug',False)
        self.inp_a = self.fig.add_axes([0,0,0.9,0.6])
        self.inp = TextBox(self.inp_a,'d')
        def sub(text):
            print(text)
            try:
                eval('print('+text+')')
            except:
                exec(text)
            self.inp.set_val('')
        self.inp.on_submit(sub)

        self.fig.show()
        plt.show(block = False)


class Setting_Window(Window):
    def __init__(self):
        Window.__init__(self,8,6,'set')

        self.b_set_mode_0_a = self.fig.add_axes([0.05,1-0.05,0.07,0.03])
        self.b_set_mode_0 = Button(self.b_set_mode_0_a,'candle')
        def sm0(val):
            G_var['plot_mode'] = 0
            update_plot()
        self.b_set_mode_0.on_clicked(sm0)

        self.b_set_mode_1_a = self.fig.add_axes([0.05+0.07+0.01,1-0.05,0.07,0.03])
        self.b_set_mode_1 = Button(self.b_set_mode_1_a,'line')
        def sm1(val):
            G_var['plot_mode'] = 1
            update_plot()
        self.b_set_mode_1.on_clicked(sm1)

        self.b_fibo_setting_a = self.fig.add_axes([0.05,1-0.05-0.03-0.05-0.01-0.1,0.1,0.14])
        self.b_fibo_setting = CheckButtons(self.b_fibo_setting_a,['ret','expend','timez','fans','arcs'],[True,True,True,True,True])
        def oc(val):
            tt = self.b_fibo_setting.get_status()
            G_var['fibo']['ret'] = tt[0]
            G_var['fibo']['expend'] = tt[1]
            G_var['fibo']['timez'] = tt[2]
            G_var['fibo']['fans'] = tt[3]
            G_var['fibo']['arcs'] = tt[4]
        self.b_fibo_setting.on_clicked(oc)

        self.ps_opt_a = self.fig.add_axes([0.05+0.15,1-0.05-0.03-0.05-0.01-0.1,0.1,0.06])
        self.ps_opt = CheckButtons(self.ps_opt_a,['volume'],[True])
        def occ(val):
            G_var['vw']['v'] = self.ps_opt.get_status()[0]
            update_plot()
        self.ps_opt.on_clicked(occ)
        plt.show(block = False)

class Pat_Setting_Window(Window):
    def __init__(self):
        Window.__init__(self,9,9,'p_set')
        self.cb_pats_a = self.fig.add_axes([0.02,0.12,0.96,0.86])
        self.cb_pats = CheckButtons(self.cb_pats_a,[x.__name__[4:] for x in PAT.PATS],G_var['seledpat'])
        def oc(val):
            G_var['seledpat'] = self.cb_pats.get_status()
            camaps()
        self.cb_pats.on_clicked(oc)
        self.b_unselect_all_a = self.fig.add_axes([0.02,0.12-0.03-0.01,0.07,0.03])
        self.b_unselect_all = Button(self.b_unselect_all_a,'clear')
        def od(val):
            for i in range(len(PAT.PATS)):
                if G_var['seledpat'][i]:
                    self.cb_pats.set_active(i)
            camaps()
        self.b_unselect_all.on_clicked(od)
        plt.show(block = False)


class Indicator:
    def __init__(self,H_type = 'M',I_type = None):
        self.H_type = H_type
        assert(I_type is not None)
        self.I_type = I_type
        
        if H_type == 'M':
            G_ind_M.append(self)
        elif H_type == 'S':
            G_ind_S.append(self)
    def finl(self):
        self.uuid = self.desc() +'@'+ str(uuid.uuid4())
    def remove(self):
        if self.H_type == 'M':
            for i in range(len(G_ind_M)):
                if G_ind_M[i].uuid == self.uuid:
                    G_ind_M.pop(i)
                    break
        elif self.H_type == 'S':
            for i in range(len(G_ind_S)):
                if G_ind_S[i].uuid == self.uuid:
                    G_ind_S.pop(i)
                    break
        G_ind.pop(self.uuid)
    def plot(self,idx):
        pass
    def desc(self):
        return self.I_type

class SMA(Indicator):
    def __init__(self,N,color,W):
        G_dbg_reg['sma'] = self
        assert(N>0)
        assert(W>0)
        Indicator.__init__(self,I_type = 'SMA')
        self.N = int(N)
        self.finl()
        self.color = color
        self.W = int(W)
        G_ind[self.uuid] = G_var['DATAF']['Adj_Close'].rolling(window = N,center = False).mean()
    def sma(a,b):
        return a.rolling(window = b,center = False).mean()
    def plot(self,idx):
        p_main.plot(range(len(idx)),G_ind[self.uuid].loc[idx],linewidth=self.W,color = '#'+self.color)

    def desc(self):
        return 'SMA' + str(self.N)


class PLOTER:
    def __init__(self,p,**param):
        self.p = p
        self.param = param
        self.pparam = {}
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
        G_ind_M.append(self)
    def plot(self,idx):
        self.p.plot(range(len(idx)),self.__call__(**self.param).loc[idx],**self.pparam)


class EMA(Indicator):
    def __init__(self,alpha,color,W):
        Indicator.__init__(self,I_type = 'EMA')
        self.alpha = alpha
        self.color = color
        self.W = W
        self.finl()
        G_ind[self.uuid] = G_var['DATAF']['Adj_Close'].ewm(alpha = self.alpha).mean()
    def plot(self,idx):
        p_main.plot(range(len(idx)),G_ind[self.uuid].loc[idx],linewidth = self.W,color = '#'+self.color)
    def desc(self):
        return 'EMA'+"%.2f"%self.alpha

class Alligator(Indicator):
    def __init__(self,colorJ,colorT,colorL):
        Indicator.__init__(self,I_type = 'Alligator')
        self.colorJ = colorJ
        self.colorT = colorT
        self.colorL = colorL
        H = G_var['DATAF']['Adj_High']
        L = G_var['DATAF']['Adj_Low']
        M = (H+L)/2
        self.finl()
        G_ind[self.uuid] = {
            'Jaw':M.rolling(window = 13,center = False).mean().shift(8),
            'Teeth':M.rolling(window = 8,center = False).mean().shift(5),
            'Lip':M.rolling(window = 5,center = False).mean().shift(3)
        }
    def plot(self,idx):
        dx = range(len(idx))
        p_main.plot(dx,G_ind[self.uuid]['Jaw'].loc[idx],color = '#'+self.colorJ)
        p_main.plot(dx,G_ind[self.uuid]['Teeth'].loc[idx],color = '#'+self.colorT)
        p_main.plot(dx,G_ind[self.uuid]['Lip'].loc[idx],color = '#'+self.colorL)
    def desc(self):
        return self.I_type

class Boll(Indicator):
    def __init__(self,N,color,alpha,n):
        Indicator.__init__(self,I_type = 'Boll')
        self.N = N
        self.color = color
        self.alpha = alpha
        self.n = n
        S = G_var['DATAF']['Adj_Close']
        m = S.rolling(window = N,center = False).mean()
        s = S.rolling(window = N,center = False).std()
        self.finl()
        G_ind[self.uuid] = {
            'ut' : m+self.n*s,
            'lt' : m-self.n*s
        }
    def plot(self,idx):
        u = G_ind[self.uuid]['ut'][idx]
        l = G_ind[self.uuid]['lt'][idx]
        p_main.fill_between(range(len(idx)),l,u,facecolor = '#'+self.color,alpha = self.alpha)
    def desc(self):
        return 'Boll {0}-{1:.2f}'.format(self.N,self.n)

class HMA(Indicator):
    def __init__(self,N,color,W):
        assert(N>2)
        assert(W>0)
        G_dbg_reg[2] = self
        Indicator.__init__(self,I_type = 'HMA')
        self.N = int(N)
        self.color = color
        self.W = W
        c = G_var['DATAF']['Adj_Close']
        self.finl()
        G_ind[self.uuid] = WMA(2*WMA(c,int(N/2)) - WMA(c,N),int(sqrt(N)))
    def hma(c,N):
        return WMA(2*WMA(c,int(N/2)) - WMA(c,N),int(sqrt(N)))
    def plot(self,idx):
        p_main.plot(range(len(idx)),G_ind[self.uuid][idx],linewidth = self.W,color = '#'+self.color)
    def desc(self):
        return 'HMA' + str(self.N)

class ANA:
    def __init__(self,oup = None):
        self.stt = pd.DataFrame()
        self.oup = oup

    def save(self):
        pass
    def proc(self,bp,sp,st = False):
        if st:
            bp,sp = sp,bp
        DD = G_var['DATAF']
        hmap = pd.Series(index = DD.index)
        umap = hmap.copy()
        with open(self.oup,'w') as ff:
            ff.write('XXX{}\n'.format(len(bp)))
            wcb,wcw = 0,0
            bbr,wwr = 1,1
            for i in range(len(bp)):
                bx,sx = bp.index[i],sp.index[i]
                if DD['Adj_High'][bx] <= DD['Adj_Low'][sx]:
                    wcw += 1
                if DD['Adj_Low'][bx] <= DD['Adj_High'][sx]:
                    wcb += 1
                hmap[bx],hmap[sx] = DD['Adj_High'][bx],DD['Adj_Low'][sx]
                umap[bx],umap[sx] = DD['Adj_Low'][bx],DD['Adj_High'][sx]
                hmap[bx:sx] = hmap[bx:sx].interpolate()
                umap[bx:sx] = umap[bx:sx].interpolate()
                wr = (DD['Adj_Low'][sx] - DD['Adj_High'][bx])/DD['Adj_High'][bx]
                br = (DD['Adj_High'][sx] - DD['Adj_Low'][bx])/DD['Adj_Low'][bx]
                bbr*=(1+br)
                wwr*=(1+wr)
                NN = len(hmap[bx:sx])
                ff.write('worst return:{}\nbest return:{}\nperiod:{}\n\n'.format(wr,br,NN ))
            ff.write('worst overall return:{}\nbest overall return:{}\n'.format(bbr-1,wwr-1))
            ff.write('wwc : {}, bwc:{}'.format(wcw,wcb))
        return hmap,umap

class HMA_ANA(Indicator,ANA):
    def __init__(self,a,**args):
        Indicator.__init__(self,I_type = 'HMA_ANA')
        oup = 'HMA_rep-{}-{}'.format(G_var['scode'],a)
        if 'oup' in args:
            oup = oup
        ANA.__init__(self,oup)
        des = 'HMA'+str(a)+'@'
        dq = None
        for key in G_ind.keys():
            if key.startswith(des):
                dq = G_ind[key]
        bp,sp = dq.copy(),dq.copy()
        inc = dq>=dq.shift(1)
        bp = inc & ~inc.shift(1).fillna(True)
        sp = ~inc & inc.shift(1).fillna(False)
        bp,sp = bs_balance(bp,sp,pd.Series(index = bp.index))
        assert(len(bp) == len(sp))
        hmap,umap = self.proc(bp,sp)
        self.finl()
        G_ind[self.uuid] = {'u':umap,'h':hmap}

    def plot(self,idx):
        p_main.plot(range(len(idx)), G_ind[self.uuid]['h'][idx] ,linewidth = 2.2,color = '#0F0F0A')
        p_main.plot(range(len(idx)), G_ind[self.uuid]['u'][idx] ,linewidth = 2.2,color = '#F0C010')

class TRIMA_ANA(Indicator,ANA):
    def __init__(self,a,b,c,**args):
        """
            a -> trend
            b -> slow
            c -> fast
        """
        Indicator.__init__(self,I_type = 'TRIMA_ANA')
        
        atype,btype,ctype,oup = 'SMA','SMA','SMA',None
        if 'atype' in args:
            atype = args['atype']
        if 'btype' in args:
            btype = args['btype']
        if 'ctype' in args:
            ctype = args['ctype']
        if 'oup' in args:
            oup = args['oup']
        if not oup:
            oup = 'rep-{}-{}-{}-{}.txt'.format(G_var['scode'],a,b,c)
        ANA.__init__(self,oup)
        adesc = atype + str(a) + '@'
        bdesc = btype + str(b) + '@'
        cdesc = ctype + str(c) + '@'
        df = pd.DataFrame()
        for key in G_ind.keys():
            if key.startswith(adesc):
                df['a'] = G_ind[key]
                print('found a')
            if key.startswith(bdesc):
                df['b'] = G_ind[key]
                print('found b')
            if key.startswith(cdesc):
                df['c'] = G_ind[key]
                print('found c')
        
        agbc = (df['a'] >= (df['b'] + df['c']/300)) & (df['a'] >= (df['c'] + df['c']/300))
        albc = (df['a'] <= (df['b'] - df['c']/300)) & (df['a'] <= (df['c'] - df['c']/300))
        dbc = (df['b'] - df['c'])
        dbcinc = (dbc > dbc.shift(1))
        dbcSpos = (dbc > df['c']/500)
        dbcSneg = (dbc < -df['c']/500)
        bp = albc & (dbcSneg & ~dbcSneg.shift(1).fillna(True))
        sb = ~albc | (dbcSpos & ~dbcSpos.shift(1).fillna(True))
        sp = agbc & (dbcSpos & ~dbcSpos.shift(1).fillna(True))
        ss = ~agbc | (dbcSneg & ~dbcSneg.shift(1).fillna(True))
        hmap = pd.Series(index = bp.index)
        bp,sb = bs_balance(bp,sb,hmap)

        assert(len(bp) == len(sb))


        hmap,umap = self.proc(bp,sb)
        self.finl()

        G_ind[self.uuid] = {'h':hmap,'u':umap}

    def plot(self,idx):
        p_main.plot(range(len(idx)), G_ind[self.uuid]['h'][idx] ,linewidth = 2.2,color = '#0F0F0A')
        p_main.plot(range(len(idx)), G_ind[self.uuid]['u'][idx] ,linewidth = 2.2,color = '#F0C010')

class MAVOT_ANA(Indicator,ANA):
    def __init__(self,**args):
        Indicator.__init__(self,I_type = 'MAVOT_ANA')
        oup = 'MAVOT'
        self.tp = 'hma'
        if 'tp' in args:
            self.tp = args['tp']
        if 'oup' in args:
            oup = args['oup']
        ANA.__init__(self,oup)
        self.N = 150
        if 'N' in args:
            self.N = args['N']
        self.eps = 0
        if 'eps' in args:
            self.eps = args['eps']
        assert abs(self.eps) <1
        self.rg = range(10,10+self.N)
        if 'rg' in args:
            self.rg = args['rg']
            print(self.rg)

        if 'voter' in args:
            self.voter = args['voter']
            self.N = len(self.voter)
        else:
            if self.tp == 'hma':
                self.voter = pd.DataFrame({i:HMA.hma(G_var['DATAF']['Adj_Close'],i) for i in self.rg}).fillna(G_var['DATAF']['Adj_Close'].iloc[0])
            else:
                self.voter = pd.DataFrame({i:SMA.sma(G_var['DATAF']['Adj_Close'],i) for i in self.rg}).fillna(G_var['DATAF']['Adj_Close'].iloc[0])
            self.N = len(self.voter)
        pthod = 0.5
        if 'pthod' in args:
            pthod = args['pthod']
        self.td = self.voter > self.voter.shift(1)
        hmap = self.td.sum(axis = 1)/self.N 
        bp = hmap>(pthod+self.eps)
        sp = hmap<(pthod-self.eps)
        bp,sp = bs_balance(bp,sp,pd.Series(index = bp.index))

        bmap,wmap = self.proc(bp,sp)
        self.finl()
        G_ind[self.uuid] = {'u':bmap,'h':wmap,'k':hmap}

    def plot(self,idx):
        p_main.plot(range(len(idx)), G_ind[self.uuid]['h'][idx] ,linewidth = 2.2,color = '#0F0F0A')
        p_main.plot(range(len(idx)), G_ind[self.uuid]['u'][idx] ,linewidth = 2.2,color = '#F0C010')
        kmap = G_ind[self.uuid]['k'][idx]
        p_main.axvspan(0,0.5,alpha = kmap.iloc[0]/1.3,color = 'blue')
        for i in range(1,len(idx)-1):
            p_main.axvspan(i-0.5,i+0.5,alpha = kmap.iloc[i]/1.3,color = 'blue')
        p_main.axvspan(len(idx)-1,len(idx)-0.5,alpha = kmap.iloc[-1]/1.3,color = 'blue')

class Command_Trigger:
    def __init__(self):
        self.trigger = False
    def __call__(self):
        self.trigger = not self.trigger
        update_plot()

class AZ(Indicator,Command_Trigger):
    def __init__(self,func = None,dt = None):
        Indicator.__init__(self,I_type = 'xxxx')
        Command_Trigger.__init__(self)
        if func:
            self.ibc = func(G_var['DATAF'])
        if dt:
            self.ibc = dt
        self.ibc = self.ibc.shift(1)
    def plot(self,idx):
        if self.trigger:
            return
        kmap = self.ibc[idx]
        p_main.axvspan(0,0.5,alpha = kmap.iloc[0]/1.22,color = 'green')
        p_main.axvspan(len(idx)-1,len(idx)-0.5,alpha = kmap.iloc[-1]/1.22,color = 'green')
        for i in range(1,len(idx)-1):
            p_main.axvspan(i-0.5,i+0.5,alpha = (kmap.iloc[i]+1)/2.22,color = 'green')
    def jkr(self,**args):
        try:
            G_var['DBGG'] = reload(G_var['DBGG'])
            print(1)
        except Exception as e:
            print(e.__repr__())
            import DBGG
            print(0)
        G_var['DBGG'].dbgg(G_var,**args)
    def akj_plot(self):
        f = plt.figure(figsize = (8,3))
        a = f.add_axes([-0.5,0,1,1])
        w = self.ibc.apply(lambda x:max(0,x))
        rm = (G_var['DATAF']['Adj_Close'] - G_var['DATAF']['Adj_Close'].shift(1))/G_var['DATAF']['Adj_Close'].shift(1)
        bres = pd.Series(np.ones(len(rm.index)),index = rm.index)
        mres = pd.Series(np.ones(len(rm.index)),index = rm.index)
        for i in range(1,len(rm.index)):
            try:
                bres.iloc[i] = bres.iloc[i-1]*(1+rm.iloc[i])
                mres.iloc[i] = mres.iloc[i-1]*(1+rm.iloc[i])*w.iloc[i] + mres.iloc[i-1]*(1-w.iloc[i])
            except:
                pass
        a.plot(bres,color = 'blue')
        a.plot(mres,color = 'red')
        f.show()
        plt.show(block = False)





class Main_Indicator_Manager_Window(Window):
    def __init__(self):
        Window.__init__(self,4,6,'MIMW')
        G_dbg_reg[0] = self

        self.b_sma_a = self.fig.add_axes([0.02,1-0.035,0.1,0.03])
        self.b_sma = Button(self.b_sma_a,'sma')
        self.b_sma.on_clicked(lambda val:SMA_Window(self))

        self.b_ema_a = self.fig.add_axes([0.02+0.01+0.1,1-0.035,0.1,0.03])
        self.b_ema = Button(self.b_ema_a,'ema')
        self.b_ema.on_clicked(lambda val:EMA_Window(self))

        self.b_alli_a = self.fig.add_axes([0.02+0.01+0.1+0.01+0.1,1-0.035,0.15,0.03])
        self.b_alli = Button(self.b_alli_a,'alligator')
        self.b_alli.on_clicked(lambda val:Alligator_Window(self))

        self.b_boll_a = self.fig.add_axes([0.02+0.01+0.1+0.01+0.1+0.01+0.1+0.05,1-0.035,0.1,0.03])
        self.b_boll = Button(self.b_boll_a,'boll')
        self.b_boll.on_clicked(lambda val:Boll_Window(self))

        self.b_hma_a = self.fig.add_axes([0.02+0.01+0.1+0.01+0.1+0.01+0.1+0.05+0.1+0.01,1-0.035,0.1,0.03])
        self.b_hma = Button(self.b_hma_a,'hma')
        self.b_hma.on_clicked(lambda val:HMA_Window(self))

        self.t_a = [self.fig.add_axes([0.03,1-0.2-0.04*i,0.4,0.03]) for i in range(6)]
        self.t = [TextBox(self.t_a[i],'',initial = '') for i in range(6)]
        self.b_a = [self.fig.add_axes([0.03+0.4+0.02,1-0.2-(0.03+0.01)*i,0.03,0.03]) for i in range(6)]
        self.b = [Button(self.b_a[i],'x') for i in range(6)]
        for i in range(6):
            self.b[i].on_clicked(self.bb(i))

        self.b_u_a = self.fig.add_axes([0.23,0.56,0.03,0.03])
        self.b_u = Button(self.b_u_a,"^")
        self.b_d_a = self.fig.add_axes([0.33,0.56,0.03,0.03])
        self.b_d = Button(self.b_d_a,"v")


        def gd(val):
            self.idk+=1
            self.update()
        def gu(val):
            self.idk-=1
            self.update()

        self.b_d.on_clicked(gd)
        self.b_u.on_clicked(gu)

        self.idk = 0
        self.idlst = []
        self.update()
     
    def bb(self,n):
        def r(val):
            G_ind_M.pop(self.idk+n).remove()
            self.update()
        return r

    def update(self):
        L = len(G_ind_M)
        self.b_u.set_active(False)
        self.b_d.set_active(False)
        if self.idk+6 > L:
            self.idk = L-min(L,6)

        for i in range(L,6):
            self.t[i].set_val('')
            self.t[i].set_active(False)
            self.b[i].set_active(False)
        for i in range(min(L,6)):
            self.t[i].set_val(G_ind_M[self.idk+i].desc())
            self.t[i].set_active(True)
            self.b[i].set_active(True)
        if (L>6)and(self.idk+6<L):
            self.b_d.set_active(True)
        if (L>6)and(self.idk>0):
            self.b_u.set_active(True)

        plt.show(block = False)
        update_plot()

        
class Aid_Tools_Manager_Window(Window):
    def __init__(self):
        Window.__init__(self,4,6,'AID')
        self.idk = 0
        self.idlst = []
        self.update()
    def update(self):
        self.fig.clear()
        self.b_line_seg_a = self.fig.add_axes([0.02,1-0.035,0.14,0.03])
        self.b_line_seg = Button(self.b_line_seg_a,'lineseg')
        def lseg(val):
            if G_HEvent[0]:
                raise Exception('event in used')
            G_HEvent[0]=f_main.canvas.mpl_connect('button_press_event', onclick)
        self.b_line_seg.on_clicked(lseg)
        update_plot()



f_main = plt.figure(figsize = [14,8])
def load(scode):
    G_var['scode'] = scode
    #G_var['DATAF'] = quandl.get("EOD/"+scode)[['Adj_Open','Adj_Close','Adj_High','Adj_Low','Adj_Volume']]
    G_var['DATAF'] = tools.get_data(scode)
    G_var['beg_d'] = G_var['DATAF'].index[0]
    zoom_setter.valmax = len(G_var['DATAF'].index)
    zoom_setter.ax.set_xlim(zoom_setter.valmin,zoom_setter.valmax)
    zoom_setter.set_val(G_var['dds'] )
    beg_setter.valmax = len(G_var['DATAF'].index)
    beg_setter.ax.set_xlim(beg_setter.valmin,beg_setter.valmax)
    G_ind_M = []
    G_ind_S = []
    G_ind = {}
    update_plot()


def all_close(event):
    for w in list(G_wind_reg.keys()):
        plt.close(G_wind_reg[w])
f_main.canvas.mpl_connect('close_event',all_close)

p_main = f_main.add_axes([0.1,0.25+0.03,0.73,0.65-0.03])
p_side = f_main.add_axes([0.1,0.03+0.03,0.73,0.18-0.03])
p_side.axes.get_xaxis().set_visible(False)

p_pat_lgd = f_main.add_axes([0.85,0.1,0.14,0.8])
p_pat_lgd.axes.get_xaxis().set_visible(False)
p_pat_lgd.axes.get_yaxis().set_visible(False)
p_pat_lgd.axis('off')

p_dtx = f_main.add_axes([0.1,0.01,0.8,0.03])
t_dtx = TextBox(p_dtx,'DBG')
dbg_log = ['']



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


class PAT_PLOTER:
    def __init__(self,p,reg = [],**param):
        self.pp = p
        self.param = param
        self.uuid = uuid.uuid4()
        self.reg = reg
        self.color = param['color']
        self.alpha = param['alpha']
        self.ss = self.__call__()
        reg.append(self)
    def plot(self,idx):
        w = self.ss[idx]
        for i in range(self.ds,len(w)):
            if w.iloc[i] != 0:
                self.pp.axvspan(i-self.ds+0.5,i+0.5,color = '#'+self.color,alpha = self.alpha)

                self.pp.text(i-self.ds+1,sum(self.pp.get_ylim())/2 , '+' if w.iloc[i]==1 else '-')
    def remove(self):
        for i in range(len(self.reg)):
            if self.reg[i].uuid == self.uuid:
                self.reg.pop(i)
                break


def seek_pat(thold = 0.7):

    pat = G_var['pat_s']
    data = G_var['DATAF']
    _,r = PAT.PAT_SEEKER(pat,data,thold)
    s = r.sum()
    z = str([str(x) for x in r[r!=0].index])
    class tem(PAT_PLOTER):
        ds = len(pat)
        def __init__(self):
            PAT_PLOTER.__init__(self,p_main,reg = G_ind_M,color = 'FF0000',alpha = 0.2)
        def __call__(self,**args):
            return r
    tem()
    update_plot()
    print('{} patterns found, {}'.format(s,z))



def data_spr(req):
    res = {}
    if 'o' in req:
        res['O'] = G_var['DATAF']['Adj_Open']
    if 'h' in req:
        res['H'] = G_var['DATAF']['Adj_High']
    if 'l' in req:
        res['L'] = G_var['DATAF']['Adj_Low']
    if 'c' in req:
        res['C'] = G_var['DATAF']['Adj_Close']
    if 'v' in req:
        res['V'] = G_var['DATAF']['Adj_Volume']
    return res


def dbg_sub(text):
    if '#' in text:
        text = re.sub(r"\#([a-zA-Z][a-zA-Z0-9_]*)",r"G_var['dbg_str_v']['\1']",text)
    s = re.match(r'!([a-zA-Z][a-zA-Z0-9_]*)\s*([a-zA-Z][a-zA-Z0-9_]*)?(.*)',text)
    if s:
        cmd = s.groups()[0]
        arg = s.groups()[1]
        oth = s.groups()[2]
        if cmd == 'load':
            load(arg)
        elif cmd == 'loaddb':
            load_pat_db()
        elif cmd == 'anapat':
            pat_anas()
        elif cmd == 'sanapat':
            pat_anas(s = True)
        elif cmd == 'vlm':
            assert arg in ['on','off']
            if arg == 'on':
                G_var['vw']['v'] = True
            else:
                G_var['vw']['v'] = False
        elif cmd == 'pp':
            assert arg in ['on','off']
            if arg == 'on':
                G_var['pp'] = True
            else:
                G_var['pp'] = False
        elif cmd == 'quit':
            quit()
        elif cmd.upper() == 'PAT':
            G_var['temp_res'] = []
            for i in range(len(PAT.PATS)):
                pat = PAT.PATS[i]
                if G_var['seledpat'][i]:
                    class res(pat,PAT_PLOTER):
                        def __init__(self):
                            pat.__init__(self,penetration=G_var['pat_pent'],**data_spr(pat.req))
                            PAT_PLOTER.__init__(self,p=p_main,reg = G_ind_M,color = G_var['cmap'][pat],alpha = G_var['amap'][pat])
                    G_var['temp_res'].append(res())
            update_legend()
        elif cmd.upper() == 'SEEKPAT':

            if oth:
                seek_pat(float(oth))
            seek_pat()
        elif cmd.upper() == 'SVPAT':
            PAT.PAT_SAVE(G_var['pat_s'],arg)
            update_spats()

        elif cmd.upper() == 'SPAT':
            
            assert arg in G_var['spats']
            G_var['ping_spats'].append(arg)
            camaps()
            pat = PAT.PAT_LOAD(arg)
            try:
                thold = float(oth)
            except:
                thold = 0.7
            class res(pat,PAT_PLOTER):
                def __init__(self):
                    pat.__init__(self,thold = thold,data = G_var['DATAF'])
                    PAT_PLOTER.__init__(self,p = p_main,reg = G_ind_M,color = G_var['fcmap'][arg],alpha = G_var['famap'][arg])
                def remove(self):
                    PAT_PLOTER.remove(self)
                    for i in range(len(G_var['ping_spats'])):
                        if G_var['ping_spats'][i] == arg:
                            G_var['ping_spats'].pop(i)
                            return
            res()
            update_legend()
        elif cmd == 'clear':
            clear_l(0)
        elif cmd.upper() in ['IND','ALGS']:

            ind = G_var[cmd.upper()].__getattribute__(arg.upper())
            para = oth.split()
            paramI = []
            paramP = {}
            paramS = {}

            I = -1
            for i in range(len(para)):
                if para[i][0] in '-+':
                    I = i
                    break
                else:
    
                    paramI.append(para[i])

            if I>=0:
                for i in range(I,len(para),2):

                    if para[i][0] == '-':
                        paramP[para[i][1:]] = para[i+1]
                    elif para[i][0] == '+':
                        paramS[para[i][1:]] = para[i+1]
                    


            pt = []
            if 'pt' not in paramP and cmd.upper() == 'IND':
                paramP['pt'] = ind.req
            else:
                paramP['pt'] = ''
                
            x = paramP['pt']
            if 'o' in x:
                pt.append('Adj_Open')
            if 'c' in x:
                pt.append('Adj_Close')
            if 'h' in x:
                pt.append('Adj_High')
            if 'l' in x:
                pt.append('Adj_Low')
            if 'v' in x:
                pt.append('Adj_Volume')
            if 'ana' not in paramP:
                paramP['ana'] = False
            paramP['pmain'] = p_main
            paramP['pside'] = p_side

            if cmd.upper() == 'IND':
                p = p_main
                paramP['data'] = G_var['DATAF'][pt]
            else:
                p = p_side
                paramP['C'] = G_var['DATAF'].Adj_Close
                paramP['O'] = G_var['DATAF'].Adj_Open
                paramP['H'] = G_var['DATAF'].Adj_High
                paramP['L'] = G_var['DATAF'].Adj_Low
                paramP['V'] = G_var['DATAF'].Adj_Volume
            if 'pp' in paramP:
                if paramP['pp'] == 's':
                    p = p_side
                else:
                    p = p_main
            if 'bin' not in paramS:
                if cmd.upper() == 'IND':
                    paramS['bin'] = False
                else:
                    paramS['bin'] = True
            if 'SMO' in paramP:
                ind = IND.SMOOTH(ind,method = eval('IND.'+paramP['SMO']),Sarg = paramS,**paramP)

            class res(ind,PLOTER):
                def __init__(self):
                    ind.__init__(self,*paramI,**paramP)
                    PLOTER.__init__(self,p,G_ind_M,**paramP)
            G_var['temp_res'] = res()

            if 'var' in paramP:
                G_var['dbg_str_v'][paramP['var']] = G_var['temp_res']

        update_plot()
    else:
        try:
            eval('print('+text+')')
        except:
            exec(text)
    dbg_log.append(text)
    if len(dbg_log) >= G_var['dbg_log_max']:
        dbg_log.pop(0)
    G_var['dbg_idx'] = len(dbg_log)
    t_dtx.set_val(G_var['dbg_def_ph'])

    
def dbg_chg(event):
    if event.key == 'up':
        G_var['dbg_idx'] -= 1
        if G_var['dbg_idx'] < 0:
            G_var['dbg_idx'] = 0
        t_dtx.set_val(dbg_log[G_var['dbg_idx']])
    elif event.key == 'down':
        G_var['dbg_idx'] += 1
        if G_var['dbg_idx'] >= len(dbg_log):
            G_var['dbg_idx'] = len(dbg_log) - 1
        t_dtx.set_val(dbg_log[G_var['dbg_idx']])
    elif event.key == 'enter':
        dbg_sub(t_dtx.text)
    elif event.key == 'control':
        if G_var['dbg_def_ph'] == '':
            G_var['dbg_def_ph'] = '!'
            t_dtx.set_val('!'+t_dtx.text)
        elif G_var['dbg_def_ph'] == '!':
            G_var['dbg_def_ph'] = ''
            if t_dtx.text[0] == '!':
                t_dtx.set_val(t_dtx.text[1:])


f_main.canvas.mpl_connect('key_press_event',dbg_chg)


#quandl.ApiConfig.api_key = "rixaXs71r2KgkmPHW9jZ"

G_ind_M = []
G_ind_S = []
G_ind = {
    
}
G_aid = []
G_wind_ctrl = {
    'set_api' : 0,
    'set_scode' : 0,
    'set' : 0,
    'p_set' : 0,
    'MIMW' : 0,
    'SMA' : 0,
    'EMA' : 0,
    'Alligator' : 0,
    'Boll' : 0,
    'HMA' : 0,
    'AID' : 0,

}
G_wind_reg = {
    
}
G_dbg_reg = {
    
}
G_HEvent = {
    
}

G_var = {
    'PAT_DB':{
        'scodes':[],
        'data':{},
        'ANA_WINDOW':10,
    },
    'ping_spats':[],
    'seledpat':[False for x in PAT.PATS],
    'pp':True,
    'dbg_str_v':{

    },
    'pat_pent':0.3, 
    'dbg_log_max':200,
    'dbg_def_ph':'',
    'dbg_idx':0,
    'G_ind_M':G_ind_M,
    'p_main':p_main,
    'p_side':p_side,
    'DATAF':[],
    'dds':70,
    'beg_d':None,
    'plot_mode':0,# 0 -> candle stick | 1 -> line
    'max_window_num':50,
    'current_window_key':0,
    'l_seg_cc':[],
    'mc_cid':None, 
    'fibo':{
        'ret':True,
        'expend':True,
        'timez':True,
        'fans':True,
        'arcs':True,
    },
    'DBGG': DBGG,
    'IND': IND,
    'ALGS' : ALGS,
    'vw':{
        'v':True
    },
}

update_spats()
camaps()

def window_keygen():
    if len(G_wind_reg.keys()) == G_var['max_window_num']:
        return -1
    cwk = G_var['current_window_key']
    while cwk in G_wind_reg:
        cwk = (cwk+1)%100
    G_var['current_window_key'] = cwk
    return cwk


def update_plot():
    D = G_var['DATAF'].loc[G_var['beg_d']:][:G_var['dds']]
    p_main.clear()
    p_side.clear()
    fmt = NSF( pd.Series(D.index).apply(date2num) )
    p_main.xaxis.set_major_formatter(fmt)
    p_side.xaxis.set_major_formatter(fmt)
    if G_var['vw']['v']:
        p_side.bar(range(len(D.index)),D['Adj_Volume'])
    p_side.axes.get_xaxis().set_visible(False)
    if G_var['pp']:
        if G_var['plot_mode']:
            p_main.plot(range(len(D.index)),D['Adj_Close'])
        else:
            candlestick2_ohlc(p_main,D['Adj_Open'].values,D['Adj_High'].values,D['Adj_Low'].values,D['Adj_Close'].values,width = 12/G_var['dds'])
        p_main.set_ylim(D['Adj_Low'].min()*0.95,D['Adj_High'].max()*1.05 )
    for x in G_ind_M:
        x.plot(D.index)
    for x in G_aid:
        if x[0] == 'seg':
            p_main.plot(range(len(D.index)),x[1][D.index],color = x[2],alpha = 0.8)
        elif x[0] == 'fibo':
            if x[1][0] == 'ret':
                p_main.plot(range(len(D.index)),x[1][1][D.index],color = '#33DF2A',alpha =0.1 )
                p_main.plot(range(len(D.index)),x[1][2][D.index],color = '#33DF2A',alpha =0.236 )
                p_main.plot(range(len(D.index)),x[1][3][D.index],color = '#33DF2A',alpha =0.382 )
                p_main.plot(range(len(D.index)),x[1][4][D.index],color = '#33DF2A',alpha =0.5 )
                p_main.plot(range(len(D.index)),x[1][5][D.index],color = '#33DF2A',alpha =0.618 )
                p_main.plot(range(len(D.index)),x[1][6][D.index],color = '#33DF2A',alpha =1 )

            for itm in x[2:]:
                if itm[0] == 'expd':
                    p_main.plot(range(len(D.index)),itm[1][D.index],linestyle='--',color = '#33DF2A',alpha = 0.618)
                if itm[0] == 'timez':
                    dx,x = itm[1],itm[2]
                    for n in [0,1,3,6,11,19,32,53]:
                        p_main.axvline(x+n*dx - int(beg_setter.val),color = 'yellow')
                if itm[0] == 'fans':
                    alp = 0.2
                    for lin in itm[1]:
                        p_main.plot(range(len(D.index)),lin[D.index],color = '#1111CC',alpha = alp)
                        alp += 0.2
                if itm[0] == 'arcs':
                    dist,x,y = itm[1:4]
                    c1 = Circle((x - int(beg_setter.val),y),radius = dist*0.382 ,color = '#CC0ADC',fill = False,alpha = 0.382)
                    c2 = Circle((x - int(beg_setter.val),y),radius = dist*0.5 ,color = '#CC0ADC',fill = False,alpha = 0.5)
                    c3 = Circle((x - int(beg_setter.val),y),radius = dist*0.618 ,color = '#CC0ADC',fill = False,alpha = 0.618)
                    p_main.add_patch(c1)
                    p_main.add_patch(c2)
                    p_main.add_patch(c3)




    
    p_main.set_xlim(0,len(D.index))
    p_side.set_xlim(0,len(D.index))
    f_main.autofmt_xdate()
    plt.show(block = False)

b_apik_a = f_main.add_axes([0.02,1-0.04,0.07,0.03])
b_apik = Button(b_apik_a,"api key")
b_apik.on_clicked(lambda val:Api_Setting_Window())

b_load_a = f_main.add_axes([0.02+0.07+0.01,1-0.04,0.07,0.03])
b_load = Button(b_load_a,'load')
b_load.on_clicked(lambda val:Scode_Setting_Window())

b_debug_a = f_main.add_axes([0.02+0.07+0.01+0.07+0.01,1-0.04,0.07,0.03])
b_debug = Button(b_debug_a,'debug')
b_debug.on_clicked(lambda val:Debug_Window() )

b_setting_a = f_main.add_axes([0.02+0.07+0.01+0.07+0.01+0.07+0.01,1-0.04,0.07,0.03])
b_setting = Button(b_setting_a,'setting')
b_setting.on_clicked(lambda val:Setting_Window() )


def update(val):
    G_var['dds'] = int(zoom_setter.val)

    v = int(beg_setter.val)
    l = len(G_var['DATAF'].index)
    G_var['beg_d'] = G_var['DATAF'].index[v]

    if zoom_setter.valmax + v != l:
        if zoom_setter.val > l - v:
            zoom_setter.val = l - v
        zoom_setter.valmax = l - v

    zoom_setter.ax.set_xlim(zoom_setter.valmin,zoom_setter.valmax)
    update_plot()


zoom_setter_a = f_main.add_axes([0.02+0.04,1-0.04-0.03-0.01,0.4,0.03])
zoom_setter = Slider(zoom_setter_a,'zoom',1,10,valinit = 1)
zoom_setter.on_changed(update)

beg_setter_a = f_main.add_axes([0.02+0.04+0.4+0.02+0.04+0.04,1-0.04-0.03-0.01,0.4,0.03])
beg_setter = Slider(beg_setter_a,'ini',1,10,valinit = 1)
beg_setter.on_changed(update)

b_idm_a = f_main.add_axes([0.02+0.07+0.01+0.07+0.01+0.07+0.01+0.07+0.01,1-0.04,0.07,0.03])
b_idm = Button(b_idm_a,'indicatorsM')
b_idm.on_clicked(lambda val:Main_Indicator_Manager_Window())

b_aid_a = f_main.add_axes([0.02+0.07+0.01+0.07+0.01+0.07+0.01+0.07+0.01+0.07+0.01,1-0.04,0.07,0.03])
b_aid = Button(b_aid_a,'aid tools')
b_aid.on_clicked(lambda val:Aid_Tools_Manager_Window())

b_lin_s_a = f_main.add_axes([0.02+0.07+0.01+0.07+0.01+0.07+0.01+0.07+0.01+0.07+0.01+0.07+0.01,1-0.04,0.07,0.03])
b_lin_s = Button(b_lin_s_a,'line seg')

def _get_coor(event):
    x = event.xdata
    y = event.ydata
    try:
        assert(event.xdata>=0)
        G_var['l_seg_cc'].append((G_var['DATAF'].index[int(beg_setter.val)+int(round(x))],y,int(beg_setter.val)+int(round(x))))
    except:
        return
    if len(G_var['l_seg_cc']) == 1:
        wcur.set_active(False)
        qcur.set_active(True)

    if len(G_var['l_seg_cc']) == 2:
        fd = True
        f_main.canvas.mpl_disconnect(G_var['mc_cid'])
        G_var['mc_cid'] = None
        nl = G_var['DATAF']['Adj_Close'].copy()
        nl[:] = None
        nl[G_var['l_seg_cc'][0][0]] = G_var['l_seg_cc'][0][1]
        nl[G_var['l_seg_cc'][1][0]] = G_var['l_seg_cc'][1][1]
        if G_var['mc_type'] == 'seg':
            nl[G_var['l_seg_cc'][0][0]:G_var['l_seg_cc'][1][0]] = nl[G_var['l_seg_cc'][0][0]:G_var['l_seg_cc'][1][0]].reset_index(drop = True).interpolate()
        elif G_var['mc_type'] == 'line':
            x1 = G_var['l_seg_cc'][0][2]
            x2 = G_var['l_seg_cc'][1][2]
            y1 = G_var['l_seg_cc'][0][1]
            y2 = G_var['l_seg_cc'][1][1]
            k = (y2-y1)/(x2-x1)
            b = y1-k*x1
            nl[:] = pd.Series(range(len(nl)))*k+b
        elif G_var['mc_type'] == 'fibo':
            y1 = G_var['l_seg_cc'][0][1]
            y2 = G_var['l_seg_cc'][1][1]
            x1 = G_var['l_seg_cc'][0][2]
            x2 = G_var['l_seg_cc'][1][2]
            d = y2-y1

            n0,n1,n2,n3,n4,n00 = [nl.copy() for i in range(6)]
            n0[:] = y1 
            n1[:] = y1 + d*0.236
            n2[:] = y1 + d*0.382
            n3[:] = y1 + d*0.5
            n4[:] = y1 + d*0.618
            n00[:] = y2
            G_aid.append(['fibo',['ret' if G_var['fibo']['ret'] else None,n0,n1,n2,n3,n4,n00]])

            if G_var['fibo']['expend']:
                n_expd = nl.copy()
                n_expd[:] = y1 + d*1.618
                G_aid[-1].append(['expd',n_expd])
            if G_var['fibo']['timez']:
                dx = x2-x1
                G_aid[-1].append(['timez', dx,x1])
            if G_var['fibo']['fans']:
                h0 = y1
                h1 = y1 + d*0.382
                h2 = y1 + d*0.5
                h3 = y1 + d*0.618
                h00 = y2

                hs = [h0,h1,h2,h3,h00]
                ks = [(hs[i]-y1)/(x2-x1) for i in range(5)]
                bs = [y1-ks[i]*x1 for i in range(5)]
                nns = [pd.Series(range(len(nl)),index = nl.index)*ks[i]+bs[i] for i in range(5)]
                G_aid[-1].append(['fans',nns])
            if G_var['fibo']['arcs']:
                dist = sqrt((x1-x2)**2+(y1-y2)**2)
                G_aid[-1].append(['arcs',dist,x2,y2])

            fd = False
        elif G_var['mc_type'] == 'patsvr':
            x1 = G_var['l_seg_cc'][0][0]
            x2 = G_var['l_seg_cc'][1][0]            
            G_var['pat_s'] = G_var['DATAF'].loc[x1:x2]

        if fd:
            G_aid.append(['seg',nl,'blue',G_var['l_seg_cc'].copy()])

        
        
        qcur.set_active(False)
    update_plot()
    f_main.show()
    plt.show(block = False)

def als(val):
    if G_var['mc_cid'] is not None:
        f_main.canvas.mpl_disconnect(G_var['mc_cid'])
    G_var['mc_type'] = 'seg'
    wcur.set_active(True)
    G_var['l_seg_cc'] = []
    G_var['mc_cid'] = f_main.canvas.mpl_connect('button_press_event',_get_coor)
    plt.show(block = False)
b_lin_s.on_clicked(als)

b_lin_a = f_main.add_axes([0.02+0.07+0.01+0.07+0.01+0.07+0.01+0.07+0.01+0.07+0.01+0.07+0.01+0.07+0.01,1-0.04,0.07,0.03])
b_lin = Button(b_lin_a,'line')
def ald(val):
    if G_var['mc_cid'] is not None:
        f_main.canvas.mpl_disconnect(G_var['mc_cid'])
    G_var['mc_type'] = 'line'
    wcur.set_active(True)
    G_var['l_seg_cc'] = []
    G_var['mc_cid'] = f_main.canvas.mpl_connect('button_press_event',_get_coor)
    plt.show(block = False)
b_lin.on_clicked(ald)

b_fibo_a = f_main.add_axes([0.02+0.07+0.01+0.07+0.01+0.07+0.01+0.07+0.01+0.07+0.01+0.07+0.01+0.07+0.01+0.07+0.01,1-0.04,0.07,0.03])
b_fibo = Button(b_fibo_a,'fibo')
def afibo(val):
    if G_var['mc_cid'] is not None:
        f_main.canvas.mpl_disconnect(G_var['mc_cid'])
    G_var['mc_type'] = 'fibo'
    wcur.set_active(True)
    G_var['l_seg_cc'] = []
    G_var['mc_cid'] = f_main.canvas.mpl_connect('button_press_event',_get_coor)
    plt.show(block = False)
b_fibo.on_clicked(afibo)

b_pat_a = f_main.add_axes([0.66+0.07+0.01,1-0.04,0.07,0.03])
b_pat = Button(b_pat_a,'pat')
b_pat.on_clicked(lambda val:Pat_Setting_Window() )

b_patsvr_a = f_main.add_axes([0.66+0.07+0.01+0.07+0.01,1-0.04,0.07,0.03])
b_patsvr = Button(b_patsvr_a,'patsvr')
def aps(val):
    if G_var['mc_cid'] is not None:
        f_main.canvas.mpl_disconnect(G_var['mc_cid'])
    G_var['mc_type'] = 'patsvr'
    wcur.set_active(True)
    G_var['l_seg_cc'] = []
    G_var['mc_cid'] = f_main.canvas.mpl_connect('button_press_event',_get_coor)
b_patsvr.on_clicked(aps)



b_clear_a = f_main.add_axes([0.02,1-0.04-0.03-0.01-0.03-0.01,0.05,0.03])
b_clear = Button(b_clear_a,'clear')
def clear_l(val):
    G_aid.clear()
    while len(G_ind_M):
        G_ind_M.pop().remove()
    while len(G_ind_S):
        G_ind_S.pop().remove()
    update_plot()
    update_legend()
b_clear.on_clicked(clear_l)


wcur = Cursor(p_main,useblit=True, color='red', linewidth=2)
qcur = Cursor(p_main,useblit=True, color='blue', linewidth=2)
wcur.set_active(False)
qcur.set_active(False)


def azer1(n):
    def rsr(df):
        window = np.array(range(n))
        return df['Adj_Close'].rolling(window = n,center = False).apply(lambda x:np.corrcoef(x,window)[0][1])
    return rsr
def azer2(n):
    def rsr(df):
        window = np.array([np.exp(i/n) for i in range(n)])
        return df['Adj_Close'].rolling(window = n,center = False).apply(lambda x:np.corrcoef(x,window)[0][1])
    return rsr
def azer3(n):
    def rsr(df):
        return abs(azer2(2*n)(df)-azer1(n)(df))
    return rsr
def azersin(n):
    def rsr(df):
        window = np.array([np.sin(i/n * np.pi ) for i in range(n)])
        return df['Adj_Close'].rolling(window = n,center = False).apply(lambda x:np.corrcoef(x,window)[0][1])
    return rsr
def azer22(n):
    def rsr(df):
        window = np.array([np.exp(i/(n+1)) for i in range(n+1)])
        window = (window - np.roll(window,1))[1:]
        return df.Adj_Close.rolling(window = n+1, center = False).apply(lambda x:np.corrcoef((x-np.roll(x,1))[1:],window)[0][1])
    return rsr

def azern22(n):
    def rsr(df):
        window = np.array([np.exp(-i/(n+1)) for i in range(n+1)])
        window = (window - np.roll(window,1))[1:]
        return df.Adj_Close.rolling(window = n+1, center = False).apply(lambda x:np.corrcoef((x-np.roll(x,1))[1:],window)[0][1])
    return rsr

def azerdsh(n):
    def rsr(df):
        return (azer2(n)(df)/2+azer22(n)(df)/2).rolling(window = n//3,center = False).mean()#.apply(lambda x: 0.8 if x>0.5 else 0.2)
    return rsr


class GSS:
    what = 0

def tgg():
    GSS.d0 = AZ(func = azerdsh(19))


def jkr(**args):
    try:
        G_var['DBGG'] = reload(G_var['DBGG'])
        print(1)
    except Exception as e:
        print(e.__repr__())
        import DBGG
        print(0)
    G_var['DBGG'].dbgg(G_var,**args)

if __name__ == '__main__':
    plt.show()




