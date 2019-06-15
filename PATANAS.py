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
quandl.ApiConfig.api_key = "rixaXs71r2KgkmPHW9jZ"
with open('scodes','r') as fp:
    scodes= eval(fp.read())

G_ubo = {}
G_pats = []
G_var = {
    'pat_N':5,
    'cur_idx':0,
    'score':[],
    'going':False
}


print('LOADING',end = '')
sys.stdout.flush()
for scode in scodes:
    #G_ubo[scode] = quandl.get('EOD/'+scode)[['Adj_Open','Adj_High','Adj_Low','Adj_Close','Adj_Volume']]
    G_ubo[scode] = tools.get_data(scode)
    print('-',end = '')
    sys.stdout.flush()
print('>DONE')



f_main = plt.figure(figsize = [6,5])
p_main = f_main.add_axes([0.02,0.06,0.82,0.92])
p_main.axes.get_xaxis().set_visible(False)
p_main.axes.get_yaxis().set_visible(False)


b_u_a = f_main.add_axes([0.85,1-0.06,0.07,0.03])
b_d_a = f_main.add_axes([0.85,1-(0.06+0.03+0.01),0.07,0.03])
b_f_a = f_main.add_axes([0.85,1-(0.06+0.03+0.01+0.03+0.01),0.07,0.03])


b_undo_a = f_main.add_axes([0.85,1-(0.06+0.03+0.01+0.03+0.01+0.03+0.01),0.07,0.03])

b_save_a = f_main.add_axes([0.85,1-(0.06+0.03+0.01+0.03+0.01+0.03+0.01+0.03+0.01),0.07,0.03])


b_u = Button(b_u_a,'up')
b_d = Button(b_d_a,'down')
b_f = Button(b_f_a,'flat')
b_undo = Button(b_undo_a,'undo')
b_save = Button(b_save_a,'save')
b_save_a.set_visible(False)

t_dbg_a = f_main.add_axes([0.06,0.01,0.78,0.04])
t_dbg = TextBox(t_dbg_a,'dbg')


def pat_proc(N = 5):
    global G_pats
    G_pats = []
    pat = G_var['DATAF'].iloc[-N:]
    print('finding',end = '')
    sys.stdout.flush()
    for scode in G_ubo:
        data = G_ubo[scode]
        _,r__ = PAT.PAT_SEEKER(pat,data,0.6)
        _,r_ = PAT.S_PAT_SEEKER(pat,data,0.4)
        r = r_*r__
        for idx in r[r!=0].index:
            gp = data.loc[idx:][:40]
            if len(gp) >= 40:
                G_pats.append(gp.reset_index(drop = True))
                print('-',end = '')
                sys.stdout.flush()
    CP = [((x-x.mean())/x.std()).iloc[G_var['pat_N']:] for x in G_pats]
    CP = [x-x.iloc[0] for x in CP]
    D_ = sum(CP)/len(G_pats)
    fn = plt.figure(figsize = [5,5])
    ax = fn.add_axes([0,0,1,1])
    ax.plot(D_.Adj_Close )
    for x in CP:
        ax.plot(x.Adj_Close,alpha = 0.1)
    print('>Done')
    print('{} samples found'.format(len(G_pats)))
    plt.show(block = False)
    #go()



def go():
    p_main.clear()
    if G_var['cur_idx'] == len(G_pats)-1:
        G_var['going'] = False
        print('finished')
        b_save_a.set_visible(True)
        plt.show(block = False)
        G_var['cur_idx'] = 0
        return
    b_save_a.set_visible(False)
    G_var['going'] = True
    D = G_pats[G_var['cur_idx']]
    candlestick2_ohlc(p_main,D.Adj_Open,D.Adj_High,D.Adj_Low,D.Adj_Close,width = 12/40)
    p_main.axvspan(-0.5,G_var['pat_N']+0.5,color = 'green',alpha = 0.3)
    plt.show(block = False)


def tsub(val):
    s = re.match(r'!([a-zA-Z][a-zA-Z0-9_]*)\s*([a-zA-Z][a-zA-Z0-9_]*)?(.*)',t_dbg.text)
    if s:
        cmd = s.groups()[0]
        arg = s.groups()[1]
        oth = s.groups()[2]

        if cmd == 'target':
            #G_var['DATAF'] = quandl.get('EOD/'+arg)[['Adj_Open','Adj_High','Adj_Low','Adj_Close','Adj_Volume']].iloc[-10:]
            G_var['DATAF'] = tools.get_data(arg).iloc[-10:]
            G_var['name'] = arg
            if oth:
                G_var['pat_N'] = int(oth)
                pat_proc(int(oth))
            else:
                pat_proc()
        elif cmd == 'go':
            go()
    else:
        try:
            print(eval(t_dbg.text))
        except:
            exec(t_dbg.text)
    t_dbg.set_val('')
t_dbg.on_submit(tsub)



def u(val):
    if G_var['going']:
        G_var['score'].append(1)
        G_var['cur_idx'] += 1
        go()

def d(val):
    if G_var['going']:
        G_var['score'].append(-1)
        G_var['cur_idx'] += 1
        go()

def f(val):
    if G_var['going']:
        G_var['score'].append(0)
        G_var['cur_idx'] += 1
        go()

def undo(val):
    G_var['score'].pop()
    G_var['cur_idx'] -= 1
    go()

def save(val):
    uu = G_var['score'].count(1)
    dd = G_var['score'].count(-1)
    ff = G_var['score'].count(0)
    pu = uu/len(G_var['score'])
    pd = dd/len(G_var['score'])
    pf = ff/len(G_var['score'])
    pat = G_var['DATAF'].iloc[-G_var['pat_N']:]
    bb = pat.index[0]
    ee = pat.index[-1]
    with open('pat_report/'+G_var['name'] + str(bb)+'-'+str(ee),'w') as fp:
        fp.write(str(bb)+'\n')
        fp.write(str(ee)+'\n')
        fp.write('up trend rate: {}\n'.format(pu))
        fp.write('down trend rate: {}\n'.format(pd))
        fp.write('unknow trend rate: {}\n'.format(pf))

    b_save_a.set_visible(False)
    plt.show(block = False)


b_u.on_clicked(u)
b_d.on_clicked(d)
b_f.on_clicked(f)
b_undo.on_clicked(undo)
b_save.on_clicked(save)

def kpe(event):
    if G_var['going']:
        if event.key == 'up':
            u(0)
        elif event.key == 'down':
            d(0)
        elif event.key == 'right':
            f(0)
        elif event.key == 'left':
            undo(0)
    elif event.key == 'control':
        t_dbg.set_val('!'+t_dbg.text)


f_main.canvas.mpl_connect('key_press_event',kpe)

if __name__ == '__main__':
    plt.show()


