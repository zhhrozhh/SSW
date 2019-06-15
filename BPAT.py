import pandas as pd
import numpy as np
import scipy as sp
import quandl
import h5py
import os
import time
import iexfinance

#matplotlib.use('tkagg')

import signal
import sys
import tools
import pytz
from datetime import timedelta
from matplotlib.widgets import Slider


with open('KeyR','r') as f:
    quandl.ApiConfig.api_key = f.read()
scodes = []
with open('scodes','r') as f:
    scodes = eval(f.read())

G_X = {}


zim = {}



def ANA_BPATS(*bpats):
    L = 0
    D = {}
    s = ''
    s0 = ''
    for bpat in bpats:
        L = len(bpat)//4
        D[bpat] = {}
    ii = 0
    t0 = time.time()
    for scode in scodes:
        d = load(scode)
        for i in range(len(d)-L-1):
            dd = d[i:i+L]
            death = False
            for t in dd:
                if(int(t,16)>64):
                    death = True
                    break
            if death:
                continue
            k = ''.join(dd)
            if k in D:
                try:
                    D[k][d[i+L]] += 1
                except:
                    D[k][d[i+L]] = 1
        ii += 1
        s0 = s
        s = '{}/{}'.format(ii,len(scodes))
        t = time.time()
        if t-t0>0.5:
            for i in range(len(s0)):
                print(u'\b',end = '',flush = True)
            print(s,end = '',flush = True)
            t0 = time.time()
    return D


def ANA_RES(resd,o,cl,ch):
    res = []
    for b,p in resd.items():
        bb = int(b,16)
        po = bb%8 #o-l
        pc = (bb-po)/8 
        H_ = cl + 8*(o-cl)/(po+1)
        L_ = ch - 8*(ch-o)/(8-po)
        C_x = cl + (1+pc)*(o-cl)/(po+1)
        C_y = ch - (1+pc)*(ch-o)/(8-po)
        lum = [H_,L_,C_x,C_y,p]
        res.append(lum)

    return pd.DataFrame(res,columns = ['H','L','Cx','Cy','p'])


    


def load_to_f():
    i=0
    for scode in scodes:
        if os.path.isfile('bpats_db/'+scode):
            i+=1
            continue
        data = tools.get_data(scode)
        o = data.Adj_Open
        c = data.Adj_Close
        h = data.Adj_High
        l = data.Adj_Low
        v = data.Adj_Volume
        save(BPAT(o,c,h,l,v),scode)
        i+=1
        print('{}/{}'.format(i,len(scodes)),end = ' ')

def update_db():
    today = pd.datetime.now(pytz.timezone('US/Eastern'))
    last = None
    with open('bpats_db/LASTUD','r') as f:
        last = pd.datetime(*map(int,f.read().split('-'))) + timedelta(1)
    if today.hour < 9 and today.hour > 4:
        today = today - timedelta(1)
    else:
        today = today - timedelta(2)
    today = pd.datetime(today.year,today.month,today.day)
    if today - last < timedelta(1):
        return -1
    for scode in scodes:
        data = None
        print(scode,end = ' ')
        try:
            data = tools.get_data(scode,start = last,end = today)
        except iexfinance.utils.exceptions.IEXSymbolError:
            continue
        except:
            print(
                'update {} failed, {} -> {}'.format(
                    scode,
                    '{}-{}-{}'.format(
                        last.year,
                        last.month,
                        last.day
                    ),
                    '{}-{}-{}'.format(
                        today.year,
                        today.month,
                        today.day
                    )
                )
            )
            continue
        o = data.Adj_Open
        c = data.Adj_Close
        h = data.Adj_High
        l = data.Adj_Low
        v = data.Adj_Volume
        zm = BPAT(o,c,h,l,v)
        with open('bpats_db/'+scode,'a') as f:
            for b in zm:
                f.write(b+' ')
    with open('bpats_db/LASTUD','w') as f:
        f.write('{}-{}-{}'.format(today.year,today.month,today.day))





def fix_f(scode):
    x = load(scode)
    def m(y):
        if int(y,16)<64:
            return y
        else:
            return '0x99'
    x_ = [m(y) for y in x ]
    save(x_,scode)

DBGLOG = {'ctfCOUNT':0,'ATT':[]}

def NGRAM_ANA(n,reset = False):
    import matplotlib
    import matplotlib.pyplot as plt
    fg = 'w' if reset else 'a'
    print('-',end = '')
    with h5py.File('NGRAM.h5',fg,libver = 'latest') as F:
        print('-',end = '')
        if 'D{}'.format(n) in F:
            del F['D{}'.format(n)]
        print('-',end = '')

        D = F.create_group('D{}'.format(n))
        print('-',end = '')
        iz = 0
        tst = 0

        tzs = [0 for i in range(100)]
        matz = [0 for i in range(100)]
        mitz = [0 for i in range(100)]
        print('-',end = '')
        fig = plt.figure(figsize = [4,4.5])
        ax0 = fig.add_axes([0.1,2/3,1-1/7,1/3-1/5])
        ax1 = fig.add_axes([0.1,0.02,1-1/7,2/3-1/5])
        
        ax0.plot([0,1],[0,0],alpha = 0)

        plt.ion()
        fig.canvas.draw()

        tc = time.time()
        D_CACHE = {}



        def ctf(D_CACHE):
            print('*')
            DBGLOG['ctfCOUNT'] += 1
            DBGLOG['inCTF'] = True
            ii = 0
            L = len(D_CACHE)
            for k in D_CACHE:
                
                if not ii%5000:
                    ax0.axvspan(0,ii/L,color = 'green',alpha = 0.5)

                if k in D:
                    D[k][()] += D_CACHE[k][1]
                else:
                    D[k] = D_CACHE[k][1]
                ii += 1
            D_CACHE.clear()
            DBGLOG['inCTF'] = False

        def ctf_K(D_CACHE,k):

            tss = time.time()
            DBGLOG['ctfCOUNT'] += 1
            DBGLOG['inCTF'] = True
            if k in D:
                D[k][()] += D_CACHE[k][1]
            else:
                D[k] = D_CACHE[k][1]
            D_CACHE.pop(k)
            DBGLOG['inCTF'] = False
            DBGLOG['ATT'].append(time.time()-tss)
        print('->')
        for scode in scodes:
            DBGLOG['scode'] = scode
            
            DBGLOG['since'] = time.time()

            DBGLOG['pg'] = iz/len(scodes)
            d = load(scode)
            
            kts = []
            if len(d)<n:
                continue
            for i in range(len(d)-n):

                t0 = time.time()
                if i>=5000 and not i%500:
                    fig.suptitle('{0}/{1},{2:.2f}%'.format(i,len(d)-n,100*iz/len(scodes)), fontsize=16)

                    plt.show(block = False)
                DBGLOG['step'] = i
                DBGLOG['steps'] = len(d)-n
                DBGLOG['step_since'] = time.time()
                death = False
                for x in d[i:i+n]:
                    if int(x,16)>63:
                        death = True
                        break
                if death:
                    continue

                tst += 1

                k = bytes([int(x,16)+48 for x in d[i:i+n]])

                DBGLOG['current_key'] = (k,d[i:i+n])
                #LRU 2[2,3]
                if k not in D_CACHE:
                    D_CACHE[k] = (1,tst,tst)

                    DBGLOG['k_in_D'] = (False,k)
                else:
                    D_CACHE[k] = (D_CACHE[k][0]+1,tst,D_CACHE[k][1])
                    DBGLOG['k_in_D'] = (True,k)

                if len(D_CACHE) >= 2**19:
                    dkk = max(D_CACHE,key = lambda x:D_CACHE[x][1]-D_CACHE[x][2])
                    ctf_K(D_CACHE,dkk)

                kts.append(time.time()-t0)
            if not len(kts):
                continue
            iz += 1
            tzs = tzs[1:]+[np.mean(kts)]
            matz = matz[1:]+[max(kts)]
            mitz = mitz[1:]+[min(kts)]
            dt = time.time()-tc

            if dt >=0.5:
                ax0.clear()
                ax1.clear()
                ax0.plot([0,1],[0,0],alpha = 0)
                ax0.axvspan(0,iz/len(scodes),color = 'blue',alpha = 0.6)
                ax1.plot(tzs,color = 'blue')
                ax1.plot(matz,color = 'green')
                ax1.plot(mitz,color = 'red')
                fig.canvas.draw()
                tc = time.time()
        ctf(D_CACHE)




def BPAT(o,c,h,l,v):

    r0 = pd.DataFrame()

    oo = o-l
    cc = c-l
    hh = h-l

    z = 1/8
    def zm(x):
        for i in range(8):
            if x <= (i+1)*z:
                return i
        return 17
    po = (oo/hh).apply(zm)
    pc = (cc/hh).apply(zm)

    return (po+pc*8).apply(lambda x:'{0:#0{1}x}'.format(int(x),4)).values

def translate(b):
    po = b%8
    pc = (b-po)/8
    for i in range(8):
        print('{}|{}'.format('-'if po == 7-i else ' ','-'if pc == 7-i else ' '))
def translates(bs):
    n = int(len(bs)/4)
    bss = [int(x,16) for x in [bs[i:i+4] for i in range(0,len(bs),4) ]]
    binfo = [(x%8,(x-x%8)/8) for x in bss]
    ll = []
    hh = []
    cc = []
    oo = []
    for o,c in binfo:
        oz,cz = 0,0
        lz,hz = 0,0
        if not len(cc):
            oz = o
        else:
            oz = cc[-1]
        cz = oz + c-o
        lz = cz - c
        hz = lz + 8
        oo.append(oz)
        cc.append(cz)
        ll.append(lz)
        hh.append(hz)
    d = pd.DataFrame([oo,cc,ll,hh],index = ['o','c','l','h'])
    if d.loc['l'].min() < 0:
        d = d-d.loc['l'].min()
    h = int(d.loc['h'].max())
    w = 3*n
    mm = np.array([[' 'for i in range(h)]for j in range(w)])
    for i in range(n):
        mm[i*3][int(d.loc['o'][i])] = '-'
        for j in range(8):
            mm[i*3+1][int(d.loc['l'][i])+j] = '|'
        mm[i*3+2][int(d.loc['c'][i])] = '-'
    mm = mm[:][::-1]
    for i in range(h):
        for j in range(w):
            print(mm[j][i],end = '')
        print()


def pprint(bs):
    n = int(len(bs)/4)
    bss = [int(x,16) for x in [bs[i:i+4] for i in range(0,len(bs),4) ]]
    binfo = [(x%8,(x-x%8)/8) for x in bss]
    w = 4*n


    for i in range(8):
        for j in range(w):
            idx = j//4
            if j%4 == 3:
                print(' ',end = '')
            elif j%4 == 2:
                print('-' if(binfo[idx][1]==(7-i)) else ' ',end = '')
            elif j%4 == 1:
                print('|',end = '')
            elif not j%4:
                print('-' if(binfo[idx][0]==(7-i)) else ' ',end = '')
        print()

def save(bpat,fn):
    with open('bpats_db/'+fn,'w') as f: 
        for b in bpat:
            f.write(b+' ')


def load(fn):
    with open('bpats_db/'+fn,'r') as f:
        return f.read().split()





