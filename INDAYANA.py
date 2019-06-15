import BPAT
import pandas as pd
import numpy as np
import scipy as sp

import time
import datetime
import signal
import sys

import tools

import threading
import shutil

FREQ_ONCE = datetime.timedelta(1)

FREQ_H = datetime.timedelta(hours = 1)
FREQ_M = datetime.timedelta(minutes = 1)


sg = signal.getsignal(signal.SIGINT)

db_lock = threading.Lock()

watch_list = []
watch_list_lock = threading.Lock()

bps = {}
bps_lock = threading.Lock()

ngram_res = {}
ngram_res_lock = threading.Lock()


PZ_DATA = {}
PZ_LOCK = threading.Lock()



def now():
    return pd.datetime.now('US/Eastern')

def _now():
    n = now()
    return datetime.timedelta(hours = n.hour,minutes = n.minute,seconds = n.second)

def update_db():
    #est time cost < 30 mins
    db_lock.acquire()
    n = now()
    print('{} updating datebase'.format('{}-{}-{} {}:{}:{}'.format(n.year,n.month,n.day,n.hour,n.minute,n.second)))
    BPAT.update_db()
    n = now()
    print('{} database updated'.format('{}-{}-{} {}:{}:{}'.format(n.year,n.month,n.day,n.hour,n.minute,n.second)))
    db_lock.release()

def backup_db():
    db_lock.acquire()
    n = now()
    shutil.copytree('bpats_db','bpats_db{:04}{:02}{:02}'.format(n.year,n.month,n.day))
    db_lock.release()

def recovery_db(timestr):
    if not os.path.isdir('bpats_db{}'.format(timestr)):
        print('backup not found')
        return
    db_lock.acquire()
    try:
        shutil.rmtree('bpats_db')
    except:
        pass
    shutil.copytree('bpats_db{}'.format(timestr),'bpats_db')
    db_lock.release()




def get_pats(N):
    def gpn_():
        # must be called after update_db
        db_lock.acquire()
        watch_list_lock.acquire()
        bps_lock.acquire()
        bps[N] = {}
        for scode in watch_list:
            bps[N][scode.upper()] = ''.join(BPAT.load(scode.upper())[-N:])
        bps_lock.release()
        watch_list_lock.release()
        db_lock.release()
    gpn_.__name__+=str(N)
    return gpn_

def process_NGRAM(N):
    def psr_():
        #must be called after get_pats
        #est time cost < 10 mins
        db_lock.acquire()
        bps_lock.acquire()
        ngram_res_lock.acquire()
        ngram_res[N] = BPAT.ANA_BPATS(*bps[N].values())
        ngram_res_lock.release()
        bps_lock.release()
        db_lock.release()
    psr_.__name__ += str(N)
    return psr_

def pz():
    
    watch_list_lock.acquire()
    ngram_res_lock.acquire()
    bps_lock.acquire()
    PZ_LOCK.acquire()
    for scode in watch_list:
        _,o,l,h = tools.intraday(scode)
        for k in bps:
            d = BPAT.ANA_RES(ngram_res[k][bps[k][scode]],o,l,h)
            try:
                PZ_DATA[k].append(d.values)
            except:
                PZ_DATA[k] = [d.values]
    PZ_LOCK.release()
    bps_lock.release()
    ngram_res_lock.release()
    watch_list_lock.release()




class SCHEDULE:
    def __init__(self):
        self.schedule = {}
    def add(self,func,start,end,freq=300,args=None):
        assert start.total_seconds>3600
        assert start.total_seconds<23*3600
        assert end.total_seconds<23*3600
        self.schedule[func.__name__] = {
            'start':start,
            'end':end,
            'func':func,
            'freq':freq,
            'args':args,
            'running':None,
            'last_run':None
        }
    def __call__(self):
        time_started = now()
        def handle(sig,f):
            pass
            sg(sig,f)
        signal.signal(SIGINT,handle)
        try:
            while True:
                for k in self.schedule:
                    d = self.schedule[k]
                    if f['running'] and not d['running'].isAlive():
                        d['running'] = None
                    if(
                        (_now()>=d['start']) and
                        (not d['running']) and
                        (_now()<d['end']) and
                        (
                            (not(d['last_run'])) or
                            (_now()-d['last_run']>d['freq'])
                        )
                    ):
                        d['last_run'] = _now()
                        d['running'] = threading.Thread(target = d['func'],args = d['args'])
                        d['running'].start()

                    
                    if (not now().hour)and(now()-time_started>datetime.timedelta(1)):
                        for kk in self.schedule:
                            if self.schedule[kk]['running']:
                                self.schedule[kk]['running'].join()
                                self.schedule[kk]['running'] = None
                                self.schedule[kk]['last_run'] = None
                        time_started = now()
                        time.sleep(1800)
        except:
            handle(2,None)




class MAIN_FIG:
    def __init__(self,targets = []):
        self.fig = plt.figure(figsize = [5,6])
        self.p_main = self.fig.add_axes([0.02,1-0.02-5*0.96/6,0.96,5*0.96/6])
        self.time = datetime.datetime.now('US/Eastern')
        self.yday = self.time - datetime.timedelta(1)

if __name__ == '__main__':
    import matplotlib
    matplotlib.use('tkagg')
    print(matplotlib.get_backend())
    import matplotlib.pyplot as plt
    MAIN_FIG()
    plt.show()
