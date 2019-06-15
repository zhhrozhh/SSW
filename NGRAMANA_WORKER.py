import BPAT
import sys
import matplotlib
import matplotlib.pyplot as plt
import signal
import time
sg = signal.getsignal(signal.SIGINT)


def cc_h(sig,f):
    BPAT.DBGLOG['interrupt_time'] = time.time()
    with open('DLOG','w') as f:
        f.write(str(BPAT.DBGLOG))
    sg(sig,f)

signal.signal(signal.SIGINT, cc_h)

n = int(sys.argv[1])
try:
    BPAT.NGRAM_ANA(n)
except:
    cc_h(2,None)
plt.show()