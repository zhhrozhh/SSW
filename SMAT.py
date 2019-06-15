import numpy as np
import scipy as sp
import pandas as pd 

def H(n):
	m = np.zeros([n,n])
	for i in range(n-1):
		a = [1/np.sqrt(i+2) for j in range(i+1)]+[-(i+1)/np.sqrt(i+2)] + [0 for j in range(i+2,n)]
		a = np.array(a)
		a = a/np.sqrt(np.dot(a,a))
		m[i] = a
	m[n-1] = np.ones(n)/np.sqrt(n)
	return m

def rer(x):
	l = len(x)
	return np.matmul(H(l),x-x.mean())[:-1]