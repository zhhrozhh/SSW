import iexfinance
from iexfinance.stocks import get_historical_data
import pandas as pd

def get_data(scode,start = None,end = None,last = None):
    scode = scode.replace('_','.')
    if last:
        return get_last(scode,last)
    data = pd.DataFrame(get_historical_data(scode,start = start,end = end,token = "pk_4d900afee23b484ba28610b49b795dcf")).T
    if not len(data):
        data = pd.DataFrame(columns = ['Adj_Close','Adj_High','Adj_Low','Adj_Open','Adj_Volume'])
    data.columns = ['Adj_Close','Adj_High','Adj_Low','Adj_Open','Adj_Volume']
    data.index = data.index.to_series().apply(lambda x:pd.datetime(*map(int,x.split('-'))))
    return data


def get_last(scode,N):
    data = pd.DataFrame(iexfinance.stocks.Stock(scode).get_historical_prices(chartLast = N))[['date','close','high','low','open','volume']]
    data.index = pd.Series(data.date.values).apply(lambda x:pd.datetime(*map(int,x.split('-'))))
    data = data.drop(['date'],axis = 1)
    data.columns = ['Adj_Close','Adj_High','Adj_Low','Adj_Open','Adj_Volume']
    return data


def intraday(scode):
    pass  