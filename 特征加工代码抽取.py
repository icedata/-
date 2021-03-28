#通用包调取
import os
import pandas as pd
import numpy as np
import itertools
import warnings

#获取线程数
CORE_NUM = int(os.environ['NUMBER_OF_PROCESSORS'])

#数据模块
class FactorData(pd.DataFrame):
    
    @property
    def _constructor(self):
        return FactorData

    @property
    def _constructor_sliced(self):
        return pd.Series
    
    @property
    def fdate(self):
        return self._fdate

    @fdate.setter
    def fdate(self, value):
        self._fdate = value
    
    @property
    def fproduct(self):
        return self._fproduct

    @fproduct.setter
    def fproduct(self, value):
        self._fproduct = value
    
    @property
    def fHEAD_PATH(self):
        return self._fHEAD_PATH

    @fHEAD_PATH.setter
    def fHEAD_PATH(self, value):
        self._fHEAD_PATH = value
    
    def __getitem__(self, key):
        try:
            s = super().__getitem__(key)
        except KeyError:
            s = load(self._fHEAD_PATH+"/tmp pkl/"+self._fproduct+"/"+key+"/"+self._fdate)
            self[key] = s
        return s

#特征构建模块的母类
import inspect
from collections import OrderedDict

class factor_template(object):
    factor_name = ""
    
    params = OrderedDict([
        ("period", np.power(2, range(10,13)))
    ])
    
    def formula(self):
        pass  
    
    def form_info(self):
        return inspect.getsource(self.formula)
    
    def info(self):
        info = ""
        info = info + "factor_name:\n"
        info = info +self.factor_name+"\n"
        info = info +"\n"
        info = info + "formula:\n"
        info = info +self.form_info()+"\n"
        info = info +"\n"
        info = info + "params:\n"
        for key in self.params.keys():
            info = info+"$"+key+":"+str(self.params.get(key))+"\n"
        return info
        
    def __repr__(self):
        return self.info()
    
    def __str__(self):
        return self.info()

#特征构建模块
class foctor_total_trade_imb_period(factor_template):
    factor_name = "total_trade.imb.period"
    
    #1024/2048/4096
    params = OrderedDict([
        ("period", np.power(2, range(10,13)))
    ])
    
    def formula(self, data, period ):
        return vanish_thre(zero_divide(ewma(data["buy.trade"]+data["buy2.trade"]-data["sell.trade"]-data["sell2.trade"], period, adjust=True), 
                           ewma(data["qty"], period, adjust=True)),1).values 

#类指向x1
x1 = foctor_total_trade_imb_period()

#特征构建模块中使用到的函数
def zero_divide(x, y):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = np.divide(x,y)
    if hasattr(y, "__len__"):
        print(res[y==0])
        res[y == 0] = 0
    elif y == 0:
        res = 0
        
    return res

def ewma(x, halflife, init=0, adjust=False):
    init_s = pd.Series(data=init)
    s = init_s.append(x)
    if adjust:
        xx = range(len(x))
        lamb=1 - 0.5**(1 / halflife)
        aa=1-np.power(1-lamb, xx)*(1-lamb)
        bb=s.ewm(halflife=halflife, adjust=False).mean().iloc[1:]
        return bb/aa
    else:
        return s.ewm(halflife=halflife, adjust=False).mean().iloc[1:]

def vanish_thre(x, thre):
    x[np.abs(x)>thre] = 0
    return x
        
#生成因子的通用框架
def build_composite_signal(file_name, signal_list, product, HEAD_PATH):
    keys = list(signal_list.params.keys())

    raw_data = load(file_name)
    data = FactorData(raw_data)
    data.fdate = file_name[-12:]
    data.fproduct = product
    data.fHEAD_PATH = HEAD_PATH
    for cartesian in itertools.product(*signal_list.params.values()):
        signal_name = signal_list.factor_name
        for i in range(len(cartesian)):
            signal_name = signal_name.replace(keys[i], str(cartesian[i])) 
        path = HEAD_PATH+"/tmp pkl new/"+product+"/"+signal_name+"/"+file_name[-12:]
        S = signal_list.formula(data, *cartesian)
        save(S, path)

#并行执行任务
import functools
import dask
from dask import compute, delayed
def parLapply(CORE_NUM, iterable, func, *args, **kwargs):
    with dask.config.set(scheduler='processes', num_workers=CORE_NUM):
        f_par = functools.partial(func, *args, **kwargs)
        result = compute([delayed(f_par)(item) for item in iterable])[0]
        return result

#数据储存路径
HEAD_PATH = "d:/intern"
DATA_PATH = HEAD_PATH + "/pkl tick new/"

#执行读写文件操作
import pandas as pd
import _pickle as cPickle
import gzip

def get_data(product, date):
    data = load(DATA_PATH + product+"/"+date+".pkl")
    return data

def load(path):
    with gzip.open(path, 'rb', compresslevel=1) as file_object:
        raw_data = file_object.read()
    return cPickle.loads(raw_data)

def save(data, path):
    serialized = cPickle.dumps(data)
    with gzip.open(path, 'wb', compresslevel=1) as file_object:
        file_object.write(serialized)

#创建特征储存路径
def create_signal_path(signal_list, product, HEAD_PATH):
    keys = list(signal_list.params.keys())
    for cartesian in itertools.product(*signal_list.params.values()):
        signal_name = signal_list.factor_name
        for i in range(len(cartesian)):
            signal_name = signal_name.replace(keys[i], str(cartesian[i]))
        
        os.makedirs(HEAD_PATH+"/tmp pkl new/"+product+"/"+signal_name, exist_ok=True)
        print(HEAD_PATH+"/tmp pkl new/"+product+"/"+signal_name)

#处理品种
product_list = ['rb', 'hc']

#加工逻辑
if __name__ == "__main__":
    for product in product_list:
        create_signal_path(x1, product, HEAD_PATH);
    for product in product_list:
        file_list = list(map(lambda x: DATA_PATH+product+"/"+x, os.listdir(DATA_PATH + product)))
        parLapply(CORE_NUM, file_list, build_composite_signal,signal_list=x1, product=product, HEAD_PATH=HEAD_PATH)


##查看特征值##
#特征值
factor_data = load(r'D:\intern\tmp pkl new\rb\total_trade.imb.1024\20190819.pkl')
#原始特征值
init_data = load(r'D:\intern\pkl tick new\rb\20190819.pkl')      