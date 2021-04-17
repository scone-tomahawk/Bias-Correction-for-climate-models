import xarray as xr
import numpy as np

from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
import rpy2.robjects as rObj

pandas2ri.activate()

def bcQDM(obs_dat,mod_dat,mod_nf,ratio=True,subs=rObj.NULL,ntau=rObj.NULL):
    _dat = rMBC.QDM(obs_dat,mod_dat,mod_nf,ratio=ratio,subsample=subs,n_tau=ntau)
    return _dat[0],_dat[1]

def apply_QDM(obs_dat,mod_dat,mod_nf,**kwargs):
    start = time.time()
   
    mod_dat,mod_nf,var = mod_dat.rename({'time':'t1'}),mod_nf.rename({'time':'t2'}),list(obs_dat.keys())
    time1,time2 = mod_dat['t1'].values,mod_nf['t2'].values
    mod_dat,mod_nf = mod_dat.dropna('t1'),mod_nf.dropna('t2')
    tRF,tPR = len(mod_dat['t1']),len(mod_nf['t2'])
    
    hist,proj=xr.apply_ufunc(bcQDM,obs_dat,mod_dat,mod_nf,kwargs=kwargs,input_core_dims=[['time'],['t1'],['t2']],
                      output_core_dims=[['t1'],['t2']],vectorize=True,dask='allowed')
    
    if (obs_dat[var[0]].ndim == 1):
        oc = hist.rename({'t1':'time'}).reindex({'time':time1})
        pc = proj.rename({'t2':'time'}).reindex({'time':time2})
    else:
        oc = hist.rename({'t1':'time'}).reindex({'time':time1}).transpose('time','lat','lon')
        pc = proj.rename({'t2':'time'}).reindex({'time':time2}).transpose('time','lat','lon')
    end = time.time()
    print("▮▮▮ Elapsed time in real time :" , time.strftime("%M:%S",time.gmtime(end-start)),"minutes ▮▮▮")
    return oc,pc
