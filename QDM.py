import xarray as xr
import numpy as np

from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
import rpy2.robjects as rObj

pandas2ri.activate()

def bcQDM(obs_h,mod_h,mod_p,ratio=True,subs=rObj.NULL,ntau=rObj.NULL):
    _dat = rMBC.QDM(obs_h,mod_h,mod_p,ratio=ratio,subsample=subs,n_tau=ntau)
    return (_dat[0],_dat[1])

def applyQDM(obs_h,mod_h,mod_p,**kwargs):
    start = time.time()
   
    mod_h,mod_p,var = mod_h.rename({'time':'t1'}),mod_p.rename({'time':'t2'}),list(obs_h.keys())
    time1,time2 = mod_h['t1'].values,mod_p['t2'].values
    mod_h,mod_p = mod_h.dropna('t1'),mod_p.dropna('t2')
     
    hist,proj = xr.apply_ufunc(bcQDM,obs_h,mod_h,mod_p,kwargs=kwargs,
                            input_core_dims=[['time'],['t1'],['t2']],
                            output_core_dims=[['t1'],['t2']],vectorize=True,dask='allowed',
                            output_dtypes=[np.float,np.float])
    
    if (obs_h[var[0]].ndim == 1):
        oc = hist.assign_coords({'t1':time1}).rename({'t1':'time'})
        pc = proj.assign_coords({'t2':time2}).rename({'t2':'time'})
    else:
        oc = hist.assign_coords({'t1':time1}).transpose('t1','lat','lon').rename({'t1':'time'})
        pc = proj.assign_coords({'t2':time2}).transpose('t2','lat','lon').rename({'t2':'time'})
    
    end = time.time()
    print("▮▮▮ Elapsed time in real time :" , time.strftime("%M:%S",time.gmtime(end-start)),"minutes ▮▮▮")
    return oc,pc
