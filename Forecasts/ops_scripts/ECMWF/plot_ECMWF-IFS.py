import sys
import os
import numpy as np
import xarray as xr
from datetime import datetime,timedelta,date
from dateutil.relativedelta import relativedelta
import gc

from sort_ECMWF_data import *

sys.path.append("/home/565/mb0427/gdata-gb02/Website/Forecasts/ops_scripts/")
from plot_map_functions import *

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("date",help="init date",type=str)
parser.add_argument("run",help="init run",type=int)
parser.add_argument("tstep",help="in timestep",type=int)
args = parser.parse_args()
tstep = args.tstep
INDATEstr = args.date
RUN = args.run

indatetime=datetime.strptime(INDATEstr,'%Y%m%d')
indatetime=indatetime+relativedelta(hours=RUN)
home_path = os.getcwd()+"/"
datapath=home_path+'scratch/IFS/'
names=['SH','Australia','SouthernAfrica','SouthAmerica','IndianOcean',
       'NH','NorthAmerica','Europe','NorthAtlantic','Asia']

def get_plot_data(indatetime,t,tstep_length=6):
    tdt=indatetime+relativedelta(hours=t)
    init_date = indatetime.strftime("%Y%m%d")
    init_time = indatetime.strftime("%H%M%S")
    init_hour = indatetime.strftime("%H")
    
    fn=datapath+"data/"+init_date+init_time+'-'+str(t)+'h-oper-fc.grib2'
    data=xr.open_dataset(fn,engine='cfgrib')#,backend_kwargs={"indexpath": idx_path+'temp_grib.idx'})
    u10=xr.open_dataset(fn,engine='cfgrib',
                      backend_kwargs={'filter_by_keys': {'shortName': '10u'}})#,"indexpath": idx_path+'temp_grib.idx'})
    v10=xr.open_dataset(fn,engine='cfgrib',
                      backend_kwargs={'filter_by_keys': {'shortName': '10v'}})#,"indexpath": idx_path+'temp_grib.idx'})
    if t > 0:
        #if t==6:
        tp0=xr.open_dataset(datapath+"data/"+init_date+init_time+'-'+str(t-tstep_length)+'h-oper-fc.grib2',engine='cfgrib',
                          backend_kwargs={'filter_by_keys': {'shortName': 'tp'}})#,"indexpath": idx_path+'temp_grib.idx'})
        tp1=xr.open_dataset(fn,engine='cfgrib',
                          backend_kwargs={'filter_by_keys': {'shortName': 'tp'}})#,"indexpath": idx_path+'temp_grib.idx'})
    else:
        tp0=xr.open_dataset(fn,engine='cfgrib',
                          backend_kwargs={'filter_by_keys': {'shortName': 'tp'}})#,"indexpath": idx_path+'temp_grib.idx'})
        t6dt=indatetime+relativedelta(hours=t)-relativedelta(hours=6)
        t6dt_date = t6dt.strftime("%Y%m%d")
        t6dt_time = t6dt.strftime("%H%M%S")
        tp1=xr.open_dataset(datapath+"data/"+t6dt_date+t6dt_time+'-'+str(6)+'h-oper-fc.grib2',engine='cfgrib',
                          backend_kwargs={'filter_by_keys': {'shortName': 'tp'}})#,"indexpath": idx_path+'temp_grib.idx'})
    
    ###### organise data and plot #####
    indataQpte=get_q_pte(data)
    indataIVT=get_IVT_data(data)
    indataTP=get_precip6h(data,u10,v10)
    indataTP['precip']=(tp1.tp-tp0.tp)*1000
    indataUP=get_upper_data(data)
    indataPV=get_IPV_data(data)
    indataRV=get_vort_thick_data(data)
    
    indata = {**indataUP, **indataTP, **indataPV, **indataIVT, **indataQpte, **indataRV}
    
    return indata

def plot_task(outfile, tdt, analysis_datetime, tt, indata, name='Australia'):
    plot_upper(outfile, tdt, analysis_datetime, tt, indata, name=name, model_name='ECMWF-IFS', dpi=600)
    plot_precip6h(outfile, tdt, analysis_datetime, tt, indata, name=name, model_name='ECMWF-IFS', cbar='on', dpi=600)
    plot_IPV(outfile, tdt, analysis_datetime, tt, 320, indata, name=name, model_name='ECMWF-IFS', dpi=600)
    plot_IPV(outfile, tdt, analysis_datetime, tt, 330, indata, name=name, model_name='ECMWF-IFS', dpi=600)
    plot_IPV(outfile, tdt, analysis_datetime, tt, 350, indata, name=name, model_name='ECMWF-IFS', dpi=600)
    plot_irrotPV(outfile, tdt, analysis_datetime, tt, indata, name=name, model_name='ECMWF-IFS', dpi=600)
    plot_IVT(outfile, tdt, analysis_datetime, tt, indata, name=name, model_name='ECMWF-IFS', dpi=600)
    plot_LowVortPTE(outfile, tdt, analysis_datetime, tt, indata, name=name, model_name='ECMWF-IFS', dpi=600)
    plot_thickness(outfile, tdt, analysis_datetime, tt, indata, name=name, model_name='ECMWF-IFS', dpi=600)
    #plot_DT(outfile, tdt, analysis_datetime, tt, indata, name=name, model_name='ECMWF-IFS', dpi=600)
    #plot_QvectPTE(outfile, tdt, analysis_datetime, tt, indata, name=name, model_name='ECMWF-IFS', dpi=600)
    gc.collect()

if tstep > 0:
    outfile = datapath+"images/"
    
    indata=get_plot_data(indatetime,tstep)
    for name in names:
        plot_task(outfile, indatetime+relativedelta(hours=tstep), indatetime, tstep, indata, name=name)
    gc.collect()
    print('done')
    
else:
    analysis_datetime = indatetime + relativedelta(hours=tstep)
    init_date = analysis_datetime.strftime("%Y%m%d")
    init_time = analysis_datetime.strftime("%H%M%S")
    outfile = datapath+"analysis/"+init_date+init_time+"_"
    
    indata=get_plot_data(analysis_datetime,0)
    for name in names:
        plot_task(outfile, analysis_datetime, analysis_datetime, 0, indata, name=name)
    gc.collect()
    print('done')

check_file_path = datapath+"plot_IFS_t"+str(tstep)+".check"
with open(check_file_path, 'w') as file:
    file.write("File size check passed!")