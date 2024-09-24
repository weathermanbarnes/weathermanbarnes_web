import os
import numpy as np
import xarray as xr
from datetime import datetime,timedelta,date
from dateutil.relativedelta import relativedelta
import gc

from sort_ECMWF_data import *
from plot_map_functions import *

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("date",help="init date",type=str)
parser.add_argument("run",help="init run",type=int)
parser.add_argument("day",help="in timestep",type=int)
parser.add_argument("set_hemisphere",help="init date",type=str)
args = parser.parse_args()
day = args.day
INDATEstr = args.date
RUN = args.run
set_hemisphere = args.set_hemisphere

indatetime=datetime.strptime(INDATEstr,'%Y%m%d')
indatetime=indatetime+relativedelta(hours=RUN)
#home_path='/home/565/mb0427/gdata-w40/Forecasts/IFS/'
home_path = os.getcwd()+"/"
#idx_path='/home/565/mb0427/gdata-w40/Forecasts/temp/'
datapath=home_path+'scratch/IFS/'
if set_hemisphere=='SH':
    names=['SH','Australia','SouthernAfrica','SouthAmerica','IndianOcean']
if set_hemisphere=='NH':    
    names=['NH','NorthAmerica','Europe','NorthAfrica','Asia']

def run_plotting(indatetime,t,names,day):
    tdt=indatetime+relativedelta(hours=t)
    init_date = indatetime.strftime("%Y%m%d")
    init_time = indatetime.strftime("%H%M%S")
    init_hour = indatetime.strftime("%H")
    #if day >= 1:
    #    fn=datapath+"data/"+init_date+init_time+'-'+str(t)+'h-oper-fc.grib2'
    #else:
    #    fn=datapath+"analysis/"+init_date+init_time+'-'+str(t)+'h-oper-fc.grib2'
    fn=datapath+"data/"+init_date+init_time+'-'+str(t)+'h-oper-fc.grib2'
    data=xr.open_dataset(fn,engine='cfgrib')#,backend_kwargs={"indexpath": idx_path+'temp_grib.idx'})
    u10=xr.open_dataset(fn,engine='cfgrib',
                      backend_kwargs={'filter_by_keys': {'shortName': '10u'}})#,"indexpath": idx_path+'temp_grib.idx'})
    v10=xr.open_dataset(fn,engine='cfgrib',
                      backend_kwargs={'filter_by_keys': {'shortName': '10v'}})#,"indexpath": idx_path+'temp_grib.idx'})
    if day >= 1:
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
    indataPV['pwat']=indataUP['pwat']
    indataPV['jet300']=indataUP['jet300']
    indataPV['wMID']=indataUP['wMID']
    indataQpte['wMID']=indataUP['wMID']
    indataRV=get_vort_thick_data(data_pl)

    if day >= 1:
        outfile = datapath+"images/"
    else:
        outfile = datapath+"analysis/"+init_date+init_time+"_"
        
    for name in names:
        plot_upper(outfile,tdt,indatetime,t,indataUP,name=name,model_name='ECMWF-IFS',dpi=600)
        plot_precip6h(outfile,tdt,indatetime,t,indataTP,name=name,model_name='ECMWF-IFS',cbar='on',dpi=600)
        plot_IPV(outfile,tdt,indatetime,t,320,indataPV,name=name,model_name='ECMWF-IFS',dpi=600)
        plot_IPV(outfile,tdt,indatetime,t,330,indataPV,name=name,model_name='ECMWF-IFS',dpi=600)
        plot_IPV(outfile,tdt,indatetime,t,350,indataPV,name=name,model_name='ECMWF-IFS',dpi=600)
        plot_irrotPV(outfile,tdt,indatetime,t,indataPV,name=name,model_name='ECMWF-IFS',dpi=600)
        plot_IVT(outfile,tdt,indatetime,t,indataIVT,name=name,model_name='ECMWF-IFS',dpi=600)
        plot_QvectPTE(outfile,tdt,indatetime,t,indataQpte,name=name,model_name='ECMWF-IFS',dpi=600)

if day >= 1:
    tstep_length=6
    start_hour=day*24-18
    end_hour=day*24
    for t in range(start_hour,end_hour+tstep_length,tstep_length):
        run_plotting(indatetime,t,names,day)
        gc.collect()
        print('done')
        
else:
    tstep_length=6
    start_hour=day*24-18
    end_hour=day*24
    for t in range(start_hour,end_hour+tstep_length,tstep_length):
        analysis_datetime = indatetime + relativedelta(hours=t)
        run_plotting(analysis_datetime,0,names,day)
        gc.collect()
        print('done')

check_file_path = datapath+"plot_ECMWF_d"+str(day)+"_"+set_hemisphere+".check"
with open(check_file_path, 'w') as file:
    file.write("File size check passed!")
