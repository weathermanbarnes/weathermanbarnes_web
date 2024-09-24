import os
import numpy as np
import xarray as xr
from datetime import datetime,timedelta,date
from dateutil.relativedelta import relativedelta
import gc
from collections import ChainMap
import concurrent.futures
import sys
sys.path.append("/home/565/mb0427/gdata-w40/Forecasts/ops_scripts/GFS/")
from sort_GFS_data import *
sys.path.append("/home/565/mb0427/gdata-w40/Forecasts/ops_scripts/GFS/")
from plot_map_functions import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("date",help="init date",type=str)
parser.add_argument("run",help="init run",type=int)
parser.add_argument("day",help="in timestep",type=int)
args = parser.parse_args()
day = args.day
INDATEstr = args.date
RUN = args.run

indatetime=datetime.strptime(INDATEstr,'%Y%m%d')
indatetime=indatetime+relativedelta(hours=RUN)
home_path='/home/565/mb0427/gdata-w40/Forecasts/GFS/'
datapath='/scratch/w40/mb0427/Forecasts/NOAA/GFS/'
#names=['SH','Australia','SouthernAfrica','SouthAmerica','IndianOcean',
#       'NH','NorthAmerica','Europe','Asia','NorthAtlantic']
names=['Australia','SouthernAfrica','SouthAmerica']

def get_plot_data(indatetime,t,day):
    tdt=indatetime+relativedelta(hours=t)
    init_date = indatetime.strftime("%Y%m%d")
    init_time = indatetime.strftime("%H%M%S")
    init_hour = indatetime.strftime("%H")
    if day >= 1:
        fn=datapath+"data/"+init_date+init_time+'-'+str(t)+'h-gfs.grib2'
    else:
        fn=datapath+"analysis/"+init_date+init_time+'-'+str(t)+'h-gfs.grib2'
    data_pl=xr.open_dataset(fn,engine='cfgrib',backend_kwargs={'filter_by_keys': {'typeOfLevel': 'isobaricInhPa'}})
    data_dt=xr.open_dataset(fn,engine='cfgrib',backend_kwargs={'filter_by_keys': {'typeOfLevel': 'potentialVorticity'}})
    
    u10=xr.open_dataset(fn,engine='cfgrib',
                      backend_kwargs={'filter_by_keys': {'shortName': '10u'}})#,"indexpath": idx_path+'temp_grib.idx'})
    v10=xr.open_dataset(fn,engine='cfgrib',
                      backend_kwargs={'filter_by_keys': {'shortName': '10v'}})#,"indexpath": idx_path+'temp_grib.idx'})
    msl=xr.open_dataset(fn,engine='cfgrib',
                      backend_kwargs={'filter_by_keys': {'shortName': 'prmsl'}})#,"indexpath": idx_path+'temp_grib.idx'})
    pwat=xr.open_dataset(fn,engine='cfgrib',
                      backend_kwargs={'filter_by_keys': {'shortName': 'pwat'}})
    
    if day >= 1:
        if t==6:
            tp1=xr.open_dataset(fn,engine='cfgrib',
                              backend_kwargs={'filter_by_keys': {'shortName': 'tp'}})#,"indexpath": idx_path+'temp_grib.idx'})
            tp0=tp1*0
        else:
            tp0=xr.open_dataset(datapath+"data/"+init_date+init_time+'-'+str(t-tstep_length)+'h-gfs.grib2',engine='cfgrib',
                              backend_kwargs={'filter_by_keys': {'shortName': 'tp'}})
            tp1=xr.open_dataset(fn,engine='cfgrib',
                              backend_kwargs={'filter_by_keys': {'shortName': 'tp'}})#,"indexpath": idx_path+'temp_grib.idx'})
    else:
        #tp0=xr.open_dataset(fn,engine='cfgrib',
        #                  backend_kwargs={'filter_by_keys': {'shortName': 'tp'}})#,"indexpath": idx_path+'temp_grib.idx'})
        t6dt=indatetime+relativedelta(hours=t)-relativedelta(hours=6)
        t6dt_date = t6dt.strftime("%Y%m%d")
        t6dt_time = t6dt.strftime("%H%M%S")
        tp1=xr.open_dataset(datapath+"analysis/"+t6dt_date+t6dt_time+'-'+str(6)+'h-gfs.grib2',engine='cfgrib',
                          backend_kwargs={'filter_by_keys': {'shortName': 'tp'}})#,"indexpath": idx_path+'temp_grib.idx'})
        tp0=tp1*0

    indataUP=get_upper_data(data_pl,msl,pwat)
    indataQpte=get_q_pte(data_pl)
    indataIVT=get_IVT_data(data_pl)
    indataTP=get_precip6h(data_pl,u10,v10)
    indataTP['precip']=(tp1.tp-tp0.tp)#*1000
    indataPV=get_IPV_data(data_pl,data_dt)
    indataRV=get_vort_thick_data(data_pl)

    #indata = ChainMap(indataUP, indataTP, indataPV, indataIVT, indataQpte, indataRV)
    indata = {**indataUP, **indataTP, **indataPV, **indataIVT, **indataQpte, **indataRV}
    
    return indata

def plot_task(name):
    plot_upper(outfile, tdt, analysis_datetime, tt, indata, name=name, model_name='GFS', dpi=600)
    plot_precip6h(outfile, tdt, analysis_datetime, tt, indata, name=name, model_name='GFS', cbar='on', dpi=600)
    plot_IPV(outfile, tdt, analysis_datetime, tt, 320, indata, name=name, model_name='GFS', dpi=600)
    plot_IPV(outfile, tdt, analysis_datetime, tt, 330, indata, name=name, model_name='GFS', dpi=600)
    plot_IPV(outfile, tdt, analysis_datetime, tt, 350, indata, name=name, model_name='GFS', dpi=600)
    plot_irrotPV(outfile, tdt, analysis_datetime, tt, indata, name=name, model_name='GFS', dpi=600)
    plot_IVT(outfile, tdt, analysis_datetime, tt, indata, name=name, model_name='GFS', dpi=600)
    plot_QvectPTE(outfile, tdt, analysis_datetime, tt, indata, name=name, model_name='GFS', dpi=600)
    plot_thickness(outfile, tdt, analysis_datetime, tt, indata, name=name, model_name='GFS', dpi=600)
    plot_DT(outfile, tdt, analysis_datetime, tt, indata, name=name, model_name='GFS', dpi=600)
    gc.collect()

if day >= 1:
    tstep_length=6
    start_hour=day*24-18
    end_hour=day*24
    for t in range(start_hour,end_hour+tstep_length,tstep_length):
        tt=t*1
        indata=get_plot_data(indatetime,t,day)
        tdt=indatetime+relativedelta(hours=t)
        analysis_datetime=indatetime+relativedelta(hours=0)
        
        outfile = datapath+"images/"

        # Use ThreadPoolExecutor or ProcessPoolExecutor for parallel execution
        with concurrent.futures.ProcessPoolExecutor() as executor:
            
            # Submit the tasks to be executed in parallel
            futures = [executor.submit(plot_task, name) for name in names]
            
            # Wait for all the futures to complete
            for future in concurrent.futures.as_completed(futures):
                # Get the result (if needed) or check for any exceptions
                try:
                    future.result()
                except Exception as e:
                    print(f"Task failed with exception: {e}")
        
        gc.collect()
        print('done')
        
else:
    tstep_length=6
    start_hour=day*24-18
    end_hour=day*24
    for t in range(start_hour,end_hour+tstep_length,tstep_length):
        tt=0 #Timestep to hand to he plotting function
        analysis_datetime = indatetime + relativedelta(hours=t) #Analysis date
        tdt=analysis_datetime+relativedelta(hours=tt)
        init_date = analysis_datetime.strftime("%Y%m%d") #Analysis date
        init_time = analysis_datetime.strftime("%H%M%S") #Analysis date

        indata=get_plot_data(analysis_datetime,tt,day) #Retrieve the data
        outfile = datapath+"analysis/"+init_date+init_time+"_"

        # Use ThreadPoolExecutor or ProcessPoolExecutor for parallel execution of the plotting
        # Parallelisation amongs the domains
        with concurrent.futures.ProcessPoolExecutor() as executor:
            
            # Submit the tasks to be executed in parallel
            futures = [executor.submit(plot_task, name) for name in names]
            
            # Wait for all the futures to complete
            for future in concurrent.futures.as_completed(futures):
                # Get the result (if needed) or check for any exceptions
                try:
                    future.result()
                except Exception as e:
                    print(f"Task failed with exception: {e}")
        
        gc.collect()
        print('done')

check_file_path = datapath+"plot_GFS_d"+str(day)+".check"
with open(check_file_path, 'w') as file:
    file.write("File size check passed!")
