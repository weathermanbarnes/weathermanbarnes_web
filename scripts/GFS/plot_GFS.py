import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '/Users/mbar0087/Documents/Monash Weather Web/functions')
import plot_GFS_functions as functions
import sort_GFS_data as sort_GFS_data
from crop import crop
from datetime import datetime, date
from dateutil.relativedelta import relativedelta
import argparse

# this tells the code which year to run this for, so that multiple versions of the script can be submitted to gadi
# parser = argparse.ArgumentParser()
# parser.add_argument("date",help="init date",type=str)
# parser.add_argument("run",help="init run",type=int)
# args = parser.parse_args()
# INDATEstr = args.date
# RUN = args.run

INDATEstr='20230505'
RUN=12

RUNstr=str(RUN).zfill(2)
NOW=datetime.strptime(INDATEstr,'%Y%m%d')
init_dt1=NOW+relativedelta(hours=RUN)

numhours=192
inpath='data/'
outpath='forecasts/'

fhour=-numhours
i=1

#namelist=['IndianOcean','Australia','PacificOcean','SouthAmerica',
#         'SouthernAfrica','SH']
namelist=['IndianOcean','Australia','SouthAmerica',
         'SouthernAfrica','SH']
#namelist=['SH']

while fhour<=numhours:
    indt=init_dt1+relativedelta(hours=fhour)
    if fhour<0:
        init_dt=indt+relativedelta(hours=0)
    else:
        init_dt=init_dt1+relativedelta(hours=0)
    print(indt)
    
    data=sort_GFS_data.get_PTe_data(inpath,indt)
    for name in namelist:
        functions.plot_PTe(inpath,outpath,indt,init_dt,i,data,name=name)
    
    data=sort_GFS_data.get_DT_data(inpath,indt)
    for name in namelist:
        functions.plot_DT(inpath,outpath,indt,init_dt,i,data,name=name)
    
    PVdata=sort_GFS_data.get_IPV_data(inpath,indt)
    for plev in [320,330,350]:
        for name in namelist:
            functions.plot_IPV(outpath,indt,init_dt,i,str(plev),PVdata,name=name)

    data=sort_GFS_data.get_IVT_data(inpath,indt)
    for name in namelist:
        functions.plot_IVT(inpath,outpath,indt,init_dt,i,data,name=name)
    
    data=sort_GFS_data.get_upper_data(inpath,indt)
    for name in namelist:
        functions.plot_upper(inpath,outpath,indt,init_dt,i,data,name=name)
        
    data['uchi']=PVdata['uchi']
    data['vchi']=PVdata['vchi']
    data['pv_iso_upper']=PVdata['pv_iso_upper']
    
    for name in namelist:
        functions.plot_irrotPV(inpath,outpath,indt,init_dt,i,data,name=name)

    data=sort_GFS_data.get_precip6h(inpath,indt)
    for name in namelist:
        functions.plot_precip6h(inpath,outpath,indt,init_dt,i,data,name=name)
     
    fhour=fhour+6
    i=i+1
