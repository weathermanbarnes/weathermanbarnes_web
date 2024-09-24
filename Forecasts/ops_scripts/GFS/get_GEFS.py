import xarray as xr
import os
from datetime import datetime,timedelta,date
from dateutil.relativedelta import relativedelta
import argparse
import glob
import subprocess
import shutil
import time

#def get_GFS(get_precip=True,get_grid=True,get_surface=True,**kwargs):
get_precip=True
get_grid=True
get_surface=True

# this tells the code which year to run this for, so that multiple versions of the script can be submitted to gadi
parser = argparse.ArgumentParser()
parser.add_argument("date",help="init date",type=str)
parser.add_argument("run",help="init run",type=int)
parser.add_argument("timestep",help="in timestep",type=int)
args = parser.parse_args()
INDATEstr = args.date
RUN = args.run
timestep = args.timestep

RUNstr=str(RUN).zfill(2)
NOW=datetime.strptime(INDATEstr,'%Y%m%d')
FULLDATE=NOW+relativedelta(hours=RUN)
DATE = FULLDATE.strftime("%Y%m%d")
init_date = FULLDATE.strftime("%Y%m%d")
init_time = FULLDATE.strftime("%H%M%S")
init_hour = FULLDATE.strftime("%H")
DATE_LONG = FULLDATE.strftime("%Y-%m-%d %H:%M") #convert date to datestring in format YYYY-MM-DD

# Delete all old forcing files
dir = "/scratch/gb02/mb0427/Website/Forecasts/NOAA/GEFS/"
datadir = dir+"data/"

check_files=True
default_file_size=5e6 #5mb
tstep_length=24
ens_length=30

for ens in range(ens_length+1):
    for t in range(timestep,timestep+tstep_length,tstep_length):
        tlong = str(t).zfill(3)
        enslong = str(ens).zfill(2)
        indt=FULLDATE+relativedelta(hours=t)

        if ens==0:
            ens_name='gec'
        else:
            ens_name='gep'

        url = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/gens/prod/gefs."+init_date+"/"+init_hour+"/atmos/pgrb2ap5/"+ens_name+enslong+".t"+init_hour+"z.pgrb2a.0p50.f"+tlong 
        outname = datadir+init_date+init_time+"-gea"+enslong+"-"+str(t)+"h-gefs.grib2"
        
        p = subprocess.Popen(['wget',url,"-O",outname],stdout=subprocess.PIPE)
        os.waitpid(p.pid,0)

        time.sleep(10)
        
        
        url = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/gens/prod/gefs."+init_date+"/"+init_hour+"/atmos/pgrb2bp5/"+ens_name+enslong+".t"+init_hour+"z.pgrb2b.0p50.f"+tlong 
        outname = datadir+init_date+init_time+"-geb"+enslong+"-"+str(t)+"h-gefs.grib2"
        
        p = subprocess.Popen(['wget',url,"-O",outname],stdout=subprocess.PIPE)
        os.waitpid(p.pid,0)

        time.sleep(10)
        
        if os.path.exists(datadir+init_date+init_time+"-geb"+enslong+"-"+str(t)+"h-gefs.grib2"):
            file_size = os.path.getsize(datadir+init_date+init_time+"-geb"+enslong+"-"+str(t)+"h-gefs.grib2")
            if file_size<=default_file_size:
                check_files=False
        
        print("done")

if check_files:
    check_file_path = dir+"get_GEFS_t"+str(timestep)+".check"
    with open(check_file_path, 'w') as file:
        file.write("File size check passed!")

