import xarray as xr
import os
from datetime import datetime,timedelta,date
from dateutil.relativedelta import relativedelta
import argparse
import glob
import subprocess
import shutil

#def get_GFS(get_precip=True,get_grid=True,get_surface=True,**kwargs):
get_precip=True
get_grid=True
get_surface=True

# this tells the code which year to run this for, so that multiple versions of the script can be submitted to gadi
parser = argparse.ArgumentParser()
parser.add_argument("date",help="init date",type=str)
parser.add_argument("run",help="init run",type=int)
parser.add_argument("day",help="in timestep",type=int)
args = parser.parse_args()
INDATEstr = args.date
RUN = args.run
day = args.day

RUNstr=str(RUN).zfill(2)
NOW=datetime.strptime(INDATEstr,'%Y%m%d')
FULLDATE=NOW+relativedelta(hours=RUN)
DATE = FULLDATE.strftime("%Y%m%d")
init_date = FULLDATE.strftime("%Y%m%d")
init_time = FULLDATE.strftime("%H%M%S")
init_hour = FULLDATE.strftime("%H")
DATE_LONG = FULLDATE.strftime("%Y-%m-%d %H:%M") #convert date to datestring in format YYYY-MM-DD

# Delete all old forcing files
dir = "/scratch/gb02/mb0427/Website/Forecasts/NOAA/GFS/data/"
os.chdir(dir)

check_files=True
default_file_size=50e6 #50mb
if day >= 1:
    tstep_length=6
    start_hour=day*24-18
    end_hour=day*24
    
    for t in range(start_hour,end_hour+tstep_length,tstep_length):
        tlong = str(t).zfill(3)
        indt=FULLDATE+relativedelta(hours=t)
        
        url = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod/gfs."+init_date+"/"+init_hour+"/atmos/gfs.t"+init_hour+"z.pgrb2.0p25.f"+tlong 
        outname = init_date+init_time+"-"+str(t)+"h-gfs.grib2"
        
        p = subprocess.Popen(['wget',url,"-O",outname],stdout=subprocess.PIPE)
        os.waitpid(p.pid,0)

        if os.path.exists(os.path.join(dir,init_date+init_time+"-"+str(t)+"h-gfs.grib2")):
            file_size = os.path.getsize(os.path.join(dir,init_date+init_time+"-"+str(t)+"h-gfs.grib2"))
            if file_size<=default_file_size:
                check_files=False
        
        print("done")

else:
    tstep_length=6
    start_hour=day*24-18
    end_hour=day*24
    
    for t in range(start_hour,end_hour+tstep_length,tstep_length):
        analysis_DATE=FULLDATE+relativedelta(hours=t)
        ana_date = analysis_DATE.strftime("%Y%m%d")
        ana_time = analysis_DATE.strftime("%H%M%S")
        ana_hour = analysis_DATE.strftime("%H")
        
        tlong = str(0).zfill(3)
        url = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod/gfs."+ana_date+"/"+ana_hour+"/atmos/gfs.t"+ana_hour+"z.pgrb2.0p25.f"+tlong 
        outname = ana_date+ana_time+"-"+str(0)+"h-gfs.grib2"
        
        p = subprocess.Popen(['wget',url,"-O",outname],stdout=subprocess.PIPE)
        os.waitpid(p.pid,0)

        if os.path.exists(os.path.join(dir,ana_date+ana_time+"-"+str(0)+"h-gfs.grib2")):
            file_size = os.path.getsize(os.path.join(dir,ana_date+ana_time+"-"+str(0)+"h-gfs.grib2"))
            if file_size<=default_file_size:
                check_files=False
        
        tlong = str(6).zfill(3)
        url = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod/gfs."+ana_date+"/"+ana_hour+"/atmos/gfs.t"+ana_hour+"z.pgrb2.0p25.f"+tlong 
        outname = ana_date+ana_time+"-"+str(6)+"h-gfs.grib2"
        
        p = subprocess.Popen(['wget',url,"-O",outname],stdout=subprocess.PIPE)
        os.waitpid(p.pid,0)

        if os.path.exists(os.path.join(dir,ana_date+ana_time+"-"+str(6)+"h-gfs.grib2")):
            file_size = os.path.getsize(os.path.join(dir,ana_date+ana_time+"-"+str(6)+"h-gfs.grib2"))
            if file_size<=default_file_size:
                check_files=False
                
        print("done")

if check_files:
    check_file_path = "/scratch/gb02/mb0427/Website/Forecasts/NOAA/GFS/get_GFS_d"+str(day)+".check"
    with open(check_file_path, 'w') as file:
        file.write("File size check passed!")

