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
cwd = os.getcwd()
dir = cwd+"/scratch/IFS/"
datadir=dir+"data/"
os.chdir(datadir)
analysisdir=dir+"analysis/"

check_files=True
default_file_size=50e6 #50mb
if day >= 1:
    tstep_length=6
    start_hour=day*24-18
    end_hour=day*24
    
    for t in range(start_hour,end_hour+tstep_length,tstep_length):
        url = 'https://data.ecmwf.int/forecasts/'+init_date+'/'+init_hour+'z/ifs/0p25/oper/'+init_date+init_time+'-'+str(t)+'h-oper-fc.grib2'

        file=init_date+init_time+'-'+str(t)+'h-oper-fc.grib2'
        if os.path.exists(os.path.join(datadir,file)):
            os.remove(os.path.join(datadir,file))
        
        p = subprocess.Popen(['wget',url],stdout=subprocess.PIPE)
        os.waitpid(p.pid,0)
        print("done")

        if os.path.exists(os.path.join(datadir,init_date+init_time+'-'+str(t)+'h-oper-fc.grib2')):
            file_size = os.path.getsize(os.path.join(datadir,init_date+init_time+'-'+str(t)+'h-oper-fc.grib2'))
            if file_size<=default_file_size:
                check_files=False

else:
    tstep_length=6
    start_hour=day*24-18
    end_hour=day*24

    for t in range(start_hour,end_hour+tstep_length,tstep_length):
        analysis_DATE=FULLDATE+relativedelta(hours=t)
        ana_date = analysis_DATE.strftime("%Y%m%d")
        ana_time = analysis_DATE.strftime("%H%M%S")
        ana_hour = analysis_DATE.strftime("%H")

        if ana_hour=='06' or ana_hour=='18':
            stream='scda'
        else:
            stream='oper'
        
        url = 'https://data.ecmwf.int/forecasts/'+ana_date+'/'+ana_hour+'z/ifs/0p25/'+stream+'/'+ana_date+ana_time+'-'+str(0)+'h-'+stream+'-fc.grib2'

        file=ana_date+ana_time+'-'+str(0)+'h-oper-fc.grib2'
        if os.path.exists(os.path.join(datadir,file)):
            os.remove(os.path.join(data,file))
        
        p = subprocess.Popen(['wget',url,'-O',os.path.join(datadir,file)],stdout=subprocess.PIPE)
        os.waitpid(p.pid,0)
        print("done")

        if os.path.exists(os.path.join(datadir,file)):
            file_size = os.path.getsize(os.path.join(datadir,file))
            shutil.copyfile(os.path.join(datadir,file), 
                            os.path.join(analysisdir,file))
            
            if file_size<=default_file_size:
                check_files=False

        ######### Get the 6 hourly data for precip plots #########
        url = 'https://data.ecmwf.int/forecasts/'+ana_date+'/'+ana_hour+'z/ifs/0p25/'+stream+'/'+ana_date+ana_time+'-'+str(6)+'h-'+stream+'-fc.grib2'
    
        file=ana_date+ana_time+'-'+str(6)+'h-oper-fc.grib2'
        if os.path.exists(os.path.join(dir,file)):
            os.remove(os.path.join(dir,file))
        
        p = subprocess.Popen(['wget',url,'-O',os.path.join(datadir,file)],stdout=subprocess.PIPE)
        os.waitpid(p.pid,0)
        print("done")

        if os.path.exists(os.path.join(datadir,file)):
            file_size = os.path.getsize(os.path.join(datadir,file))
            shutil.copyfile(os.path.join(datadir,file), 
                            os.path.join(analysisdir,file))
            
            if file_size<=default_file_size:
                check_files=False

if check_files:
    check_file_path = dir+"get_ECMWF_d"+str(day)+".check"
    with open(check_file_path, 'w') as file:
        file.write("File size check passed!")
