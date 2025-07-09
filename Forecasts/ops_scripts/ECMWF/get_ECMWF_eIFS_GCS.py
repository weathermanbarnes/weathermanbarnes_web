import xarray as xr
import os
from datetime import datetime,timedelta,date
from dateutil.relativedelta import relativedelta
import argparse
import glob
import subprocess
import json
import pandas as pd

#def get_GFS(get_precip=True,get_grid=True,get_surface=True,**kwargs):
get_precip=True
get_grid=True
get_surface=True

# this tells the code which year to run this for, so that multiple versions of the script can be submitted to gadi
parser = argparse.ArgumentParser()
parser.add_argument("date",help="init date",type=str)
parser.add_argument("run",help="init run",type=int)
parser.add_argument("timestep",help="init run",type=int)
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
cwd = os.getcwd()
rundir = cwd+"/scratch/eIFS/"
datadir= cwd+"/scratch/eIFS/data/"
os.chdir(datadir)
files = os.listdir(datadir)

## Max and min temp
ltypes=['sfc']#,'sfc','sfc']
llists=['sfc']#,'sfc','sfc']
paras=['mx2tX']#,'10u','10v']

check_files=True
default_file_size=0.1e6 #50mb
if timestep>=168:
    tres=6
else:
    tres=3
#for t in range(timestep,timestep+24,tres):
for t in range(timestep-24+tres,timestep+tres,tres):
    tlong = str(0).zfill(3)
    indt=FULLDATE+relativedelta(hours=t)
    outfile=datadir+init_date+init_time+'-'+str(t)+'h-enfo-ef.index'
    url = 'https://storage.googleapis.com/ecmwf-open-data/'+init_date+'/'+init_hour+'z/ifs/0p25/enfo/'+init_date+init_time+'-'+str(t)+'h-enfo-ef'
    p = subprocess.Popen(['curl','-k',url+'.index','-o',outfile],stdout=subprocess.PIPE)
    os.waitpid(p.pid,0)
    print("done")

    infile=datadir+init_date+init_time+'-'+str(t)+'h-enfo-ef.index'

    data = []
    with open(infile) as f:
        for line in f:
            data.append(json.loads(line))
    data=pd.DataFrame(data)
    data['number'] = data['number'].fillna(0).astype(int)
    data['levelist'] = data['levelist'].fillna('sfc')#.astype(int)

    if t>144:
        paras=[p[:-1]+'6' for p in paras]
    else:
        paras=[p[:-1]+'3' for p in paras]
    for ltype,llist,para in zip(ltypes,llists,paras):
        df = data[data.levtype==ltype][data.levelist==llist][data.param==para]

        for index,row in df.iterrows():
            start_bytes = row._offset
            end_bytes = row._offset + row._length - 1

            outfile=datadir+row.date+row.time+'00-'+row.step+'h-enfo-ef_'
            outfile=outfile+row.param+'_'+row.levelist+'_'+str(row.number)+'.grib2'

            pcurl = subprocess.Popen(['curl',
                                      url+'.grib2',
                                      '--range',str(start_bytes)+'-'+str(end_bytes),
                                      '-o',outfile],stdout=subprocess.PIPE)
            os.waitpid(pcurl.pid,0)

            if os.path.exists(outfile):
                file_size = os.path.getsize(outfile)
                if file_size<=default_file_size:
                    check_files=False


ltypes=['pl','pl','pl','pl','pl','sfc','sfc']#,'sfc','sfc']
llists=['850','850','500','300','300','sfc','sfc']#,'sfc','sfc']
paras=['u','v','gh','u','v','msl','tp']#,'10u','10v']

check_files=True
default_file_size=0.1e6 #50mb
for t in range(timestep,timestep+24,24):
    tlong = str(0).zfill(3)
    indt=FULLDATE+relativedelta(hours=t)
    outfile=init_date+init_time+'-'+str(t)+'h-enfo-ef.index'
    url = 'https://storage.googleapis.com/ecmwf-open-data/'+init_date+'/'+init_hour+'z/ifs/0p25/enfo/'+init_date+init_time+'-'+str(t)+'h-enfo-ef'
    p = subprocess.Popen(['curl','-k',url+'.index','-o',outfile],stdout=subprocess.PIPE)
    os.waitpid(p.pid,0)
    print("done")

    infile=datadir+init_date+init_time+'-'+str(t)+'h-enfo-ef.index'

    data = []
    with open(infile) as f:
        for line in f:
            data.append(json.loads(line))
    data=pd.DataFrame(data)
    data['number'] = data['number'].fillna(0).astype(int)
    data['levelist'] = data['levelist'].fillna('sfc')#.astype(int)

    for ltype,llist,para in zip(ltypes,llists,paras):
        df = data[data.levtype==ltype][data.levelist==llist][data.param==para]
        
        for index,row in df.iterrows():
            start_bytes = row._offset
            end_bytes = row._offset + row._length - 1
        
            outfile=datadir+row.date+row.time+'00-'+row.step+'h-enfo-ef_'
            outfile=outfile+row.param+'_'+row.levelist+'_'+str(row.number)+'.grib2'
            
            pcurl = subprocess.Popen(['curl',
                                      url+'.grib2',
                                      '--range',str(start_bytes)+'-'+str(end_bytes),
                                      '-o',outfile],stdout=subprocess.PIPE)
            os.waitpid(pcurl.pid,0)
            
            if os.path.exists(outfile):
                file_size = os.path.getsize(outfile)
                if file_size<=default_file_size:
                    check_files=False

if check_files:
    check_file_path = rundir+"get_ECMWF_eIFS_t"+str(timestep)+".check"
    with open(check_file_path, 'w') as file:
        file.write("File size check passed!")
