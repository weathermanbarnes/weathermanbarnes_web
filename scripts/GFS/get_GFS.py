import wget
import xarray as xr
import os
from datetime import datetime,timedelta,date
from dateutil.relativedelta import relativedelta
import argparse
import glob

#def get_GFS(get_precip=True,get_grid=True,get_surface=True,**kwargs):
get_precip=True
get_grid=True
get_surface=True

# this tells the code which year to run this for, so that multiple versions of the script can be submitted to gadi
parser = argparse.ArgumentParser()
parser.add_argument("date",help="init date",type=str)
parser.add_argument("run",help="init run",type=int)
args = parser.parse_args()
INDATEstr = args.date
RUN = args.run

RUNstr=str(RUN).zfill(2)
NOW=datetime.strptime(INDATEstr,'%Y%m%d')
FULLDATE=NOW+relativedelta(hours=RUN)
DATE = FULLDATE.strftime("%Y%m%d")
DATE_LONG = FULLDATE.strftime("%Y-%m-%d %H:%M") #convert date to datestring in format YYYY-MM-DD

# Delete all old forcing files
dir = "/g/data/w40/mb0427/MonashWeb/data/"
files = os.listdir(dir)
for file in files:
    if file.startswith("GFS_"):
        os.remove(os.path.join(dir,file))
        
SUBREGION="subregion=&leftlon=-180&rightlon=180&toplat=90&bottomlat=-90&"
#SUBREGION=""
LEVS="lev_5_mb=on&lev_10_mb=on&lev_20_mb=on&lev_50_mb=on&lev_70_mb=on&lev_100_mb=on&lev_150_mb=on&lev_200_mb=on&lev_250_mb=on"
LEVS=LEVS+"&lev_300_mb=on&lev_350_mb=on&lev_400_mb=on&lev_450_mb=on&lev_500_mb=on&lev_550_mb=on&lev_600_mb=on&lev_650_mb=on&lev_700_mb=on"
LEVS=LEVS+"&lev_750_mb=on&lev_750_mb=on&lev_800_mb=on&lev_850_mb=on&lev_900_mb=on&lev_925_mb=on&lev_950_mb=on&lev_975_mb=on&lev_1000_mb=on&"
LEVS=LEVS+"lev_PV%3D-2e-06_(Km'%5E2%2Fkg%2Fs)_surface=on&"
OPT="var_TMP=on&var_UGRD=on&var_VGRD=on&var_HGT=on&var_SPFH=on&var_POT=on&var_VVEL=on&"
OPT=OPT+"var_PRES=on&var_RH=on&"

error_count_max=1000
error_count=0
numhours = 192 #length of forecast in hours
if get_grid:
    for t in range(-numhours,numhours+6,6):
        if t<0:
            while error_count<error_count_max:
              try:
                    tlong = str(0).zfill(3)
                    indt=FULLDATE+relativedelta(hours=t)
                    indtstr = indt.strftime("%Y%m%d_%H")
                    inYMDstr = indt.strftime("%Y%m%d")
                    inHstr = indt.strftime("%H")
                    print(indtstr)
                    dir_path = dir+'GFS_' + indtstr +'.grib2'
                    url = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl?file=gfs.t"+inHstr+"z.pgrb2.0p25.f" + tlong + "&" + LEVS + SUBREGION + OPT + "dir=%2Fgfs.$" + inYMDstr + "%2F"+inHstr+"%2Fatmos"
                    filename = wget.download(url, out=dir_path)
                    
                    break
            
              except IOError:
                print("Error: Download Failed. Retrying..... "+str(error_count)+" times")
                error_count=error_count+1
        else:
            while error_count<error_count_max:
              try:
                    tlong = str(t).zfill(3)
                    indt=FULLDATE+relativedelta(hours=t)
                    indtstr = indt.strftime("%Y%m%d_%H")
                    print(indtstr)
                    dir_path = dir+'GFS_' + indtstr +'.grib2'
                    url = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl?file=gfs.t"+RUNstr+"z.pgrb2.0p25.f" + tlong + "&" + LEVS + SUBREGION + OPT + "dir=%2Fgfs.$" + DATE + "%2F"+RUNstr+"%2Fatmos"
                    filename = wget.download(url, out=dir_path)
                    
                    break
            
              except IOError:
                print("Error: Download Failed. Retrying..... "+str(error_count)+" times")

######################################################################
###### Surface #######################################################
######################################################################
#SUBREGION="subregion=&leftlon=100&rightlon=180&toplat=0&bottomlat=-60&"
LEVS="lev_10_m_above_ground=on&lev_surface=on&lev_mean_sea_level=on&lev_entire_atmosphere_%5C%28considered_as_a_single_layer%5C%29=on&"
OPT="var_MSLET=on&var_UGRD=on&var_VGRD=on&var_PWAT=on&"

error_count=0
if get_surface:
    for t in range(-numhours,numhours+6,6):
        if t<0:
            while error_count<error_count_max:
              try:
                    tlong = str(0).zfill(3)
                    indt=FULLDATE+relativedelta(hours=t)
                    indtstr = indt.strftime("%Y%m%d_%H")
                    inYMDstr = indt.strftime("%Y%m%d")
                    inHstr = indt.strftime("%H")
                    print(indtstr)
                    dir_path = dir+'GFS_surface_' + indtstr +'.grib2'
                    url = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl?file=gfs.t"+inHstr+"z.pgrb2.0p25.f" + tlong + "&" + LEVS + SUBREGION + OPT + "dir=%2Fgfs.$" + inYMDstr + "%2F"+inHstr+"%2Fatmos"
                    filename = wget.download(url, out=dir_path)

                    break

              except IOError:
                print("Error: Download Failed. Retrying..... "+str(error_count)+" times")
                error_count=error_count+1

        else:
            while error_count<error_count_max:
              try:
                    tlong = str(t).zfill(3)
                    indt=FULLDATE+relativedelta(hours=t)
                    indtstr = indt.strftime("%Y%m%d_%H")
                    print(indtstr)
                    dir_path = dir+'GFS_surface_' + indtstr +'.grib2'
                    url = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl?file=gfs.t"+RUNstr+"z.pgrb2.0p25.f" + tlong + "&" + LEVS + SUBREGION + OPT + "dir=%2Fgfs.$" + DATE + "%2F"+RUNstr+"%2Fatmos"
                    filename = wget.download(url, out=dir_path)

                    break

              except IOError:
                print("Error: Download Failed. Retrying..... "+str(error_count)+" times")
                error_count=error_count+1

######################################################################
###### Precipitation (Download and Merge 6 hourly) ###################
######################################################################
if get_precip:
    LEVS="lev_surface=on&"
    OPT="var_APCP=on&"
    error_count=0
    for t in range(-numhours-6,1,1):
        while True:
          try:
                indt=FULLDATE+relativedelta(hours=t)
                indtstr = indt.strftime("%Y%m%d_%H")
                inYMDstr = indt.strftime("%Y%m%d")
                inHstr = indt.strftime("%H")
                inH = int(inHstr)
                if 1 <= inH <= 6:
                    inRUN=0
                    inF=inH
                if 7 <= inH <= 12:
                    inRUN=6
                    inF=inH-6
                if 13 <= inH <= 18:
                    inRUN=12
                    inF=inH-12
                if 19 <= inH <= 23:
                    inRUN=18
                    inF=inH-18
                if inH == 0:
                    inYMDstr = (indt+relativedelta(days=-1)).strftime("%Y%m%d")
                    inRUN=18
                    inF=6
                inRUNstr=str(inRUN).zfill(2)
                tlong = str(inF).zfill(3)
                print(indtstr)
                dir_path = dir+'GFS_precip_' + indtstr +'.grib2'
                url = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl?file=gfs.t"+inRUNstr+"z.pgrb2.0p25.f" + tlong + "&" + LEVS + SUBREGION + OPT + "dir=%2Fgfs.$" + inYMDstr + "%2F"+inRUNstr+"%2Fatmos"
                filename = wget.download(url, out=dir_path)

                break

          except IOError:
            print("Error: Download Failed. Retrying.....")
            error_count=error_count+1

    for t in range(1,120+1,1):
        while True:
          try:
                tlong = str(t).zfill(3)
                indt=FULLDATE+relativedelta(hours=t)
                indtstr = indt.strftime("%Y%m%d_%H")
                print(indtstr)
                dir_path = dir+'GFS_precip_' + indtstr +'.grib2'
                url = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl?file=gfs.t"+RUNstr+"z.pgrb2.0p25.f" + tlong + "&" + LEVS + SUBREGION + OPT + "dir=%2Fgfs.$" + DATE + "%2F"+RUNstr+"%2Fatmos"
                filename = wget.download(url, out=dir_path)

                break

          except IOError:
            print("Error: Download Failed. Retrying.....")
            error_count=error_count+1

    for t in range(123,numhours+3,3):
        while True:
          try:
                tlong = str(t).zfill(3)
                indt=FULLDATE+relativedelta(hours=t)
                indtstr = indt.strftime("%Y%m%d_%H")
                print(indtstr)
                dir_path = dir+'GFS_precip_' + indtstr +'.grib2'
                url = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl?file=gfs.t"+RUNstr+"z.pgrb2.0p25.f" + tlong + "&" + LEVS + SUBREGION + OPT + "dir=%2Fgfs.$" + DATE + "%2F"+RUNstr+"%2Fatmos"
                filename = wget.download(url, out=dir_path)

                break

          except IOError:
            print("Error: Download Failed. Retrying.....")
            error_count=error_count+1

    ######### Hourly to 6 hourly accumulation
    sdt=FULLDATE+relativedelta(hours=-numhours-5)
    edt=FULLDATE+relativedelta(hours=120)
    pdt=sdt
    while pdt<=edt:
        dd=[dir+'GFS_precip_'+(pdt+relativedelta(hours=f)).strftime("%Y%m%d_%H")+'.grib2' for f in range(6)]
        ds = xr.merge([xr.open_dataset(f).expand_dims("valid_time").drop('time').drop('step').drop('surface').rename({'valid_time': 'time'}) for f in dd])
        ds_merge = ds.sum('time')
        #ds_merge['time']=xr.open_dataset(dd[-1]).drop('time').drop('step').drop('surface').rename({'valid_time': 'time'})['time']
        precip6H_timestr=(pdt+relativedelta(hours=5)).strftime("%Y%m%d_%H")
        ds_merge.to_netcdf(path=dir+'GFS_precip6H_'+precip6H_timestr+'.nc')
        
        #print(dd)
        pdt=pdt+relativedelta(hours=6)
     
    ######### 3 hourly to 6 hourly accumulation
    sdt=FULLDATE+relativedelta(hours=123)
    edt=FULLDATE+relativedelta(hours=numhours)
    pdt=sdt
    while pdt<=edt:
        dd=[dir+'GFS_precip_'+(pdt+relativedelta(hours=f)).strftime("%Y%m%d_%H")+'.grib2' for f in range(0,5,3)]
        ds = xr.merge([xr.open_dataset(f).expand_dims("valid_time").drop('time').drop('step').drop('surface').rename({'valid_time': 'time'}) for f in dd])
        ds_merge = ds.sum('time')
        #ds_merge['time']=xr.open_dataset(dd[-1]).drop('time').drop('step').drop('surface').rename({'valid_time': 'time'})['time']
        precip6H_timestr=(pdt+relativedelta(hours=3)).strftime("%Y%m%d_%H")
        ds_merge.to_netcdf(path=dir+'GFS_precip6H_'+precip6H_timestr+'.nc')
        
        #print(dd)
        pdt=pdt+relativedelta(hours=6)


