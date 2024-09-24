#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import requests
import os
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import numpy as np
import glob
import xarray as xr
import cartopy.crs as ccrs
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, BoundaryNorm
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import cmocean
from google.cloud import storage
import os

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/565/mb0427/gadi_web_bucket_access.json"
degrees = u'\N{DEGREE SIGN}'
bucket_name = "www.weathermanbarnes.com"  # Replace with your bucket name

def generate_date_times(start_time, end_time, interval_hours=3):
    date_times = []
    current_time = start_time

    while current_time <= end_time:
        date_times.append(current_time)
        current_time += timedelta(hours=interval_hours)

    return date_times

def upload_to_bucket(bucket_name, source_file_name, destination_blob_name):
    """
    Uploads a file to the bucket.
    
    Args:
    bucket_name (str): Name of the bucket.
    source_file_name (str): Path to the file to upload.
    destination_blob_name (str): Name of the blob in the bucket.
    """
    # Initialize a storage client
    storage_client = storage.Client()
    
    # Get the bucket
    bucket = storage_client.bucket(bucket_name)
    
    # Create a blob object
    blob = bucket.blob(destination_blob_name)
    
    # Upload the file
    blob.upload_from_filename(source_file_name)
    
    print(f"File {source_file_name} uploaded to {destination_blob_name}.")

def set_cache_control(bucket_name, blob_name, cache_control_value):
    """Set the Cache-Control metadata for a GCS blob."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    blob.cache_control = cache_control_value
    blob.patch()

    print(f"Cache-Control for {blob_name} set to {cache_control_value}.")

outpath='/scratch/w40/mb0427/Forecasts/obs/'
numdays=7
start_date=datetime.today()

################################################################################################################################
################################################################################################################################
################################################################################################################################

# Get data (spin-up new stations)
for id in ['IDV60901.95867','IDV60901.94870','IDT60901.94970','IDV60901.94864','IDQ60801.94287']:
    id_split = id.split('.')
    bom_url = 'http://reg.bom.gov.au/fwo/'+id_split[0]+'/'+id+'.axf'
    bom_csv = outpath+id+".csv"
    r = requests.get(bom_url)
    f = open(bom_csv, "w")
    f.write(r.text)
    f.close()

    data = pd.read_csv(bom_csv,skiprows=19)
    data = data[data['sort_order'] != '[$]']
    data['local_datetimes']=[datetime.strptime(str(int(dt)),'%Y%m%d%H%M%S') for dt in data['local_date_time_full[80]']]

    for dt in list(data['local_datetimes'].dt.date.unique())[0:3]:
        dt_data = data[data['local_datetimes'].dt.date == pd.to_datetime(dt).date()]
        dt_data.to_csv(outpath+id+'_'+dt.strftime('%Y%m%d')+'.csv')

################################################################################################################################
#IDV60901.95867
# Get data and plot (operations)
for id in ['IDV60901.95936', 'IDS60901.94648', 'IDS60901.94672', 'IDV60901.94866',
          'IDT60901.94619','IDN60903.94926','IDN60901.94767','IDQ60901.94576',
          #'IDV60901.95867',
          'IDV60901.94870','IDT60901.94970','IDV60901.94864','IDQ60801.94287',
          'IDD60901.94120','IDW60901.94610']:
    id_split = id.split('.')
    bom_url = 'http://reg.bom.gov.au/fwo/'+id_split[0]+'/'+id+'.axf'
    bom_csv = outpath+id+".csv"
    r = requests.get(bom_url)
    f = open(bom_csv, "w")
    f.write(r.text)
    f.close()

    data = pd.read_csv(bom_csv,skiprows=19)
    data = data[data['sort_order'] != '[$]']
    data['local_datetimes']=[datetime.strptime(str(int(dt)),'%Y%m%d%H%M%S') for dt in data['local_date_time_full[80]']]

    for dt in list(data['local_datetimes'].dt.date.unique())[0:3]:
        dt_data = data[data['local_datetimes'].dt.date == pd.to_datetime(dt).date()]
        dt_data.to_csv(outpath+id+'_'+dt.strftime('%Y%m%d')+'.csv')
    
    dt_list=generate_date_times(start_date-relativedelta(days=numdays),
                                   start_date,
                                   interval_hours=24)
    
    first=True
    for dt in dt_list[::-1]:
        
        bom_dt_csv=outpath+id+"_"+dt.strftime('%Y%m%d')+".csv"
        if first:
            data = pd.read_csv(bom_dt_csv)
            first=False
        else:
            data = pd.concat([data, pd.read_csv(bom_dt_csv)], ignore_index=True)#, on='local_datetimes')
    data=data.sort_values(by='local_datetimes', ascending=False) 
    
    header_df=pd.read_csv(bom_csv,skiprows=0,nrows=16)
    #data = data[data['sort_order'] != '[$]']
    
    header={}
    header['name']=str(header_df.iloc[9].name.split('name[80]=')[1]).split('"')[1]
    header['state']=header_df.iloc[13].name.split('state[80]=')[1].split('"')[1]
    
    for k in data.keys():
        #print(k.split('[80]')[0])
        data=data.rename(columns={k: k.split('[80]')[0]})
    
    data['local_datetimes']=[datetime.strptime(str(int(dt)),'%Y%m%d%H%M%S') for dt in data['local_date_time_full']]
    data['rain_accum'] = [float(d) if d != '-' else 0.0 for d in data['rain_trace']]

    data_axis_24hrly=pd.DataFrame()
    #data_axis_24hrly['local_datetimes_24hrly_raw']= [dt for dt in data['local_datetimes'].to_list() if dt.hour % 24 == 0 and dt.minute ==0]
    data_axis_24hrly['local_datetimes_24hrly']=generate_date_times(start_date.replace(hour=0, minute=0, second=0)-relativedelta(days=numdays),
                                                                   start_date.replace(hour=0, minute=0, second=0),interval_hours=24)
    #data_axis_24hrly['axis_dts_raw']= [dt.strftime('%m/%d\n%H:%M') for dt in data_axis_24hrly['local_datetimes_24hrly_raw']]
    data_axis_24hrly['axis_dts']= [dt.strftime('%a\n%m/%d\n%H:%M') for dt in data_axis_24hrly['local_datetimes_24hrly']]
    #print(data_axis_24hrly['local_datetimes_24hrly'])

    data_axis_12hrly=pd.DataFrame()
    data_axis_12hrly['local_datetimes_12hrly']= [dt for dt in data['local_datetimes'].to_list() if dt.hour % 12 == 0 and dt.minute ==0]
    data_axis_12hrly['axis_dts']= [dt.strftime('%m/%d\n%H:%M') for dt in data_axis_12hrly['local_datetimes_12hrly']]

    data_axis_6hrly=pd.DataFrame()
    data_axis_6hrly['local_datetimes_6hrly']= [dt for dt in data['local_datetimes'].to_list() if dt.hour % 6 == 0 and dt.minute ==0]
    data_axis_6hrly['axis_dts']= [dt.strftime('%m/%d\n%H:%M') for dt in data_axis_6hrly['local_datetimes_6hrly']]
    
    data_3hrly=pd.DataFrame()
    data_3hrly['local_datetimes_3hrly']= [dt for dt in data['local_datetimes'].to_list() if dt.hour % 3 == 0 and dt.minute ==0]
    selected_dt_idxs_3hrly= [i for i, dt in enumerate(data['local_datetimes']) if dt.hour % 3 == 0 and dt.minute ==0]
    data_3hrly['rain_accum_3hrly']=[data['rain_accum'][d] for d in selected_dt_idxs_3hrly]
    
    data_1hrly=pd.DataFrame()
    data_1hrly['local_datetimes_1hrly']= [dt for dt in data['local_datetimes'] if dt.hour % 1 == 0 and dt.minute ==0]
    selected_dt_idxs_1hrly= [i for i, dt in enumerate(data['local_datetimes']) if dt.hour % 1 == 0 and dt.minute ==0]
    data_1hrly['rain_accum_1hrly']=[data['rain_accum'][d] for d in selected_dt_idxs_1hrly]
    
    data_09h00=pd.DataFrame()
    data_09h00['local_datetimes_09h00']= [dt for dt in data['local_datetimes'] if dt.hour == 9 and dt.minute ==0]
    selected_dt_idxs_09h00= [i for i, dt in enumerate(data['local_datetimes']) if dt.hour == 9 and dt.minute ==0]
    data_09h00['rain_accum_09h00']=[data['rain_accum'][d] for d in selected_dt_idxs_09h00]
    
    diffs=[]
    for i in range(len(data_3hrly['rain_accum_3hrly']) - 1):
        diff=data_3hrly['rain_accum_3hrly'][i] - data_3hrly['rain_accum_3hrly'][i+1]
        if diff<0:
            diff=data_3hrly['rain_accum_3hrly'][i]
        diffs.append(diff)
    diffs.append(0.0)
    data_3hrly['rain_accum_3hrly']=diffs

    data['dewpt'][data['dewpt']==-9999.0]=np.nan
    data['air_temp'][data['air_temp']==-9999.0]=np.nan
    date_times = pd.to_datetime(data['local_datetimes'])
    df = pd.DataFrame({'datetime': date_times, 'air_temp': data['air_temp']})
    tmax = df.loc[df.groupby(df['datetime'].dt.date)['air_temp'].idxmax()]
    tmax_dt=[ts.to_pydatetime() for ts in tmax.datetime.to_list()]
    tmax=tmax.air_temp.to_list()
    tmin = df.loc[df.groupby(df['datetime'].dt.date)['air_temp'].idxmin()]
    tmin_dt=[ts.to_pydatetime() for ts in tmin.datetime.to_list()]
    tmin=tmin.air_temp.to_list()
    
    nan_list=np.empty(len(data['wind_dir']))
    nan_list[:]=np.nan
    
    wind_dir_val=list(nan_list)
    
    for i,d in enumerate(data['wind_dir']):
        if d=='N':
            dval=0
        elif d=='NNE':
            dval=22.5
        elif d=='NE':
            dval=45
        elif d=='ENE':
            dval=67.5
        elif d=='E':
            dval=90
        elif d=='ESE':
            dval=112.5
        elif d=='SE':
            dval=135
        elif d=='SSE':
            dval=157.5
        elif d=='S':
            dval=180
        elif d=='SSW':
            dval=202.5
        elif d=='SW':
            dval=225
        elif d=='WSW':
            dval=247.5
        elif d=='W':
            dval=270
        elif d=='WNW':
            dval=292.5
        elif d=='NW':
            dval=315
        elif d=='NNW':
            dval=337.5
        else:
            dval=np.nan
        wind_dir_val[i]=dval
        
    data['wind_dir_val']=wind_dir_val
    
    fig,(ax0,ax1,ax2,ax3,ax4)=plt.subplots(5,1, figsize=(10,10))
    
    ######################## Plot temperature ########################
    xmin=np.floor(min([i for i in data['dewpt'] if i is not None])/5)*5
    xmax=np.ceil(max([i for i in data['air_temp'] if i is not None])/5)*5
    
    ax1.plot(data['local_datetimes'],data['dewpt'],color='grey',alpha=0.6)
    ax1.set_ylabel('Dewpoint Temp. ['+degrees+'C]',color='grey')
    ax1.set_ylim((xmin,xmax))
    
    ax1v2 = ax1.twinx()
    ax1v2.plot(data['local_datetimes'],data['air_temp'],color='red')
    ax1v2.tick_params(axis='y', labelcolor='red')
    ax1v2.set_ylabel('Temperature ['+degrees+'C]', color='red')
    ax1v2.set_ylim((xmin,xmax))
    
    for i,t in enumerate(tmax):
        ax1v2.text(tmax_dt[i], t, f'{t:.1f}', 
                 ha='center', va='bottom', fontweight='bold', zorder=1e2)
    for i,t in enumerate(tmin):
        ax1v2.text(tmin_dt[i], t, f'{t:.1f}', 
                 ha='center', va='top', fontweight='bold', zorder=1e2)
    
    ######################## Plot Press/Dir ########################
    ax2.plot(data['local_datetimes'],data['press_msl'],color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    ax2.set_ylabel('MSLP [hPa]', color='green')
    
    ax2v2 = ax2.twinx()
    ax2v2.plot(data['local_datetimes'],data['wind_dir_val'],color='blue')
    ax2v2.set_yticks(np.arange(0,382.5,22.5))
    ax2v2.set_ylim((0,360))
    ax2v2.set_yticklabels(['N','','','',
                           'E','','','',
                           'S','','','',
                           'W','','','','N'])
    ax2v2.tick_params(axis='y', labelcolor='blue')
    ax2v2.set_ylabel('Wind Direction', color='blue')
    
    ######################## Plot Wind Speed ########################
    xmax=np.ceil(max([i for i in data['gust_kt'] if i is not None])/5)*5
    
    ax3.plot(data['local_datetimes'],data['gust_kt'],color='grey',alpha=0.6)
    ax3.set_ylabel('Wind Gusts [kt]', color='grey')
    ax3.set_ylim((0,xmax))
    
    ax3v2 = ax3.twinx()
    ax3v2.plot(data['local_datetimes'],data['wind_spd_kt'],color='blue')
    ax3v2.tick_params(axis='y', labelcolor='blue')
    ax3v2.set_ylabel('Wind Speed [kt]', color='blue')
    ax3v2.set_ylim((0,xmax))
    
    ######################## Plot Precip ########################
    ax4.bar([dt-relativedelta(hours=1.5) for dt in data_3hrly['local_datetimes_3hrly']],
            data_3hrly['rain_accum_3hrly'], width=0.1, zorder=1e2)
    for i, value in enumerate(data_3hrly['rain_accum_3hrly']):
        if value>0.0:
            ax4.text([dt-relativedelta(hours=1.5) for dt in data_3hrly['local_datetimes_3hrly']][i], value + value/100, ' '+f'{value:.1f}', 
                     ha='center', va='bottom', rotation=90, zorder=1e2)
        #elif value>0.0 and value <=1.0:
        #    ax4.text(data_3hrly['local_datetimes_3hrly'][i], value + 0.1, f'{value:.1f}', ha='center', va='bottom')
    max_rain_3hrly=[item for item in data_3hrly['rain_accum_3hrly'] if str(item) != 'nan']

    if max(max_rain_3hrly)<1:
        ax4.set_ylim((0,1.5))
        lineY=1.25
    else:
        ax4.set_ylim((0,max(max_rain_3hrly)+np.ceil(max(max_rain_3hrly)/2)))
        lineY=max(max_rain_3hrly)+(max(max_rain_3hrly)/3)

    ####### Latest 24 hour accumulation up to current time ######
    ax4.plot([data_09h00['local_datetimes_09h00'][0]+relativedelta(minutes=30),
                  data['local_datetimes'][0]],
                    [lineY,lineY],
                    color='red', 
                    linestyle='-', 
                    linewidth=2)
    ax4.text(data['local_datetimes'][0],
            lineY,
            str([i for i in data['rain_accum'] if i is not np.nan][0]), 
            ha='center', va='bottom', color='red')

    ################### All 24 hour accumulations ##################
    for i in range(len(data_09h00['local_datetimes_09h00'])-1):
        ax4.plot([data_09h00['local_datetimes_09h00'][i+1]+relativedelta(minutes=30),
                  data_09h00['local_datetimes_09h00'][i]-relativedelta(minutes=30)],
                    [lineY,lineY],
                    color='red', 
                    linestyle='-', 
                    linewidth=2)
        ax4.text(data_09h00['local_datetimes_09h00'][i+1]+relativedelta(hours=12),
                lineY,
                str(data_09h00['rain_accum_09h00'][i]), 
                ha='center', va='bottom', color='red')
    
    ax4.set_ylabel('Precipitation [mm]', color='black')
    
    for ax in (ax1,ax2,ax3,ax4):
        ax.set_xticks(data_axis_24hrly['local_datetimes_24hrly'])
        ax.set_xticklabels('')
        ax.set_xticks(data_axis_6hrly['local_datetimes_6hrly'], minor=True)
       
        ax.xaxis.grid(True, which='major', linestyle='-', linewidth=1.5)
        ax.yaxis.grid(True, which='major', linestyle='-', linewidth=0.4)
        ax.grid(True, which='minor', linestyle='-', linewidth=0.4)
        
        ax.set_xlim((data_axis_24hrly['local_datetimes_24hrly'][0].replace(hour=0, minute=0, second=0, microsecond=0),
                     data['local_datetimes'][0]+relativedelta(hours=3)))
    
    ax4.set_xticklabels(data_axis_24hrly['axis_dts'])
    ax0.set_axis_off()
    
    table_data = [
                  ['Date: ', data['local_datetimes'][0].strftime('%d %B %Y')],
                  ['Time: ', data['local_datetimes'][0].strftime('%Hh%M')],
                  ['Temperature: ', str(data['air_temp'][0])+degrees+'C'],
                  ['Dewpoint: ', str(data['dewpt'][0])+degrees+'C'],
                  ['Apparent Temperature: ', str(data['apparent_t'][0])+degrees+'C'],
                  ['Relative Humidity: ', str(data['rel_hum'][0])+'%'],
                  ['Wind Speed [kt]: ', str(data['wind_spd_kt'][0])+' kt '+str(data['wind_dir'][0])],
                  ['Wind Gusts [kt]: ', str(data['gust_kt'][0])+' kt'],
                  ['Max. Temp: ', str(tmax[-1])+' ('+tmax_dt[-1].strftime('%Hh%M')+')'],
                  ['Min Temp: ', str(tmin[-1])+' ('+tmin_dt[-1].strftime('%Hh%M')+')']
                 ]
    
    table = ax0.table(cellText=table_data, 
                      edges='open', colLabels=None, cellLoc='center', loc='bottom', 
                      bbox=[0.25, 0, 0.5, 1.5])
    
    fig.tight_layout()
    
    pos1 = ax0.get_position() # get the original position 
    pos2 = [pos1.x0, pos1.y0-0.05,  pos1.width, pos1.height+0.01] 
    ax0.set_position(pos2) # set a new position
    
    fig.suptitle(header['name']+', '+header['state'], fontsize=20)

    outfn=header['name'].replace(" ", "_").replace("_/_","")+'.jpg'
    outfile=outpath+outfn
    plt.savefig(outfile, dpi=300)

    destination_blob_name = "data/OBS/station/"+outfn
    upload_to_bucket(bucket_name, outfile, destination_blob_name)
    cache_control_value = "no-store"  # or "max-age=60"
    set_cache_control(bucket_name, destination_blob_name, cache_control_value)
