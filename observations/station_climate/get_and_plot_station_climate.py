import pandas as pd
import requests
import os
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import numpy as np
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, BoundaryNorm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker

from PIL import ImageOps,Image # pip install Pillow
import sys
import glob
# Trim all png images with white background in a folder
# Usage "python PNGWhiteTrim.py ../someFolder padding"
def crop(path, in_padding=1,**kwargs):
    Image.MAX_IMAGE_PIXELS = None
    
    try:
        padding = int(in_padding)
        padding = np.asarray([-1*padding, -1*padding, padding, padding])
    except :
        print("Usage: python PNGWhiteTrim.py ../someFolder padding")
        sys.exit(1)
    
    filePaths = glob.glob(path) #search for all png images in the folder
    
    if len(filePaths) == 0:
        print("No files detected!")
    
    for filePath in filePaths:
        image=Image.open(filePath)
        image.load()
        imageSize = image.size
    
        # remove alpha channel
        invert_im = image.convert("RGB")
    
        # invert image (so that white is 0)
        invert_im = ImageOps.invert(invert_im)
        imageBox = invert_im.getbbox()
        imageBox = tuple(np.asarray(imageBox)+padding)
    
        cropped=image.crop(imageBox)
        print(filePath, "Size:", imageSize, "New Size:", imageBox)
        cropped.save(filePath)

from google.cloud import storage

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/565/mb0427/gadi_web_bucket_access.json"
#os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/mbar0087/michael-local-web-key.json"
bucket_name = "www.weathermanbarnes.com"  # Replace with your bucket name

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

##########################################################################################    
    
id_dict={
   'Adelaide Airport': {'station_name': 'adelaide_airport',
                         'state_name': 'sa',
                         'stat_clim_t': '023034',
                         'stat_clim_r': '023034',
                         'stat_clim_s': '023034'},
   'Perth Airport': {'station_name': 'perth_airport',
                         'state_name': 'wa',
                         'stat_clim_t': '009021',
                         'stat_clim_r': '009021',
                         'stat_clim_s': '009021'},
   'Brisbane Airport': {'station_name': 'brisbane_aero',
                         'state_name': 'qld',
                         'stat_clim_t': '040842',
                         'stat_clim_r': '040842',
                         'stat_clim_s': '040842'},
   'Darwin Airport': {'station_name': 'darwin_airport',
                         'state_name': 'nt',
                         'stat_clim_t': '014015',
                         'stat_clim_r': '014015',
                         'stat_clim_s': '014015'},
   'Melbourne Airport': {'station_name': 'melbourne_airport',
                         'state_name': 'vic',
                         'stat_clim_t': '086282',
                         'stat_clim_r': '086282',
                         'stat_clim_s': '086282'},
   'Melbourne (Olympic Park)': {'station_name': 'melbourne_(olympic_park)',
                                 'state_name': 'vic',
                                 'stat_clim_t': '086338',
                                 'stat_clim_r': '086338',
                                 'stat_clim_s': '086338'},
   'Hobart': {'station_name': 'hobart_(ellerslie_road)',
                                 'state_name': 'tas',
                                 'stat_clim_t': '094029',
                                 'stat_clim_r': '094029',
                                 'stat_clim_s': '094029'},
   'Hobart Airport': {'station_name': 'hobart_airport',
                                 'state_name': 'tas',
                                 'stat_clim_t': '094250',
                                 'stat_clim_r': '094264',
                                 'stat_clim_s': '094264'},
   'Moorabbin Airport': {'station_name': 'moorabbin_airport',
                                 'state_name': 'vic',
                                 'stat_clim_t': '086077',
                                 'stat_clim_r': '086077',
                                 'stat_clim_s': '086077'},
   'Sydney Airport': {'station_name': 'sydney_airport_amo',
                                 'state_name': 'nsw',
                                 'stat_clim_t': '066037',
                                 'stat_clim_r': '066037',
                                 'stat_clim_s': '066024'},
   'Canberra Airport': {'station_name': 'canberra_airport',
                                 'state_name': 'nsw',
                                 'stat_clim_t': '070351',
                                 'stat_clim_r': '070351',
                                 'stat_clim_s': '066052'},
}

##########################################################################################

outpath='/home/565/mb0427/gdata-w40/Forecasts/ops_scripts/observations/station_climate/'
inpath_clim='/home/565/mb0427/gdata-w40/Forecasts/ops_scripts/observations/station_climate/climatologies/'
destination_path='data/OBS/station_climate/'

nowdt=datetime.now()-relativedelta(days=1)
nowyr=nowdt.year
nowmn=nowdt.month
nowmnstr=nowdt.strftime('%Y%m')

##########################################################################################

for stat in id_dict.keys():
    print(stat)
    data=[]
    for i in range(1,nowmn+1):
        idt=datetime(nowyr,i,1)
        idtmnstr=idt.strftime('%Y%m')
        idfile=id_dict[stat]['station_name']+'-'+idtmnstr
        bom_url = 'http://reg.bom.gov.au/clim_data/IDCKWCDEA0/tables/'+id_dict[stat]['state_name']+'/'+id_dict[stat]['station_name']+'/'+id_dict[stat]['station_name']+'-'+idtmnstr+'.csv'
        bom_csv = outpath+'current_year/'+idfile+".csv"
        r = requests.get(bom_url)
        f = open(bom_csv, "w")
        f.write(r.text)
        f.close()

        header = pd.read_csv(bom_csv)
        header=header.astype(str)
        header=(header.iloc[6] + '_' +header.iloc[7] + '_' + header.iloc[8])
        header=header.map(lambda x: x.lstrip('nan_'))
        #header=header.map(lambda x: x.lstrip(' '))
        df = pd.read_csv(bom_csv,skiprows=13,names=header.to_list(),skipfooter=1,engine='python')
        #print(df)
        data.append(df)
    data=pd.concat(data)
    data['Date'] = pd.to_datetime(data['Date'],format='%d/%m/%Y')
    data['Maximum_Temperature'] = pd.to_numeric(data['Maximum_Temperature'], errors='coerce').fillna(np.nan)
    data['Rain_0900-0900'] = pd.to_numeric(data['Rain_0900-0900'], errors='coerce').fillna(np.nan)
    data['Solar_Radiation'] = pd.to_numeric(data['Solar_Radiation'], errors='coerce').fillna(np.nan)
    
    ####################################################################################

    max_temp_clim_csv=inpath_clim+'IDCJAC0010_'+id_dict[stat]['stat_clim_t']+'_1800_Data.csv'
    df = pd.read_csv(max_temp_clim_csv)
    df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day']])
    df['MD']=[row['Date'].strftime('%m%d') for index, row in df.iterrows()]
    df = df[df.Date<datetime(nowyr,1,1)]

    meanTx=[]
    stdevTx=[]
    mds=sorted(df['MD'].unique())
    for md in mds:
        meanTx.append(df[df.MD==md]['Maximum temperature (Degree C)'].mean(skipna=True))
        stdevTx.append(df[df.MD==md]['Maximum temperature (Degree C)'].std(skipna=True))
    dates=[datetime.strptime(str(nowyr)+i,'%Y%m%d') for i in mds]

    fig,ax=plt.subplots(1,1, figsize=(12,4))

    ax.bar(data['Date'],data['Maximum_Temperature'],color='dodgerblue')
    ax.plot(dates,meanTx,color='grey')
    ax.fill_between(dates, np.array(meanTx)-np.array(stdevTx), np.array(meanTx)+np.array(stdevTx), color='grey', alpha=0.25)
    ax.scatter(data['Date'].iloc[-1],data['Maximum_Temperature'].iloc[-1],
               s=5, color='dodgerblue')
    ax.text(data['Date'].iloc[-1],data['Maximum_Temperature'].iloc[-1],
           str(data['Maximum_Temperature'].iloc[-1])+'\n'+data['Date'].iloc[-1].strftime('%d/%m'),
           va='center',ha='left')
    ax.set_xlim(dates[0],dates[-1])
    ax.set_ylabel('Maximum Temperature [C]')
    ax.set_xlabel('Date')
    ax.set_title('Maximum Temperature for '+stat+'\n'+str(nowyr)+' (blue) | Daily Climatology ('+str(df.Year.min())+'-'+str(df.Year.max())+', grey)')
    ax.grid()
    outfn=id_dict[stat]['station_name']+'_Tx.png'
    outfile=outpath+outfn
    plt.savefig(outfile,dpi=600)
    crop(outfile,in_padding=10)
    plt.close(fig)
    
    destination_blob_name = destination_path+outfn
    upload_to_bucket(bucket_name, outfile, destination_blob_name)
    cache_control_value = "no-store"  # or "max-age=60"
    set_cache_control(bucket_name, destination_blob_name, cache_control_value)
    
    ####################################################################################
    
    rain_clim_csv=inpath_clim+'IDCJAC0009_'+id_dict[stat]['stat_clim_r']+'_1800_Data.csv'
    df = pd.read_csv(rain_clim_csv)
    df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day']])
    df['MD']=[row['Date'].strftime('%m%d') for index, row in df.iterrows()]
    df = df[df.Date<datetime(nowyr,1,1)]

    meanRx=[]
    stdevRx=[]
    mds=sorted(df['MD'].unique())
    for md in mds:
        meanRx.append(df[df.MD==md]['Rainfall amount (millimetres)'].mean(skipna=True))
        stdevRx.append(df[df.MD==md]['Rainfall amount (millimetres)'].std(skipna=True))
    dates=[datetime.strptime(str(nowyr)+i,'%Y%m%d') for i in mds]

    fig,ax=plt.subplots(1,1, figsize=(12,4))

    ax.plot(data['Date'],np.cumsum(data['Rain_0900-0900']),color='black')
    ax.plot(dates,np.cumsum(meanRx),color='grey')
    #ax.fill_between(dates, np.cumsum(meanRx)-np.cumsum(stdevRx), np.cumsum(meanRx)+np.cumsum(stdevRx), color='grey', alpha=0.25)
    ax.scatter(data['Date'].iloc[-1],np.cumsum(data['Rain_0900-0900']).iloc[-1],
               s=5, color='black')
    ax.text(data['Date'].iloc[-1],np.cumsum(data['Rain_0900-0900']).iloc[-1],
           str(int(np.cumsum(data['Rain_0900-0900']).iloc[-1]))+'\n'+data['Date'].iloc[-1].strftime('%d/%m'),
           va='center',ha='left')
    
    axv2 = ax.twinx()
    #axv2.plot(data['local_datetimes'],data['wind_spd_kt'],color='blue')
    axv2.bar(data['Date'],data['Rain_0900-0900'],color='dodgerblue')
    axv2.tick_params(axis='y', labelcolor='dodgerblue')
    axv2.set_ylabel('Daily Rainfall [mm]', color='dodgerblue')
    #axv2.set_ylim((0,xmax))
    
    ax.set_xlim(dates[0],dates[-1])
    ax.set_ylabel('Rainfall [mm]')
    ax.set_xlabel('Date')
    ax.set_title('24hr Rainfall for '+stat+'\n'+str(nowyr)+' Cumulative (black) | '+str(nowyr)+' Daily (blue) | Cumulative Daily Climatology ('+str(df.Year.min())+'-'+str(df.Year.max())+', grey)')
    ax.grid()
    outfn=id_dict[stat]['station_name']+'_Rx.png'
    outfile=outpath+outfn
    plt.savefig(outfile,dpi=600)
    crop(outfile,in_padding=10)
    plt.close(fig)
    
    destination_blob_name = destination_path+outfn
    upload_to_bucket(bucket_name, outfile, destination_blob_name)
    cache_control_value = "no-store"  # or "max-age=60"
    set_cache_control(bucket_name, destination_blob_name, cache_control_value)
    
    ####################################################################################
    
    solar_clim_csv=inpath_clim+'IDCJAC0016_'+id_dict[stat]['stat_clim_s']+'_1800_Data.csv'
    df = pd.read_csv(solar_clim_csv)
    df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day']])
    df['MD']=[row['Date'].strftime('%m%d') for index, row in df.iterrows()]
    df = df[df.Date<datetime(nowyr,1,1)]

    meanSx=[]
    stdevSx=[]
    mds=sorted(df['MD'].unique())
    for md in mds:
        meanSx.append(df[df.MD==md]['Daily global solar exposure (MJ/m*m)'].mean(skipna=True))
        stdevSx.append(df[df.MD==md]['Daily global solar exposure (MJ/m*m)'].std(skipna=True))
    dates=[datetime.strptime(str(nowyr)+i,'%Y%m%d') for i in mds]

    fig,ax=plt.subplots(1,1, figsize=(12,4))

    ax.bar(data['Date'],data['Solar_Radiation'],color='dodgerblue')
    ax.plot(dates,meanSx,color='grey')
    ax.fill_between(dates, np.array(meanSx)-np.array(stdevSx), np.array(meanSx)+np.array(stdevSx), color='grey', alpha=0.25)
    ax.set_xlim(dates[0],dates[-1])
    ax.set_ylabel('Solar Radiation [$MJ.m^{-2}$]')
    ax.set_xlabel('Date')
    ax.set_title('Solar Radiation for '+stat+'\n'+str(nowyr)+' (blue) | Daily Climatology ('+str(df.Year.min())+'-'+str(df.Year.max())+', grey)')
    ax.grid()
    outfn=id_dict[stat]['station_name']+'_Sx.png'
    outfile=outpath+outfn
    plt.savefig(outfile,dpi=600)
    crop(outfile,in_padding=10)
    plt.close(fig)
    
    destination_blob_name = destination_path+outfn
    upload_to_bucket(bucket_name, outfile, destination_blob_name)
    cache_control_value = "no-store"  # or "max-age=60"
    set_cache_control(bucket_name, destination_blob_name, cache_control_value)
