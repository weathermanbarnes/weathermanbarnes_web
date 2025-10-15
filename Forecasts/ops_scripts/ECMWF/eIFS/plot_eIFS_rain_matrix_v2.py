import sys
import os
import io
import xarray as xr
import numpy as np
from scipy import interpolate
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from tqdm import tqdm
import matplotlib.pyplot as plt
import nclcmaps
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, BoundaryNorm
import pickle
import cfgrib
import subprocess

from PIL import Image # pip install Pillow
import sys
import glob
from PIL import ImageOps
import numpy as np

from google.cloud import storage # Still needed for other GCS operations like uploading plots
storage_client = storage.Client()

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

def set_cache_control(bucket_name, blob_name, cache_control_value):
    """Set the Cache-Control metadata for a GCS blob."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    blob.cache_control = cache_control_value
    blob.patch()

def save_blob_to_bucket(img_buffer,output_bucket_name,output_blob_name,cache_control='no-store'):
    # 3. Get the output bucket and upload the image
    output_bucket = storage_client.bucket(output_bucket_name)
    output_blob = output_bucket.blob(output_blob_name)

    output_blob.upload_from_file(img_buffer, content_type='image/png')
    set_cache_control(output_bucket_name, output_blob_name, 'no-store')

    print(f"Plot '{output_blob_name}' uploaded to bucket '{output_bucket_name}'.")


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("date",help="init date",type=str)
parser.add_argument("run",help="init run",type=int)
args = parser.parse_args()
INDATEstr = args.date
RUN = args.run

outpath=f'data/ECMWF/eIFS/point_forecasts'
inpath= f"/mnt/ecmwf-data/{INDATEstr}/{RUN}z/ifs/0p25/enfo/"

indatetime=datetime.strptime(INDATEstr,'%Y%m%d')
indatetime=indatetime+relativedelta(hours=RUN)

figsize=(30,17)

inrun=indatetime.strftime('%H%M')
indate=indatetime.strftime('%Y%m%d')

plot_extent=[100,170,-50,-5]
trange=range(36,361,24)
accum_precip=[]
valid_times=[]
for it,t in enumerate(tqdm(trange, total=len(trange))):
    fn=inpath+indate+inrun+'00-'+str(t)+'h-enfo-ef.grib2'
    idxfn=''
    idxfn=indate+inrun+'00-'+str(t)+'h-enfo-ef.index'
    pcurl = subprocess.Popen(['curl','-s',inpath+idxfn,'-o',outpath+idxfn],stdout=subprocess.PIPE)
    os.waitpid(pcurl.pid,0)
    prcpcf=cfgrib.open_dataset(fn,engine='cfgrib', backend_kwargs={'indexpath': idxfn, 'filter_by_keys': {'shortName': 'tp','dataType': 'cf'}}, decode_timedelta=True).expand_dims(number=[0])
    prcppf=cfgrib.open_dataset(fn,engine='cfgrib', backend_kwargs={'indexpath': idxfn, 'filter_by_keys': {'shortName': 'tp','dataType': 'pf'}}, decode_timedelta=True)

    prcp=xr.concat([prcpcf,prcppf],dim='number')

    if t==24:
        prcp24=prcp*1
    else:
        fn=inpath+indate+inrun+'00-'+str(t-24)+'h-enfo-ef.grib2'
        idxfn=''
        idxfn=indate+inrun+'00-'+str(t-24)+'h-enfo-ef.index'
        pcurl = subprocess.Popen(['curl','-s',inpath+idxfn,'-o',idxfn],stdout=subprocess.PIPE)
        os.waitpid(pcurl.pid,0)
        prcpcf0=cfgrib.open_dataset(fn,engine='cfgrib', backend_kwargs={'indexpath': idxfn, 'filter_by_keys': {'shortName': 'tp','dataType': 'cf'}}, decode_timedelta=True).expand_dims(number=[0])
        prcppf0=cfgrib.open_dataset(fn,engine='cfgrib', backend_kwargs={'indexpath': idxfn, 'filter_by_keys': {'shortName': 'tp','dataType': 'pf'}}, decode_timedelta=True)

        prcp0=xr.concat([prcpcf0,prcppf0],dim='number')
        prcp24=prcp-prcp0

    dstr_init_long=prcp24.time.dt.strftime('%H%M UTC %d %b %Y').values
    valid_times.append(prcp.valid_time)
    accum_precip.append(prcp24)

accum_precip = xr.concat(accum_precip,dim='time')    
valid_times=xr.concat(valid_times,dim='valid_time')
accum_precip['time']=valid_times.values

print(accum_precip)

with open('city_metadata.pkl', 'rb') as handle:
    city_dict = pickle.load(handle)

rain_levels =  [0.2,0.5,1,2,5,15,20,25,30,35,40,45,50,60,70,80,90,100,125,150,200]
cmap = nclcmaps.cmap('WhiteBlueGreenYellowRed')(range(26,256,int(np.floor(256/len(rain_levels)+2))))
cmap = ListedColormap(np.concatenate([cmap,nclcmaps.cmap('MPL_gist_nca')(range(102,129,7))]))
norm = BoundaryNorm(rain_levels, cmap.N)

loc_sets={}
loc_sets['Australian Captial Cities']=['Perth','Adelaide','Melbourne','Hobart','Sydney','Canberra','Brisbane','Darwin']
loc_sets['Victoria']=['Melbourne','Monash','Lilydale','Geelong','Ballarat','Bendigo','Mildura','Traralgon, Gippsland']
loc_sets['Tasmania']=['Hobart','Kingston','Port Arthur','Bruny Island','Queenstown','Launceston','Devonport','Burnie']
loc_sets['New South Wales']=['Sydney','Canberra','Wagga Wagga','Wollongong','Newcastle','Port Macquarie','Coffs Harbour','Lismore']
loc_sets['Queensland']=['Brisbane','Gold Coast','Sunshine Coast','Bundaberg','Rockhampton','Mackay','Townsville','Cairns']
loc_sets['Western Australia']=['Perth','Geraldton','Margaret River','Kalgoorlie-Boulder','Albany','Exmouth','Port Hedland','Broome']
loc_sets['South African Captial Cities']=['Cape Town','George','Port Elizabeth','East London','Durban','Bloemfontein','Johannesburg','Pretoria']

for plot_name in loc_sets.keys():
    locs=loc_sets[plot_name]
    fig,axs=plt.subplots(1,len(locs),figsize=(33, 12))
    iloc=-1
    for ax,loc in zip(axs,locs):
        inlat=city_dict[loc]['latitude']
        inlon=city_dict[loc]['longitude']
        state=city_dict[loc]['state']
        
        accum_precip_loc=accum_precip.sel(latitude=inlat,longitude=inlon,method='nearest')
        
        masked_data=np.transpose(accum_precip_loc.tp.values)*1000
        masked_data=np.vstack([masked_data, np.max(masked_data,axis=0)])
        masked_data=np.vstack([masked_data, np.quantile(masked_data,0.75,axis=0)])
        masked_data=np.vstack([masked_data, np.mean(masked_data,axis=0)])
        masked_data=np.vstack([masked_data, np.quantile(masked_data,0.25,axis=0)])
        masked_data=np.vstack([masked_data, np.min(masked_data,axis=0)])
        masked_data = np.ma.masked_where(masked_data < 0.2, masked_data)
        ax.imshow(masked_data,cmap=cmap,norm=norm)#vmin=0.2,vmax=800)
        for it in range(masked_data.shape[0]):
            for im in range(masked_data.shape[1]):
                rainval=np.round(masked_data.data[it,im],1)
                if rainval<1 and rainval>=0.1:
                    raintext=str(rainval)[1::]
                elif rainval<0.1:
                    raintext=str(' ')
                else:
                    raintext=str(int(np.round(masked_data.data[it,im],1)))
                ax.text(im, it, raintext, ha='center', va='center', color='black', 
                             fontsize=6)#, fontweight='bold')
        
        mondays = np.where(accum_precip_loc.time.dt.weekday == 0)[0]  # Indices of Mondays
        for monday in mondays:
            ax.axvline(x=monday-0.5, color="black", linestyle="-", linewidth=0.75, alpha=1)
        fridays = np.where(accum_precip_loc.time.dt.weekday == 4)[0]  # Indices of Fridays
        for friday in fridays:
            ax.axvline(x=friday+0.5, color="black", linestyle="--", linewidth=0.75, alpha=1)
            
        ax.axhline(y=len(accum_precip.number)-0.5, color="black", linestyle="-", linewidth=3, alpha=1)
        ax.axhline(y=len(accum_precip.number)+0.5, color="black", linestyle="-", linewidth=1, alpha=1)
        ax.axhline(y=len(accum_precip.number)+1.5, color="black", linestyle="-", linewidth=1, alpha=1)
        ax.axhline(y=len(accum_precip.number)+2.5, color="black", linestyle="-", linewidth=1, alpha=1)
        ax.axhline(y=len(accum_precip.number)+3.5, color="black", linestyle="-", linewidth=1, alpha=1)
        ax.axhline(y=len(accum_precip.number)+4.5, color="black", linestyle="-", linewidth=1, alpha=1)
        
        ensname=np.hstack([accum_precip_loc.number.values.astype('<U4'),np.array('Max')])
        ensname=np.hstack([accum_precip_loc.number.values.astype('<U4'),np.array('Q3')])
        ensname=np.hstack([accum_precip_loc.number.values.astype('<U4'),np.array('Mean')])
        ensname=np.hstack([accum_precip_loc.number.values.astype('<U4'),np.array('Q1')])
        ensname=np.hstack([accum_precip_loc.number.values.astype('<U4'),np.array('Min')])
        yticks=ax.set_yticks(list(range(len(ensname))))
        yticklabs=ax.set_yticklabels(ensname)
        ax.set_ylabel('Ensemble Member')
        xticks=ax.set_xticks(list(range(len(accum_precip_loc.time))))
        xticklabs=ax.set_xticklabels(accum_precip_loc.time.dt.strftime('%a %d/%m/%y').values,rotation=90,fontsize=8)
        ax.set_xlabel('Forecast Valid Date')
        ax.tick_params(length=0) 
        
        ax.set_title(loc+' ('+state+')')
    
    fig.suptitle('24hr Precipitation Forecast for '+plot_name+' [mm] | ECMWF-eIFS | Init: ' + dstr_init_long ,y=0.935,fontsize=20)

    outblob = io.BytesIO()  # Create a new, unique in-memory buffer
    fig.savefig(outblob, format='png', bbox_inches='tight', dpi=300) # Changed plt.savefig to fig.savefig
    outblob.seek(0)

    save_blob_to_bucket(outblob,"www.weathermanbarnes.com",
                        f'{outpath}/ECMWF-eIFS_Rain_{plot_name.replace(" ", "")}.jpg')
    
    #outfile=outpath+'images/ECMWF-eIFS_Rain_'+plot_name.replace(" ", "")+'.jpg'
    #plt.savefig(outfile, dpi=300)
    #crop(outfile,in_padding=10)
#plt.show()
        
