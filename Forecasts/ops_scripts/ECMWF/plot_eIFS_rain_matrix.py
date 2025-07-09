import sys
sys.path.append("/home/565/mb0427/gdata-gb02/Website/Forecasts/ops_scripts/")
import os
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

from PIL import Image # pip install Pillow
import sys
import glob
from PIL import ImageOps
import numpy as np
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

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("date",help="init date",type=str)
parser.add_argument("run",help="init run",type=int)
args = parser.parse_args()
INDATEstr = args.date
RUN = args.run

home_path='/home/565/mb0427/gdata-gb02/Website/Forecasts/ops_scripts/ECMWF/'
outpath='/scratch/gb02/mb0427/Website/Forecasts/ECMWF/eIFS/'

indatetime=datetime.strptime(INDATEstr,'%Y%m%d')
indatetime=indatetime+relativedelta(hours=RUN)

centlon=0
figsize=(30,17)
barblength=7
ens_length=50

inrun=indatetime.strftime('%H%M')
indate=indatetime.strftime('%Y%m%d')

plot_extent=[100,170,-50,-5]
trange=range(24,360+24,24)
accum_precip=[]
valid_times=[]
for it,t in enumerate(tqdm(trange, total=len(trange))):
    accum_precip_t=[]
    for ii,i in enumerate(range(ens_length+1)):
        fpath = outpath+"data/"
        fn=fpath+indate+inrun+'00-'+str(t)+'h-enfo-ef_tp_sfc_'+str(i)+'.grib2'
        prcp=xr.open_dataset(fn,engine='cfgrib')

        if t==24:
            prcp24=prcp*1
        else:
            fn0=fpath+indate+inrun+'00-'+str(t-24)+'h-enfo-ef_tp_sfc_'+str(i)+'.grib2'
            prcp0=xr.open_dataset(fn0,engine='cfgrib')
            prcp24=prcp-prcp0

        prcp24=prcp24.expand_dims(['number'])
        accum_precip_t.append(prcp24)
        dstr_init_long=prcp24.time.dt.strftime('%H%M UTC %d %b %Y').values

    valid_times.append(prcp.valid_time)
    
    accum_precip_t = xr.concat(accum_precip_t,dim='number')
    accum_precip.append(accum_precip_t)
accum_precip = xr.concat(accum_precip,dim='time')    
valid_times=xr.concat(valid_times,dim='valid_time')
accum_precip['time']=valid_times.values

with open('/home/565/mb0427/gdata-gb02/Website/Forecasts/ops_scripts/city_metadata.pkl', 'rb') as handle:
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
        masked_data=np.vstack([masked_data, np.mean(masked_data,axis=0)])
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
            
        ax.axhline(y=51-0.5, color="black", linestyle="-", linewidth=3, alpha=1)

        
        ensname=np.hstack([accum_precip_loc.number.values.astype('<U4'),np.array('Mean')])
        yticks=ax.set_yticks(list(range(len(ensname))))
        yticklabs=ax.set_yticklabels(ensname)
        ax.set_ylabel('Ensemble Member')
        xticks=ax.set_xticks(list(range(len(accum_precip_loc.time))))
        xticklabs=ax.set_xticklabels(accum_precip_loc.time.dt.strftime('%a %d/%m/%y').values,rotation=90,fontsize=8)
        ax.set_xlabel('Forecast Valid Date')
        ax.tick_params(length=0) 
        
        ax.set_title(loc+' ('+state+')')
    
    fig.suptitle('24hr Precipitation Forecast for '+plot_name+' [mm] | ECMWF-eIFS | Init: ' + dstr_init_long ,y=0.935,fontsize=20)
    
    outfile=outpath+'images/ECMWF-eIFS_Rain_'+plot_name.replace(" ", "")+'.jpg'
    plt.savefig(outfile, dpi=300)
    crop(outfile,in_padding=10)
#plt.show()
        