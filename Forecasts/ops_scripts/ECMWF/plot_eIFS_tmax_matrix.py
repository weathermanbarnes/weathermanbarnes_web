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

tmax=[]
valid_times=[]
for it,t in enumerate(tqdm(trange, total=len(trange))):
    if t>144:
        res=6
    else:
        res=3
    tmax_t=[]
    for ii,i in enumerate(range(ens_length+1)):
        tmax_tt=[]
        for tt in range(t-24+res,t+res,res):
            fpath = outpath+"data/"
            fn=fpath+indate+inrun+'00-'+str(tt)+'h-enfo-ef_mx2t'+str(res)+'_sfc_'+str(i)+'.grib2'
            tm=xr.open_dataset(fn,engine='cfgrib')['mx2t'+str(res)]
            tmax_tt.append(tm)
            dstr_init_long=tm.time.dt.strftime('%H%M UTC %d %b %Y').values
        tmax_tt=xr.concat(tmax_tt,dim='time')
        tmax_tt=tmax_tt.rename('mx2t')
        tmax_t.append(tmax_tt.max(dim='time').expand_dims('number'))
    tmax_t=xr.concat(tmax_t,dim='number')
    tmax_t['time']=indatetime+relativedelta(hours=t)
    tmax.append(tmax_t)
    valid_times.append(tm.valid_time)
valid_times=xr.concat(valid_times,dim='valid_time')
tmax=xr.concat(tmax,dim='time')
tmax_C=np.round(tmax-273.15,0)
tmax_C['time']=valid_times.values

del tmax_t, tmax_tt, tmax, tm

with open('/home/565/mb0427/gdata-gb02/Website/Forecasts/ops_scripts/city_metadata.pkl', 'rb') as handle:
    city_dict = pickle.load(handle)

tlevels =  range(0,55,5)
cmap = nclcmaps.cmap('ncl_default')(range(26,256,int(np.floor(256/len(tlevels)+2))))
cmap = cmap[0:-1]
cmap = ListedColormap(np.concatenate([cmap,
                                        nclcmaps.cmap('MPL_gist_nca')(range(102,103,7)),
                                        np.array([[0.5,0.5,0.5,1]]),
                                        np.array([[0,0,0,1]])]))

tlevels =  range(-5,55,5)
norm = BoundaryNorm(tlevels, cmap.N)

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
    
        tmax_loc=tmax_C.sel(latitude=inlat,longitude=inlon,method='nearest')
        
        masked_data=np.transpose(tmax_loc.values)
        masked_data=np.vstack([masked_data, np.mean(masked_data,axis=0)])
        cf=ax.imshow(masked_data,cmap=cmap,norm=norm)#vmin=0.2,vmax=800)
        for it in range(masked_data.shape[0]):
            for im in range(masked_data.shape[1]):
                tempval=np.round(masked_data.data[it,im],0)
                temptext=str(int(tempval))
                if tempval>=45:
                    fontcolor='white'
                else: 
                    fontcolor='black'
                ax.text(im, it, temptext, ha='center', va='center', color=fontcolor, 
                             fontsize=6)#, fontweight='bold')
        
        mondays = np.where(tmax_loc.time.dt.weekday == 0)[0]  # Indices of Mondays
        for monday in mondays:
            ax.axvline(x=monday-0.5, color="black", linestyle="-", linewidth=0.5, alpha=1)
        fridays = np.where(tmax_loc.time.dt.weekday == 4)[0]  # Indices of Fridays
        for friday in fridays:
            ax.axvline(x=friday+0.5, color="black", linestyle="--", linewidth=0.75, alpha=1)
            
        ax.axhline(y=51-0.5, color="black", linestyle="-", linewidth=3, alpha=1)

        
        ensname=np.hstack([tmax_loc.number.values.astype('<U4'),np.array('Mean')])
        yticks=ax.set_yticks(list(range(len(ensname))))
        yticklabs=ax.set_yticklabels(ensname)
        ax.set_ylabel('Ensemble Member')
        xticks=ax.set_xticks(list(range(len(tmax_loc.time))))
        xticklabs=ax.set_xticklabels(tmax_loc.time.dt.strftime('%a %d/%m/%y').values,rotation=90,fontsize=8)
        ax.set_xlabel('Forecast Valid Date')
        ax.tick_params(length=0) 
        
        ax.set_title(loc+' ('+state+')')
    
    fig.suptitle('Maximum Temperature Forecast for '+plot_name+' [C] | ECMWF-eIFS | Init: ' + dstr_init_long ,y=0.935,fontsize=20)
    
    outfile=outpath+'images/ECMWF-eIFS_Tmax_'+plot_name.replace(" ", "")+'.jpg'
    plt.savefig(outfile, dpi=300)
    crop(outfile,in_padding=10)
    #plt.show()
