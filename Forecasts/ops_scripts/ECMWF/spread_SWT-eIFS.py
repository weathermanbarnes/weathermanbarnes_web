import sys
sys.path.append("Forecasts/ops_scripts/")
import os
import xarray as xr
import numpy as np
from scipy import interpolate
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from tqdm import tqdm
import matplotlib.pyplot as plt
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

def assign(u,v,clusterU,clusterV):
    if(np.shape(u)!=np.shape(clusterU)[1:]): 
        sys.exit('Wind velocity field not the same shape as cluster field, interpolate data to cluster grid first.')
    return np.argmin(np.sum((u[None,:,:]-clusterU)**2+(v[None,:,:]-clusterV)**2,axis=(-1,-2)))+1

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("date",help="init date",type=str)
parser.add_argument("run",help="init run",type=int)
args = parser.parse_args()
INDATEstr = args.date
RUN = args.run

home_path='Forecasts/ops_scripts/ECMWF/'
outpath='../scratch/eIFS/'

indatetime=datetime.strptime(INDATEstr,'%Y%m%d')
indatetime=indatetime+relativedelta(hours=RUN)

cluster_file=home_path+'../weather_regimes/clusters_030_850_1952-01_2023-12_12.nc'
clusters=xr.open_dataset(cluster_file)
lat_cluster,lon_cluster = np.meshgrid(clusters.latitude, clusters.longitude, indexing='ij')

with open(home_path+'../weather_regimes/SWT_WR_definition_v2.pkl', 'rb') as fp:
    SWTs = pickle.load(fp)
SWTnames = [SWTs[i]["WR"]+'-'+SWTs[i]["SWT"] for i in SWTs]

centlon=0
figsize=(30,17)
barblength=7
ens_length=50

inrun=indatetime.strftime('%H%M')
indate=indatetime.strftime('%Y%m%d')

trange=range(24,360+24,24)
cluster_labels=np.zeros((len(range(ens_length+1)),len(trange)))
cluster_spread=np.zeros((len(clusters.clusterID.values),len(trange)))

for it,t in enumerate(tqdm(trange, total=len(trange))):
    for ii,i in enumerate(range(ens_length+1)):
        fpath = outpath+"data/"
        fn=fpath+indate+inrun+'00-'+str(t)+'h-enfo-ef_u_850_'+str(i)+'.grib2'
        u=xr.open_dataset(fn,engine='cfgrib')
        u=u.reindex(latitude=list(reversed(u.latitude)))
        #ulat,ulon = np.meshgrid(u.latitude, u.longitude, indexing='ij')
        ulat=u.latitude.values; ulon=u.longitude.values
        f = interpolate.RegularGridInterpolator((ulat,ulon),u.u.values)
        u_int = f((lat_cluster,lon_cluster))
        fn=fpath+indate+inrun+'00-'+str(t)+'h-enfo-ef_v_850_'+str(i)+'.grib2'
        v=xr.open_dataset(fn,engine='cfgrib')
        #v=v.reindex(latitude=list(reversed(u.latitude)))
        vlat=v.latitude.values; vlon=v.longitude.values
        f = interpolate.RegularGridInterpolator((vlat,vlon),v.v.values)
        v_int = f((lat_cluster,lon_cluster))

        label = assign(u_int,v_int,clusters.clusterU.values,clusters.clusterV.values)
        SWTindex=SWTnames.index(SWTs[label]["WR"]+'-'+SWTs[label]["SWT"])
        cluster_labels[ii,it] = label
        cluster_spread[SWTindex,it]=cluster_spread[SWTindex,it]+1

regticks=[4,6,11,14,17,19,25]
datestrings=[(indatetime+relativedelta(hours=t)).strftime('%Y-%m-%d') for t in trange]

fig,ax=plt.subplots(1,1,figsize=(6, 8))

cluster_spread=np.int64(np.round(cluster_spread/(ens_length+1)*100))
masked_data = np.ma.masked_where(cluster_spread < 1, cluster_spread)
ax.imshow(masked_data,cmap='Spectral',vmin=0,vmax=100)
for it in range(masked_data.shape[0]):
    for im in range(masked_data.shape[1]):
        ax.text(im, it, str(int(np.round(cluster_spread[it,im],0))), ha='center', va='center', color='black', 
                     fontsize=6)#, fontweight='bold')
for regtick in regticks:
    ax.axhline(regtick-0.5,color='black')
    #ax.axvline(regtick-0.5,color='black')
yticks=ax.set_yticks(list(range(len(SWTs))))
yticklabs=ax.set_yticklabels(SWTnames)
ax.set_ylabel('Synoptic Weather Type')
xticks=ax.set_xticks(list(range(len(trange))))
xticklabs=ax.set_xticklabels(datestrings,rotation=90)
ax.set_xlabel('Forecast Valid Date')
ax.tick_params(length=0) 
ax.invert_yaxis()

outfile=outpath+'images/ECMWF-eIFS_AustralianSynopticWeatherTypes.jpg'
plt.savefig(outfile, dpi=300)
crop(outfile,in_padding=10)

fig, ax = plt.subplots(figsize=(12, 5))

# Plotting bars for each category in the groups
colors=[(134/255,0/255,34/255),(241/255,0/255,241/255),(255/255,134/255,255/255),(255/255,241/255,255/255), #Pink WH
        (255/255,204/255,51/255),(255/255,245/255,204/255), #Yellow CH
        (153/255,15/255,15/255),(178/255,44/255,44/255),(204/255,81/255,81/255),(229/255,126/255,126/255),(255/255,178/255,178/255), #Red EH
        (153/255,84/255,15/255),(204/255,142/255,81/255),(255/255,216/255, 178/255), #Brown TH
        (107/255,153/255,15/255),(163/255,204/255,81/255),(195/255,229/255,126/255), #Green FH (133/255,178/255,44/255),
        (66/255,44/255,178/255),(143/255,126/255,229/255), #Purple WCT
        (5/255,67/255,113/255),(15/255,107/255,153/255),(44/255,133/255,178/255),(81/255,163/255,204/255),(126/255,195/255,229/255),(178/255,229/255,255/255), #Blue COL
        (0/255,60/255,48/255),(1/255,102/255,95/255),(53/255,151/255,143/255),(128/255,205/255,193/255),(199/255,234/255,229/255)
       ]

# Number of bars (columns)
num_bars = cluster_spread.shape[1]
x = np.arange(num_bars)
bottom = np.zeros(num_bars)

regticks=[4,6,11,14,17,19,25]
for i, (row, color) in enumerate(zip(cluster_spread, colors)):
    # Draw the bar
    bars = plt.bar(x, row, bottom=bottom, color=color, label=SWTnames[i])
    # Add a black line to the bottom edge of every fourth stack
    if i in regticks:
        for bar in bars:
            x_pos = bar.get_x()
            width = bar.get_width()
            y_pos = bar.get_y()
            plt.plot([x_pos+0.01, x_pos + width-0.02], [y_pos, y_pos], color='black', linewidth=1)
    # Update the bottom for the next stack
    bottom += row

ax.legend(loc='upper right', bbox_to_anchor=(1.25, 0.85), ncols=2)

xticks=ax.set_xticks(list(range(len(trange))))
datestrings=[(indatetime+relativedelta(hours=t)).strftime('%Y-%m-%d') for t in trange]
xticklabs=ax.set_xticklabels(datestrings,rotation=90)
ax.set_xlabel('Forecast Valid Date')
ax.tick_params(length=0) 

fig.tight_layout()
outfile=outpath+'images/ECMWF-eIFS_AustralianSWT_bars.jpg'
plt.savefig(outfile, dpi=300)
#crop(outfile,in_padding=100)
