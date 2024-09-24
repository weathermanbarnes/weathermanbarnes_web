import sys
sys.path.append("/home/565/mb0427/gdata-gb02/Website/Forecasts/ops_scripts/")
import nclcmaps
import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, BoundaryNorm
import cartopy.crs as ccrs
from cartopy.feature import NaturalEarthFeature, LAND
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from windspharm.xarray import VectorWind
from windspharm.tools import prep_data, recover_data, order_latdim
import metpy.calc as mpcalc
from metpy.units import units
from tqdm import tqdm

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
parser.add_argument("timestep",help="in timestep",type=int)
args = parser.parse_args()
timestep = args.timestep
INDATEstr = args.date
RUN = args.run

indatetime=datetime.strptime(INDATEstr,'%Y%m%d')
indatetime=indatetime+relativedelta(hours=RUN)

home_path='/home/565/mb0427/gdata-gb02/Website/Forecasts/ops_scripts/ECMWF/'
outpath='/scratch/gb02/mb0427/Website/Forecasts/ECMWF/eIFS/'

import warnings
warnings.filterwarnings("ignore")

name='Australia'
plot_extent=[110,160,-40,-10]
#plot_extent=[80,180,-60,-5]
#plot_extent=[140,180,-50,-30]
centlon=0
figsize=(25,18)
barblength=7
ens_length=50

inrun=indatetime.strftime('%H%M')
indate=indatetime.strftime('%Y%m%d')
#trange=range(24,264,24)
trange=[timestep]
for t in trange:
    fig, axs = plt.subplots(7, 8, figsize=figsize, subplot_kw=dict(projection=ccrs.PlateCarree())) 
    ax_list=axs.reshape(-1)

    for i in tqdm(range(ens_length+1), total=ens_length+1):
        if i>=ens_length-2:
            ax=ax_list[i+1]
        else:
            ax=ax_list[i]
        
        i_long=str(i).zfill(3)
        
        fpath = outpath+"data/"
        fn=fpath+indate+inrun+'00-'+str(t)+'h-enfo-ef_tp_sfc_'+str(i)+'.grib2'
        prcp=xr.open_dataset(fn,engine='cfgrib')
        
        prcp=prcp.sel(latitude=slice(plot_extent[3]+5,plot_extent[2]-5),longitude=slice(plot_extent[0]-5,plot_extent[1]+5))
        if t==24:
            prcp24=prcp*1
        else:
            fn0=fpath+indate+inrun+'00-'+str(t-24)+'h-enfo-ef_tp_sfc_'+str(i)+'.grib2'
            prcp0=xr.open_dataset(fn0,engine='cfgrib')
            prcp24=prcp-prcp0
        
        fn=fpath+indate+inrun+'00-'+str(t)+'h-enfo-ef_msl_sfc_'+str(i)+'.grib2'
        mslp=xr.open_dataset(fn,engine='cfgrib')#,backend_kwargs={"indexpath": home_path+'temp/temp_grib.idx'})
        mslp=mslp.sel(latitude=slice(plot_extent[3]+5,plot_extent[2]-5),longitude=slice(plot_extent[0]-5,plot_extent[1]+5))
        
        dt=datetime.strptime(str(mslp.valid_time.values),'%Y-%m-%dT%H:%M:%S.000000000')
        init_dt=datetime.strptime(str(mslp.time.values),'%Y-%m-%dT%H:%M:%S.000000000')

        dstr=dt.strftime('%Y%m%d_%H')
        dstr_long=dt.strftime('%H%M UTC %d %b %Y')
        dstr_init_long=init_dt.strftime('%H%M UTC %d %b %Y')
        
        rain_levels =  [1,2,5,10,15,20,30,40,50,60,80,100,120,150,200,250,300,400,600,800,1000]
        #rain_levels =  [0.2,0.5,1,2,5,15,20,25,30,35,40,45,50,60,70,80,90,100,125,150,200]
        cmap = nclcmaps.cmap('WhiteBlueGreenYellowRed')(range(26,256,int(np.floor(256/len(rain_levels)+2))))
        cmap = ListedColormap(np.concatenate([cmap,nclcmaps.cmap('MPL_gist_nca')(range(102,129,7))]))
        cf=ax.contourf(prcp24.longitude,prcp24.latitude,prcp24.tp*1000,levels=rain_levels,
                norm=BoundaryNorm(rain_levels,len(rain_levels)),cmap=cmap,extend='max',
                transform=ccrs.PlateCarree())
        plot_levels = list(range(800,1400,4))
        cmslp=ax.contour(mslp.longitude,mslp.latitude,mslp.msl/100,levels=plot_levels,
                         colors='black',linewidths=1,linestyles='-',
                         transform=ccrs.PlateCarree())
        
        ax.add_feature(LAND,facecolor='lightgrey')
        ax.coastlines(linewidths=0.4)
        gl = ax.gridlines(transform=ccrs.PlateCarree(), draw_labels=True,
                          linewidth=0.4, color='lightgrey', alpha=0.5, linestyle='--')
        
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        gl.top_labels = False
        gl.right_labels = False
        gl.bottom_labels = False
        gl.left_labels = False
        
        #if i==0:
        #    ax.set_title('Control', fontsize=12)
        #else:
        ax.set_title(i_long, fontsize=12)
    
    ax_list[-1].set_axis_off()
    ax_list[-2].set_axis_off()
    ax_list[-3].set_axis_off()
    ax_list[-4].set_axis_off()
    ax_list[-8].set_axis_off()
        
    fig.tight_layout(w_pad=0.1)
    fig.subplots_adjust(left=0.065, right=0.97, top=0.93, bottom=0.11, wspace=0.14)

    fig.suptitle('MSLP (black contours) | Precip (24hr accumlation, shaded)\nECMWF eIFS Postage Stamps | Init: ' + dstr_init_long +' | Valid: '+dstr_long, 
              fontsize=20)
    
    ax_pos=[0.575,0.175,0.35,0.02] #[left, bottom, width, height]
    #fig=plt.gcf()
    cbar_ax = fig.add_axes(ax_pos)
    cb=fig.colorbar(cf,orientation="horizontal", cax=cbar_ax)
    cb.ax.tick_params(labelsize=15)
    cb.set_label('Accumulated 24 hour Precipitation [mm]', rotation=0, fontsize=15)
    
    outfile=outpath+'images/ECMWF-eIFS_'+name+'_'+'PostagePrecip24hr_'+str(t)+'.jpg'
    plt.savefig(outfile, dpi=300)
    crop(outfile,in_padding=50)

    check_file_path = outpath+"plot_eIFS_Precip_t"+str(t)+".check"
    with open(check_file_path, 'w') as file:
        file.write("File size check passed!")
