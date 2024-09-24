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

name='Australia'
plot_extent=[80,180,-60,-5]
centlon=0
figsize=(30,17)
barblength=7
ens_length=50

inrun=indatetime.strftime('%H%M')
indate=indatetime.strftime('%Y%m%d')

trange=[timestep] #range(timestep,timestep+24,24)
for t in trange:
    fig, axs = plt.subplots(7, 8, figsize=figsize, subplot_kw=dict(projection=ccrs.PlateCarree())) 
    ax_list=axs.reshape(-1)

    #jet_cmap = ListedColormap(jet_cmap)  
    for i in tqdm(range(ens_length+1), total=ens_length+1):
        if i>=ens_length-2:
            ax=ax_list[i+1]
        else:
            ax=ax_list[i]
        
        i_long=str(i).zfill(3)
        
        ax.set_extent(plot_extent,crs=ccrs.PlateCarree())
        
        fpath = outpath+"data/"
        fn=fpath+indate+inrun+'00-'+str(t)+'h-enfo-ef_gh_500_'+str(i)+'.grib2'
        z500=xr.open_dataset(fn,engine='cfgrib')#,backend_kwargs={"indexpath": home_path+'temp/temp_grib.idx'})
        z500=z500.sel(latitude=slice(plot_extent[3]+5,plot_extent[2]-5),longitude=slice(plot_extent[0]-5,plot_extent[1]+5))
        
        fn=fpath+indate+inrun+'00-'+str(t)+'h-enfo-ef_u_300_'+str(i)+'.grib2'
        u300=xr.open_dataset(fn,engine='cfgrib')#,backend_kwargs={"indexpath": home_path+'temp/temp_grib.idx'})
        u300=u300.sel(latitude=slice(plot_extent[3]+5,plot_extent[2]-5),longitude=slice(plot_extent[0]-5,plot_extent[1]+5))
        
        fn=fpath+indate+inrun+'00-'+str(t)+'h-enfo-ef_v_300_'+str(i)+'.grib2'
        v300=xr.open_dataset(fn,engine='cfgrib')#,backend_kwargs={"indexpath": home_path+'temp/temp_grib.idx'})
        v300=v300.sel(latitude=slice(plot_extent[3]+5,plot_extent[2]-5),longitude=slice(plot_extent[0]-5,plot_extent[1]+5))
        
        jet300=mpcalc.wind_speed(u300.u, v300.v).data
        
        fn=fpath+indate+inrun+'00-'+str(t)+'h-enfo-ef_msl_sfc_'+str(i)+'.grib2'
        mslp=xr.open_dataset(fn,engine='cfgrib')#,backend_kwargs={"indexpath": home_path+'temp/temp_grib.idx'})
        mslp=mslp.sel(latitude=slice(plot_extent[3]+5,plot_extent[2]-5),longitude=slice(plot_extent[0]-5,plot_extent[1]+5))

        dt=datetime.strptime(str(z500.valid_time.values),'%Y-%m-%dT%H:%M:%S.000000000')
        init_dt=datetime.strptime(str(z500.time.values),'%Y-%m-%dT%H:%M:%S.000000000')

        dstr=dt.strftime('%Y%m%d_%H')
        dstr_long=dt.strftime('%H%M UTC %d %b %Y')
        dstr_init_long=init_dt.strftime('%H%M UTC %d %b %Y')
        
        plot_levels = [50,60,70,80,90,100,120,140,160]
        cf=ax.contourf(u300.longitude,u300.latitude,jet300*1.94384,
                       levels=plot_levels,cmap='Blues',extend='max',
                       transform=ccrs.PlateCarree())

        plot_levels = list(range(800,1400,4))
        cmslp=ax.contour(mslp.longitude,mslp.latitude,mslp.msl/100,levels=plot_levels,
                         colors='grey',linewidths=1,linestyles='--',
                         transform=ccrs.PlateCarree())
        
        plot_levels = list(range(4000,6100,50))
        c=ax.contour(z500.longitude,z500.latitude,z500.gh,levels=plot_levels,
                     colors='black',linewidths=0.8,
                     transform=ccrs.PlateCarree())
        
        nvec=10
        q=ax.quiver(u300['longitude'][::nvec], u300['latitude'][::nvec], 
                     u300['u'][::nvec,::nvec], v300['v'][::nvec,::nvec],
                      scale=15,scale_units='xy',width=0.005,minshaft=1.5,
                      transform=ccrs.PlateCarree())

        ax.add_feature(LAND,facecolor='lightgrey')
        ax.coastlines(linewidths=0.4)
        gl = ax.gridlines(transform=ccrs.PlateCarree(), draw_labels=True,
                          linewidth=0.4, color='lightgrey', alpha=0.5, linestyle='--')
        #xlocators=list(np.arange(-180,190,10))
        #gl.xlocator = mticker.FixedLocator(xlocators)
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
    #ax_list[-5].set_axis_off()

    fig.tight_layout(pad=0.4)
    fig.subplots_adjust(left=0.065, right=0.97, top=0.94, bottom=0.11, wspace=0.14)

    fig.suptitle('500hPa GPH (black contours) | MSLP (grey dashed) | 300hPa Jet (blue shading)\nECMWF eIFS Postage Stamps | Init: ' + dstr_init_long +' | Valid: '+dstr_long, 
              fontsize=20)
    
    ax_pos=[0.575,0.175,0.35,0.02] #[left, bottom, width, height]
    #fig=plt.gcf()
    cbar_ax = fig.add_axes(ax_pos)
    cb=fig.colorbar(cf,orientation="horizontal", cax=cbar_ax)
    cb.ax.tick_params(labelsize=15)
    cb.set_label('Wind Speed [m/s]', rotation=0, fontsize=15)
    
    outfile=outpath+'images/ECMWF-eIFS_'+name+'_'+'PostageUpper_'+str(t)+'.jpg'
    plt.savefig(outfile, dpi=300)
    crop(outfile,in_padding=50)
    #plt.show()

    check_file_path = outpath+"plot_eIFS_Upper_t"+str(t)+".check"
    with open(check_file_path, 'w') as file:
        file.write("File size check passed!")