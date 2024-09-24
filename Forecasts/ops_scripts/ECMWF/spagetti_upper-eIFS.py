import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.path as mpath
theta = np.linspace(0, 2*np.pi, 100)
map_circle = mpath.Path(np.vstack([np.sin(theta), np.cos(theta)]).T * 0.5 + [0.5, 0.5]) #This for the polar stereographic plots
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

from plot_map_functions import *

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

model_name='ECMWF-eIFS'
names=['NH','SH','Australia']

ens_length=50

inrun=indatetime.strftime('%H%M')
indate=indatetime.strftime('%Y%m%d')

trange=[timestep] #range(timestep,timestep+24,24)

for name in names:
    plot_extent,centlon,figsize,barblength=get_domain_settings(name)
    if name=='SH' or name=='NH':
        figsize=(20,13)
    else:
        figsize=(12,5)
    for t in trange:
        
        fig=plt.figure(figsize=figsize)
        if name=='SH':
            proj=ccrs.SouthPolarStereo(central_longitude=centlon)
        elif name=='NH':
            proj=ccrs.NorthPolarStereo()
        else: 
            proj=ccrs.PlateCarree(central_longitude=centlon)
        data_crs=ccrs.PlateCarree()
        
        ax=plt.axes(projection=proj)
        ax.set_extent(plot_extent,crs=data_crs)
        if name=='SH' or name=='NH':
            ax.set_boundary(map_circle, transform=ax.transAxes)
    
        z500_full=[]
        for i in tqdm(range(ens_length+1), total=ens_length):
            
            fpath = outpath+"data/"
            
            fn=fpath+indate+inrun+'00-'+str(t)+'h-enfo-ef_gh_500_'+str(i)+'.grib2'
            z500=xr.open_dataset(fn,engine='cfgrib')#,backend_kwargs={"indexpath": home_path+'temp/temp_grib.idx'})
            #z500=z500.sel(latitude=slice(plot_extent[3]+5,plot_extent[2]-5),longitude=slice(plot_extent[0]-5,plot_extent[1]+5))
            z500_full.append(z500)
            
            plot_levels = [5280,5520,5760]#list(range(4000,6100,50))
            c=ax.contour(z500.longitude,z500.latitude,z500.gh,levels=plot_levels,
                         colors='grey',linewidths=0.8,
                         transform=ccrs.PlateCarree())
    
    
        z500mean=xr.concat(z500_full,dim='number').mean(dim='number')
        c=ax.contour(z500.longitude,z500.latitude,z500.gh,levels=plot_levels,
                         colors='black',linewidths=1.2,
                         transform=ccrs.PlateCarree())
        
        dt=datetime.strptime(str(z500.valid_time.values),'%Y-%m-%dT%H:%M:%S.000000000')
        init_dt=datetime.strptime(str(z500.time.values),'%Y-%m-%dT%H:%M:%S.000000000')
    
        dstr=dt.strftime('%Y%m%d_%H')
        dstr_long=dt.strftime('%H%M UTC %d %b %Y')
        dstr_init_long=init_dt.strftime('%H%M UTC %d %b %Y')
        
        ax.coastlines(linewidths=1)#, transform=data_crs)
        gl = ax.gridlines(transform=data_crs, draw_labels=True,
                          linewidth=0.4, color='lightgrey', alpha=0.5, linestyle='--')
        if name=='SH' or name=='NH':
            xlocators=list(np.arange(-180,190,30))
        else:
            xlocators=list(np.arange(-180,190,10))
        gl.xlocator = mticker.FixedLocator(xlocators)
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        gl.top_labels = False
        gl.right_labels = False
        if name=='SH' or name=='NH':
            gl.bottom_labels = False 
        gl.xlabel_style = gl.ylabel_style ={'size': 8, 'color': 'black'}
        
        plt.title('500hPa GPH (5280, 5520 and 5760 gpm) | Ensemble members (grey contours) | Ensemble mean (black contours) \n'+model_name.replace("-", " ") +' Forecast | Init: ' + dstr_init_long +' | Valid: '+dstr_long, 
                  fontsize=10)
    
        copywrite_text='\xa9 Michael A. Barnes\nwww.weathermanbarnes.com'
        ax.text(0.01, 0.015, copywrite_text, fontsize=6, 
                 horizontalalignment='left', verticalalignment='bottom', 
                 transform=ax.transAxes,
                 bbox=dict(facecolor='white', alpha=0.95),zorder=1e3)
        
        outfile=outpath+'images/'+model_name+'_'+name+'_Spagetti500hPa_'+str(timestep)+'.jpg'
        plt.savefig(outfile, dpi=300)
        crop(outfile,padding=30,pad_type='y-only')
        plt.close('all') 