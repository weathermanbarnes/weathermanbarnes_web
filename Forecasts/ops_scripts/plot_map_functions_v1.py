import os
import numpy as np
import xarray as xr
import netCDF4 as nc
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, BoundaryNorm
import matplotlib.path as mpath
theta = np.linspace(0, 2*np.pi, 100)
map_circle = mpath.Path(np.vstack([np.sin(theta), np.cos(theta)]).T * 0.5 + [0.5, 0.5]) #This for the polar stereographic plots
import cartopy.crs as ccrs
from scipy.ndimage import gaussian_filter
from scipy.ndimage.filters import maximum_filter, minimum_filter
from cartopy.feature import NaturalEarthFeature, LAND
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from windspharm.xarray import VectorWind
from windspharm.tools import prep_data, recover_data, order_latdim
from metpy.calc import equivalent_potential_temperature,dewpoint_from_relative_humidity,potential_vorticity_baroclinic
from metpy.calc import potential_temperature,isentropic_interpolation_as_dataset
from metpy.units import units
import pykdtree
import gc

from PIL import Image # pip install Pillow
import sys
import glob
from PIL import ImageOps
import numpy as np
# Trim all png images with white background in a folder
# Usage "python PNGWhiteTrim.py ../someFolder padding"
def crop(path, in_padding=1,pad_type='all',**kwargs):
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

        print(imageBox,imageSize)

        if pad_type=='y-only':
            imageBox=(0,imageBox[1],imageSize[0],imageBox[3])
    
        cropped=image.crop(imageBox)
        print(filePath, "Size:", imageSize, "New Size:", imageBox)
        cropped.save(filePath)

def get_domain_settings(name):
    if name=='Australia':
        plot_extent=[100,180,-60,-5]
        centlon=0
        figsize=(10,8)
        barblength=7
        sublonE=35; sublonW=20
    if name=='SouthernAfrica':
        plot_extent=[-20,60,-60,-5]
        centlon=0
        figsize=(10,8)
        barblength=7
        sublonE=20; sublonW=20
    if name=='SouthAmerica':
        plot_extent=[-105,-15,-60,10]
        centlon=0
        figsize=(11,10)
        barblength=7
        sublonE=20; sublonW=35
    if name=='IndianOcean':
        plot_extent=[40,120,-60,-5]
        centlon=0
        figsize=(10,8)
        barblength=7
        sublonE=20; sublonW=20
    if name=='PacificOcean':
        plot_extent=[170,290,-60,-5]
        centlon=180
        figsize=(15,8)
        barblength=7
        sublonE=20; sublonW=35
    if name=='SH':
        plot_extent=[-180,179.75,-60,-5]
        centlon=130
        figsize=(11,12)
        barblength=5
        sublonE=0; sublonW=0
    if name=='Europe':
        plot_extent=[-50,25,20,70]
        centlon=0
        figsize=(10,8)
        barblength=7
        sublonE=20; sublonW=20
    if name=='NorthAmerica':
        plot_extent=[-140,-60,10,65]
        centlon=0
        figsize=(10,8)
        barblength=7
        sublonE=20; sublonW=20
    if name=='NorthAfrica':
        plot_extent=[-30,60,-5,45]
        centlon=0
        figsize=(10,7)
        barblength=7
        sublonE=20; sublonW=20
    if name=='Asia':
        plot_extent=[50,160,0,70]
        centlon=0
        figsize=(10,8)
        barblength=7
        sublonE=35; sublonW=20
    if name=='NH':
        plot_extent=[-180,179.75,5,60]
        centlon=0
        figsize=(11,12)
        barblength=5
        sublonE=0; sublonW=0
        
    return plot_extent,centlon,figsize,barblength#,[sublonW, sublonE]
        
def plot_maxmin_points1(lon, lat, data, extrema, nsize, symbol, symbol_size=12, color='k',
                       plot_extent=[100,180,-60,-5], plotValue=True, transform=None):
    """
    This function will find and plot relative maximum and minimum for a 2D grid. The function
    can be used to plot an H for maximum values (e.g., High pressure) and an L for minimum
    values (e.g., low pressue). It is best to used filetered data to obtain  a synoptic scale
    max/min value. The symbol text can be set to a string value and optionally the color of the
    symbol and any plotted value can be set with the parameter color
    lon = plotting longitude values (2D)
    lat = plotting latitude values (2D)
    data = 2D data that you wish to plot the max/min symbol placement
    extrema = Either a value of max for Maximum Values or min for Minimum Values
    nsize = Size of the grid box to filter the max and min values to plot a reasonable number
    symbol = String to be placed at location of max/min value
    color = String matplotlib colorname to plot the symbol (and numerica value, if plotted)
    plot_value = Boolean (True/False) of whether to plot the numeric value of max/min point
    The max/min symbol will be plotted on the current axes within the bounding frame
    (e.g., clip_on=True)
    """
    
    ax=plt.gca(projection=transform)
    
    if (extrema == 'max'):
        data_ext = maximum_filter(data, nsize, mode='nearest')
    elif (extrema == 'min'):
        data_ext = minimum_filter(data, nsize, mode='nearest')
    else:
        raise ValueError('Value for hilo must be either max or min')

    mxy, mxx = np.where(data_ext == data)

    for i in range(len(mxy)):
        if lon[mxx[i]]>plot_extent[0]:
            if lon[mxx[i]]<plot_extent[1]:
                if lat[mxy[i]]>plot_extent[2]:
                    if lat[mxy[i]]<plot_extent[3]:
                        ax.text(lon[mxx[i]], lat[mxy[i]], symbol, color=color, size=symbol_size*2,
                                clip_on=True, horizontalalignment='center', verticalalignment='center',
                                transform=transform)
                        if plotValue:
                            ax.text(lon[mxx[i]], lat[mxy[i]],
                                    '\n' + str(np.int(data[mxy[i], mxx[i]])),
                                    color=color, size=symbol_size, clip_on=True, fontweight='bold',
                                    horizontalalignment='center', verticalalignment='top', transform=transform)

def plot_maxmin_points(lon, lat, data, extrema, nsize, symbol, color='k',
                       plotValue=True, transform=None):
    """
    This function will find and plot relative maximum and minimum for a 2D grid. The function
    can be used to plot an H for maximum values (e.g., High pressure) and an L for minimum
    values (e.g., low pressue). It is best to used filetered data to obtain  a synoptic scale
    max/min value. The symbol text can be set to a string value and optionally the color of the
    symbol and any plotted value can be set with the parameter color
    lon = plotting longitude values (2D)
    lat = plotting latitude values (2D)
    data = 2D data that you wish to plot the max/min symbol placement
    extrema = Either a value of max for Maximum Values or min for Minimum Values
    nsize = Size of the grid box to filter the max and min values to plot a reasonable number
    symbol = String to be placed at location of max/min value
    color = String matplotlib colorname to plot the symbol (and numerica value, if plotted)
    plot_value = Boolean (True/False) of whether to plot the numeric value of max/min point
    The max/min symbol will be plotted on the current axes within the bounding frame
    (e.g., clip_on=True)
    """
    
    ax=plt.gca(projection=transform)
    
    if (extrema == 'max'):
        data_ext = maximum_filter(data, nsize, mode='nearest')
    elif (extrema == 'min'):
        data_ext = minimum_filter(data, nsize, mode='nearest')
    else:
        raise ValueError('Value for hilo must be either max or min')

    mxy, mxx = np.where(data_ext == data)

    for i in range(len(mxy)):
        print(lon[mxy[i]])
        ax.text(lon[mxy[i], mxx[i]], lat[mxy[i], mxx[i]], symbol, color=color, size=24,
                clip_on=True, horizontalalignment='center', verticalalignment='center',
                transform=transform)
        ax.text(lon[mxy[i], mxx[i]], lat[mxy[i], mxx[i]],
                '\n' + str(np.int(data[mxy[i], mxx[i]])),
                color=color, size=12, clip_on=True, fontweight='bold',
                horizontalalignment='center', verticalalignment='top', transform=transform)

def plot_precip6h(outpath,dt,init_dt,fignum,data,name='Australia',cbar='on',model_name='GFS',dpi=150,**kwargs):
    dstr=dt.strftime('%Y%m%d_%H')
    dstr_long=dt.strftime('%H%M UTC %d %b %Y')
    dstr_init_long=init_dt.strftime('%H%M UTC %d %b %Y')
    
    plot_extent,centlon,figsize,barblength=get_domain_settings(name)
    
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
    
    rain_levels =  [0.2,0.5,1,2,5,15,20,25,30,35,40,45,50,60,70,80,90,100,125,150,200]
    cmap = plt.cm.get_cmap('gist_ncar', len(rain_levels))
    cf=ax.contourf(data['lons'],data['lats'],data['precip'],levels=rain_levels,
            norm=BoundaryNorm(rain_levels,len(rain_levels)),cmap=cmap,extend='max',
            transform=data_crs)

    nvec=10
    if name=='SH' or name=='NH':
         q=plt.barbs(data['lons'], data['lats'], 
                    data['u10'], data['v10'],
                    length=4, fill_empty=True, pivot='middle',#flagcolor=[.3,.3,.3],barbcolor=[.3,.3,.3],
                    sizes=dict(emptybarb=0.05, spacing=0.2, height=0.3), linewidth=0.6, regrid_shape=40,
                    transform=data_crs)
    else:
        q=plt.barbs(data['lons'][::nvec], data['lats'][::nvec], 
                    data['u10'][::nvec,::nvec], data['v10'][::nvec,::nvec],
                    length=4, fill_empty=True, pivot='middle',#flagcolor=[.3,.3,.3],barbcolor=[.3,.3,.3],
                    sizes=dict(emptybarb=0.05, spacing=0.2, height=0.3),linewidth=0.6,
                    transform=data_crs)
            
    plot_levels = list(range(800,1400,4))
    c=ax.contour(data['lons'],data['lats'],data['mslp'],
                 levels=plot_levels,colors='black',linewidths=1.5,
                 transform=data_crs)
    fmt = '%i'
    ax.clabel(c, c.levels, inline=True, fmt=fmt, fontsize=8)  
    
    ax.add_feature(LAND,facecolor='lightgrey')
    ax.coastlines(linewidths=0.4)
    gl = ax.gridlines(crs=data_crs, draw_labels=True,
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
    
    plt.title('MSLP (black contours) | 10m Wind (arrows) | Precip (accumlated, shaded) \n'+model_name +' Forecast | Init: ' + dstr_init_long +' | Valid: '+dstr_long, 
              fontsize=10)
    if cbar=='on':
        ax_pos=[0.15,0.12,0.7,0.015] #[left, bottom, width, height]
        fig=plt.gcf()
        cbar_ax = fig.add_axes(ax_pos)
        cb=fig.colorbar(cf,orientation="horizontal", cax=cbar_ax, ticks=[int(r) for r in rain_levels])
        cb.ax.tick_params(labelsize=8)
        cb.set_label('Accumulated Rainfall [ mm ]', rotation=0, fontsize=8)

    copywrite_text='\xa9 Michael A. Barnes\nwww.weathermanbarnes.com'
    ax.text(0.01, 0.015, copywrite_text, fontsize=6, 
             horizontalalignment='left', verticalalignment='bottom', 
             transform=ax.transAxes,
             bbox=dict(facecolor='white', alpha=0.95))

    outfile=outpath+model_name+'_'+name+'_Precip6H_'+str(fignum)+'.jpg'
    plt.savefig(outfile, dpi=dpi)
    crop(outfile)
    plt.close('all') 

def plot_QvectPTE(outpath,dt,init_dt,fignum,data,name='Australia',model_name='GFS',dpi=150,**kwargs):
    
    plot_extent,centlon,figsize,barblength=get_domain_settings(name)
    
    dstr=dt.strftime('%Y%m%d_%H')
    dstr_long=dt.strftime('%H%M UTC %d %b %Y')
    dstr_init_long=init_dt.strftime('%H%M UTC %d %b %Y')
    
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
    
    jet = plt.get_cmap('jet')
    purple_shades = np.array([
        [108/256,27/256,67/256,1],
        [89/256,35/256,92/256,1],  # Dark purple
        [157/256,78/256,161/256,1],  # Medium purple
        [247/256,219/256,249/256,1]])  # Light purple
    purple_interpolated = np.linspace(purple_shades[0], purple_shades[-1], 100)
    jet_colors = jet(np.linspace(0, 1, 256))
    new_colors = np.vstack((jet_colors, purple_interpolated))
    jet_purple_cmap = LinearSegmentedColormap.from_list('jet_with_purple', new_colors)

    plot_levels = np.arange(250,376,6)
    cf=ax.contourf(data['lons'],data['lats'],data['pte850'],
                   levels=plot_levels,cmap=jet_purple_cmap,extend='both',
                   transform=data_crs)
    
    plot_levels = list(range(1000,3000,30))
    c=ax.contour(data['lons'],data['lats'],data['z850'],levels=plot_levels,colors='black',
                 linewidths=1.5,transform=data_crs)
    fmt = '%i'
    ax.clabel(c, c.levels, inline=True, fmt=fmt, fontsize=8)  

    plot_levels = [-1,-0.75,-0.5]
    c1=ax.contour(data['lons'],data['lats'],data['wMID'],levels=plot_levels,
                   colors='grey',linestyles='-',
                   transform=data_crs)
    
    nvec=10
    if name=='SH' or name=='NH':
        q=ax.quiver(data['lons'], data['lats'], 
                     data['uqvect850'], data['vqvect850'],
                    angles="xy", regrid_shape=40,
                    transform=data_crs, zorder=1e2)
    else:
        q=ax.quiver(data['lons'][::nvec], data['lats'][::nvec], 
                     data['uqvect850'][::nvec,::nvec], data['vqvect850'][::nvec,::nvec],
                      scale=5e-13,scale_units='xy',width=0.005,minshaft=2,# minlength=1,
                      transform=data_crs,zorder=1e2)
                  
    ax.add_feature(LAND,facecolor='lightgrey')
    #if name=='SH' or name=='NH':
    #    ax.coastlines(linewidths=1)#, transform=data_crs)
    #else:
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
    
    plt.title('850hPa GPH (black contours) | 850hPa Q-vectors (quivers) | 400-700hPa Up. Vert. Vel. (grey contours) | 850hPa Equiv. Pot. Temp (shading)\n'+model_name +' Forecast | Init: ' + dstr_init_long +' | Valid: '+dstr_long, 
              fontsize=10)
    
    ax_pos=[0.25,0.12,0.5,0.015] #[left, bottom, width, height]
    fig=plt.gcf()
    cbar_ax = fig.add_axes(ax_pos)
    cb=fig.colorbar(cf,orientation="horizontal", cax=cbar_ax)
    cb.ax.tick_params(labelsize=8)
    cb.set_label('Equivalent Potential Temperature [K]', rotation=0, fontsize=8)

    copywrite_text='\xa9 Michael A. Barnes\nwww.weathermanbarnes.com'
    ax.text(0.01, 0.015, copywrite_text, fontsize=6, 
             horizontalalignment='left', verticalalignment='bottom', 
             transform=ax.transAxes,
             bbox=dict(facecolor='white', alpha=0.95),zorder=1e3)
    
    outfile=outpath+model_name+'_'+name+'_QvecPTE850_'+str(fignum)+'.jpg'
    plt.savefig(outfile, dpi=dpi)
    crop(outfile,padding=30)
    plt.close('all')

def plot_upper(outpath,dt,init_dt,fignum,data,name='Australia',model_name='GFS',dpi=150,**kwargs):
    
    plot_extent,centlon,figsize,barblength=get_domain_settings(name)
    
    dstr=dt.strftime('%Y%m%d_%H')
    dstr_long=dt.strftime('%H%M UTC %d %b %Y')
    dstr_init_long=init_dt.strftime('%H%M UTC %d %b %Y')
    
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
    plot_levels = [50,60,70,80,90,100,120,140,160]
    cf=ax.contourf(data['lons'],data['lats'],data['jet300']*1.94384,
                   levels=plot_levels,cmap='Blues',extend='max',
                   transform=data_crs)
    
    plot_levels = list(range(800,1400,4))
    cmslp=ax.contour(data['lons'],data['lats'],data['mslp'],
                     levels=plot_levels,colors='grey',linewidths=1,linestyles='--',
                     transform=data_crs)
    
    #plot_levels = [-30e-1 -25e-1 -20e-1 -15e-1, -10e-1, -5e-1]
    plot_levels = [-0.5,-0.4,-0.3,-0.2]
    c1=ax.contourf(data['lons'],data['lats'],data['wMID'],levels=plot_levels,
                   cmap='Reds_r',extend='min',
                   transform=data_crs)
    plot_levels = list(range(4000,6100,50))
    
    c=ax.contour(data['lons'],data['lats'],data['z500'],levels=plot_levels,colors='black',
                 linewidths=0.8,transform=data_crs)
    nvec=10
    if name=='SH' or name=='NH':
        q=ax.quiver(data['lons'], data['lats'], 
                     data['ujet300'], data['vjet300'],
                    angles="xy", transform=data_crs, regrid_shape=40)
    else:
        q=ax.quiver(data['lons'][::nvec], data['lats'][::nvec], 
                     data['ujet300'][::nvec,::nvec], data['vjet300'][::nvec,::nvec],
                      scale=15,scale_units='xy',width=0.005,minshaft=2,# minlength=1,
                      transform=data_crs)
                  
    ax.add_feature(LAND,facecolor='lightgrey')
    ax.coastlines(linewidths=0.4)#, transform=data_crs)
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
    
    # cb=plt.colorbar(cf,orientation="horizontal")
    plt.title('500hPa GPH (black contours) | 400-700hPa Vert. Vel. (red shading) | MSLP (grey dashed) | 300hPa Jet (blue shading)\n'+model_name +' Forecast | Init: ' + dstr_init_long +' | Valid: '+dstr_long, 
              fontsize=10)
    
    ax_pos=[0.25,0.12,0.5,0.015] #[left, bottom, width, height]
    fig=plt.gcf()
    cbar_ax = fig.add_axes(ax_pos)
    cb=fig.colorbar(cf,orientation="horizontal", cax=cbar_ax)
    cb.ax.tick_params(labelsize=8)
    cb.set_label('Wind Speed [m/s]', rotation=0, fontsize=8)

    copywrite_text='\xa9 Michael A. Barnes\nwww.weathermanbarnes.com'
    ax.text(0.01, 0.015, copywrite_text, fontsize=6, 
             horizontalalignment='left', verticalalignment='bottom', 
             transform=ax.transAxes,
             bbox=dict(facecolor='white', alpha=0.95))
    
    outfile=outpath+model_name+'_'+name+'_UpperLevel_'+str(fignum)+'.jpg'
    plt.savefig(outfile, dpi=dpi)
    crop(outfile,padding=30)
    plt.close('all') 
    
def plot_irrotPV(outpath,dt,init_dt,fignum,data,name='Australia',model_name='GFS',dpi=150,**kwargs):
    
    plot_extent,centlon,figsize,barblength=get_domain_settings(name)
    
    dstr=dt.strftime('%Y%m%d_%H')
    dstr_long=dt.strftime('%H%M UTC %d %b %Y')
    dstr_init_long=init_dt.strftime('%H%M UTC %d %b %Y')
    
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
    
    plot_levels = list(range(20,80,5))
    cf_pw=ax.contourf(data['lons'],data['lats'],data['pwat'],
                   levels=plot_levels,cmap='GnBu',extend='max',
                   transform=data_crs)
    
    jet_cmap = plt.cm.get_cmap('RdPu', 16)(range(5,16))
    jet_cmap = ListedColormap(jet_cmap)
    plot_levels = [30,35,40,45,50,60,70,80,90,100,120]
    cf=ax.contourf(data['lons'],data['lats'],data['jet300'],
                   levels=plot_levels,cmap=jet_cmap,extend='max',
                   transform=data_crs)

    plot_levels = [-1,-0.75,-0.5]
    c1=ax.contour(data['lons'],data['lats'],data['wMID'],levels=plot_levels,
                   colors='red',linestyles='-',
                   transform=data_crs)

    
    if name=='SH' or name=='SouthernAfrica' or name=='SouthAmerica' or name=='Australia' or name=='IndianOcean':
        plot_levels = list(range(-15,0,1))
    else:
        plot_levels = list(range(1,16,1))
    cpv=ax.contour(data['lons'],data['lats'],gaussian_filter(data['pv_iso_upper']*1e6,sigma=2),
                     levels=plot_levels,colors='black',linestyles='-',
                     linewidths=1.5,transform=data_crs)
    fmt = '%i'
    ax.clabel(cpv, cpv.levels[1::2], inline=True, fmt=fmt, fontsize=14)  


    nvec=10
    if name=='SH' or name=='NH':
        q=plt.quiver(data['lons'], data['lats'], 
                 data['uchi'], data['vchi'],
                 angles="xy", transform=data_crs, regrid_shape=40)
    else:
        q=plt.quiver(data['lons'][::nvec],data['lats'][::nvec], 
                 data['uchi'][::nvec,::nvec],
                 data['vchi'][::nvec,::nvec],
                 scale=2,scale_units='xy',width=0.005,minshaft=2,#minlength=2,
                 transform=ccrs.PlateCarree())
                  
    ax.add_feature(LAND,facecolor='lightgrey')
    ax.coastlines(linewidths=0.4,zorder=1e4)
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
    
    plt.title('200-300hPa PV (black contours) | Irrot. Wind (quiver) | 400-700hPa Up. Vert. Vel. (red contours) | 300hPa Jet (magenta shading)\n'+model_name +' Forecast | Init: ' + dstr_init_long +' | Valid: '+dstr_long, 
              fontsize=10)
    
    ax_pos=[0.525,0.12,0.35,0.015] #[left, bottom, width, height]
    fig=plt.gcf()
    cbar_ax = fig.add_axes(ax_pos)
    cb=fig.colorbar(cf,orientation="horizontal", cax=cbar_ax)
    cb.ax.tick_params(labelsize=8)
    cb.set_label('Wind Speed [m/s]', rotation=0, fontsize=8)
    
    ax_pos=[0.15,0.12,0.35,0.015] #[left, bottom, width, height]
    fig=plt.gcf()
    cbar_ax = fig.add_axes(ax_pos)
    cb=fig.colorbar(cf_pw,orientation="horizontal", cax=cbar_ax)
    cb.ax.tick_params(labelsize=8)
    cb.set_label('Precip. Water [mm]', rotation=0, fontsize=8)

    copywrite_text='\xa9 Michael A. Barnes\nwww.weathermanbarnes.com'
    ax.text(0.01, 0.015, copywrite_text, fontsize=6, 
             horizontalalignment='left', verticalalignment='bottom', 
             transform=ax.transAxes,
             bbox=dict(facecolor='white', alpha=0.95),
             zorder=1e3)
    
    outfile=outpath+model_name+'_'+name+'_IrrotPV_'+str(fignum)+'.jpg'
    plt.savefig(outfile, dpi=dpi)
    crop(outfile)
    plt.close('all') 
    
def plot_IVT(outpath,dt,init_dt,fignum,data,name='Australia',model_name='GFS',dpi=150,**kwargs):
    
    plot_extent,centlon,figsize,barblength=get_domain_settings(name)
    
    dstr=dt.strftime('%Y%m%d_%H')
    dstr_long=dt.strftime('%H%M UTC %d %b %Y')
    dstr_init_long=init_dt.strftime('%H%M UTC %d %b %Y')
    
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
    
    plot_levels = [250,300,400,500,600,700,800,1000,1200,1400,1600,1800]
    ivt_cmap = plt.cm.get_cmap('cool', len(plot_levels))
    cf=ax.contourf(data['lons'],data['lats'],data['IVT'],levels=plot_levels,
            norm=BoundaryNorm(plot_levels,len(plot_levels)),cmap=ivt_cmap,extend='max',
            transform=data_crs)
    c=ax.contour(data['lons'],data['lats'],data['IVT'],levels=plot_levels,colors='grey',
                 linewidths=0.4,transform=data_crs)
    nvec=10
    if name=='SH' or name=='NH':
        q=ax.quiver(data['lons'], data['lats'], 
                     data['uIVT'], data['vIVT'],
                    angles="xy", transform=data_crs, regrid_shape=40)
    else:
        q=plt.quiver(data['lons'][::nvec], data['lats'][::nvec], 
                     data['uIVT'][::nvec,::nvec], data['vIVT'][::nvec,::nvec],\
                     scale=250,scale_units='xy',width=0.0105,minshaft=2,minlength=0.01,
                     transform=data_crs)
    plot_levels = list(range(0,6000,30))
    c=ax.contour(data['lons'],data['lats'],data['z700'],levels=plot_levels,
                 colors='black',linewidths=0.6,
                 transform=data_crs)
    fmt = '%i'
    ax.clabel(c, c.levels, inline=True, fmt=fmt, fontsize=8)
    
    ax.add_feature(LAND,facecolor='lightgrey')
    ax.coastlines(linewidths=0.4)
    gl = ax.gridlines(crs=data_crs, draw_labels=True,
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

    plt.title('700hPa GPH (black contours) | IVT (shaded, arrows)\n'+model_name +' Forecast | Init: ' + dstr_init_long +' | Valid: '+dstr_long, 
              fontsize=10)
    
    ax_pos=[0.25,0.12,0.5,0.015] #[left, bottom, width, height]
    fig=plt.gcf()
    cbar_ax = fig.add_axes(ax_pos)
    cb=fig.colorbar(cf,orientation="horizontal", cax=cbar_ax)
    cb.ax.tick_params(labelsize=8)
    cb.set_label('IVT', rotation=0, fontsize=8)

    copywrite_text='\xa9 Michael A. Barnes\nwww.weathermanbarnes.com'
    ax.text(0.01, 0.015, copywrite_text, fontsize=6, 
             horizontalalignment='left', verticalalignment='bottom', 
             transform=ax.transAxes,
             bbox=dict(facecolor='white', alpha=0.95))
    
    outfile=outpath+model_name+'_'+name+'_IVT_'+str(fignum)+'.jpg'
    plt.savefig(outfile, dpi=dpi)
    crop(outfile)
    plt.close('all') 
    
def plot_IPV(outpath,dt,init_dt,fignum,plot_level,data,name='Australia',model_name='GFS',dpi=150,**kwargs):
    
    plot_extent,centlon,figsize,barblength=get_domain_settings(name)
    
    dstr=dt.strftime('%Y%m%d_%H')
    dstr_long=dt.strftime('%H%M UTC %d %b %Y')
    dstr_init_long=init_dt.strftime('%H%M UTC %d %b %Y')

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

    if name=='SH' or name=='SouthernAfrica' or name=='SouthAmerica' or name=='Australia' or name=='IndianOcean':
        pv_clevs=[-12,-10,-8,-6,-4,-3,-2,-1.5,-1,-0.5]
        pv_cmap = plt.cm.get_cmap('RdPu_r', 10)(range(0,6))
        pv_cmap = np.concatenate((pv_cmap,plt.cm.get_cmap('Blues_r', 6)(range(2,5))))
        #pv_cmap = np.concatenate((pv_cmap,plt.cm.get_cmap('Greys', 255)(range(0,1))))
        pv_cmap = ListedColormap(pv_cmap)
        PVextend='min'
    else:
        pv_clevs=[0.5,1,1.5,2,3,4,6,8,10,12]
        pv_cmap = plt.cm.get_cmap('RdPu_r', 10)(range(0,6))
        pv_cmap = np.concatenate((pv_cmap,plt.cm.get_cmap('Blues_r', 6)(range(2,5))))
        #pv_cmap = np.concatenate((pv_cmap,plt.cm.get_cmap('Greys', 255)(range(0,1))))
        pv_cmap = ListedColormap(pv_cmap[::-1])
        PVextend='max'
    
    cf=plt.contourf(data['pv'+str(plot_level)].longitude,data['pv'+str(plot_level)].latitude,
                    gaussian_filter(data['pv'+str(plot_level)]/1e-6,sigma=1),
                    levels=pv_clevs,cmap=pv_cmap,norm=BoundaryNorm(pv_clevs,len(pv_clevs)),
                    transform=data_crs,extend=PVextend)
    c=ax.contour(data['pv'+str(plot_level)].longitude,data['pv'+str(plot_level)].latitude,
                 gaussian_filter(data['pv'+str(plot_level)]/1e-6,sigma=1),
                 levels=pv_clevs[0:-1],colors='grey',linewidths=0.4,
                 transform=data_crs)
    
    nvec=10
    if name=='SH' or name=='NH':
        q=ax.quiver(data['u'+str(plot_level)].longitude,data['u'+str(plot_level)].latitude, 
                    data['u'+str(plot_level)],data['v'+str(plot_level)],
                    angles="xy", transform=data_crs, regrid_shape=40)
    else:
        q=plt.quiver(data['u'+str(plot_level)].longitude[::nvec], data['u'+str(plot_level)].latitude[::nvec], 
                     data['u'+str(plot_level)][::nvec,::nvec], data['v'+str(plot_level)][::nvec,::nvec],\
                     scale=25,scale_units='xy',width=0.0205,minshaft=1.3,minlength=0.01,
                     transform=data_crs)
    
    ax.add_feature(LAND,facecolor='lightgrey')
    ax.coastlines(linewidths=1)
    gl = ax.gridlines(crs=data_crs, draw_labels=True,
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

    plt.title(str(plot_level)+'K PV (shaded) and wind (quivers)\n'+model_name +' Forecast | Init: ' + dstr_init_long +' | Valid: '+dstr_long,
              fontsize=10)
    
    ax_pos=[0.25,0.08,0.5,0.02] #[left, bottom, width, height]
    fig=plt.gcf()
    cbar_ax = fig.add_axes(ax_pos)
    cb=fig.colorbar(cf,orientation="horizontal", cax=cbar_ax)
    cb.ax.tick_params(labelsize=8)
    cb.set_label('Potential Vorticity [PVU]', rotation=0, fontsize=8)

    copywrite_text='\xa9 Michael A. Barnes\nwww.weathermanbarnes.com'
    ax.text(0.01, 0.015, copywrite_text, fontsize=6, 
             horizontalalignment='left', verticalalignment='bottom', 
             transform=ax.transAxes,
             bbox=dict(facecolor='white', alpha=0.95))
    
    outfile=outpath+model_name+'_'+name+'_IPV-'+str(plot_level)+'K_'+str(fignum)+'.jpg'
    plt.savefig(outfile, dpi=dpi)
    crop(outfile)
    #plt.clf()
    plt.show()
    plt.close('all') 

def plot_thickness(outpath,dt,init_dt,fignum,data,name='Australia',model_name='GFS',dpi=150,**kwargs):

    plot_extent,centlon,figsize,barblength=get_domain_settings(name)
    
    dstr=dt.strftime('%Y%m%d_%H')
    dstr_long=dt.strftime('%H%M UTC %d %b %Y')
    dstr_init_long=init_dt.strftime('%H%M UTC %d %b %Y')
    
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

    if name=='SH' or name=='SouthernAfrica' or name=='SouthAmerica' or name=='Australia' or name=='IndianOcean':
        plot_levels = np.arange(-30,-4,2)
    else:
        plot_levels = np.arange(6,32,2)#[250,300,400,500,600,700,800,1000,1200,1400,1600,1800]
    cmap = plt.cm.get_cmap('BuPu_r', len(plot_levels))
    cf=ax.contourf(data['lons'],data['lats'],data['vort500']*1e5,levels=plot_levels,
            norm=BoundaryNorm(plot_levels,len(plot_levels)),cmap=cmap,extend='min',
            transform=data_crs)
    
    plot_levels = list(range(4000,6100,50))
    c=ax.contour(data['lons'],data['lats'],data['z500'],levels=plot_levels,colors='black',
                 linewidths=0.8,transform=data_crs)
    
    plot_levels = [-0.8,-0.7,-0.6,-0.5,-0.4,-0.3]
    c1=ax.contour(data['lons'],data['lats'],data['wMID'],
                   levels=plot_levels,colors='darkorange',linewidths=0.8,linestyles='-',
                   transform=data_crs)
    
    plot_levels = list(range(546,700,6))
    c=ax.contour(data['lons'],data['lats'],data['thickness'],
                 levels=plot_levels,colors='red',
                 linestyles='--',linewidths=1.5,transform=data_crs)
    fmt = '%i'
    ax.clabel(c, c.levels, inline=True, fmt=fmt, fontsize=10, inline_spacing=0)
    plot_levels = list(range(546-120,546,6))
    c=ax.contour(data['lons'],data['lats'],data['thickness'],
                 levels=plot_levels,colors='blue',
                 linestyles='--',linewidths=1.5,transform=data_crs)
    fmt = '%i'
    ax.clabel(c, c.levels, inline=True, fmt=fmt, fontsize=10, inline_spacing=0)
    
    ax.add_feature(LAND,facecolor='lightgrey')
    ax.coastlines(linewidths=0.4)
    gl = ax.gridlines(crs=data_crs, draw_labels=True,
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
    
    plt.title('500-1000hPa Thickness (red/blue, dam) | 500hPa Cyc. Rel. Vort. (shading) | 500hPa GPH (black) | 400-700hPa Vert. Vel. (orange)\n'+model_name +' Forecast | Init: ' + dstr_init_long +' | Valid: '+dstr_long, 
              fontsize=10)
    
    ax_pos=[0.25,0.12,0.5,0.015] #[left, bottom, width, height]
    fig=plt.gcf()
    cbar_ax = fig.add_axes(ax_pos)
    cb=fig.colorbar(cf,orientation="horizontal", cax=cbar_ax)
    cb.ax.tick_params(labelsize=8)
    cb.set_label('Cyclonic Relative Vorticity [$10^{-5}$]', rotation=0, fontsize=8)
    
    copywrite_text='\xa9 Michael A. Barnes\nwww.weathermanbarnes.com'
    ax.text(0.01, 0.015, copywrite_text, fontsize=6, 
             horizontalalignment='left', verticalalignment='bottom', 
             transform=ax.transAxes,
             bbox=dict(facecolor='white', alpha=0.95),zorder=10)

    outfile=outpath+model_name+'_'+name+'_Thickness_'+str(fignum)+'.jpg'
    plt.savefig(outfile, dpi=dpi)
    crop(outfile)
    plt.close('all') 

def plot_DT(outpath,dt,init_dt,fignum,data,name='Australia',model_name='GFS',dpi=150,**kwargs):
    
    plot_extent,centlon,figsize,barblength=get_domain_settings(name)
    
    dstr=dt.strftime('%Y%m%d_%H')
    dstr_long=dt.strftime('%H%M UTC %d %b %Y')
    dstr_init_long=init_dt.strftime('%H%M UTC %d %b %Y')
    
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
    
    plot_levels = np.arange(260,400,10)#[250,300,400,500,600,700,800,1000,1200,1400,1600,1800]
    dtPTsmooth=gaussian_filter(data['dtPT'],sigma=1)
    cmap = plt.cm.get_cmap('plasma', len(plot_levels))
    cf=ax.contourf(data['lons'],data['lats'],dtPTsmooth,levels=plot_levels,
            linestyles='-',linewidths=0.8,
            norm=BoundaryNorm(plot_levels,len(plot_levels)),cmap=cmap,extend='both',
            transform=data_crs)

    if name=='SH' or name=='SouthernAfrica' or name=='SouthAmerica' or name=='Australia' or name=='IndianOcean':
        plot_levels = np.arange(-7,-4.5,0.5)
    else:
        plot_levels = np.arange(5,7.5,0.5)
    c=ax.contour(data['lons'],data['lats'],data['vortLOW']*1e5,levels=plot_levels,
            colors='black',linestyles='-',linewidths=0.8,
            #norm=BoundaryNorm(plot_levels,len(plot_levels)),cmap=cmap,extend='min',
            transform=data_crs)
    
    ax.add_feature(LAND,facecolor='lightgrey')
    ax.coastlines(linewidths=0.4)
    gl = ax.gridlines(crs=data_crs, draw_labels=True,
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
    
    plt.title('2PVU Potential Temperature (shading) | 850-925hPa Cyc. Rel. Vort. (black)\n'+model_name +' Forecast | Init: ' + dstr_init_long +' | Valid: '+dstr_long, 
              fontsize=10)
    
    ax_pos=[0.25,0.12,0.5,0.015] #[left, bottom, width, height]
    fig=plt.gcf()
    cbar_ax = fig.add_axes(ax_pos)
    cb=fig.colorbar(cf,orientation="horizontal", cax=cbar_ax)
    cb.ax.tick_params(labelsize=8)
    cb.set_label('Potentia Temperature [K]', rotation=0, fontsize=8)
    
    copywrite_text='\xa9 Michael A. Barnes\nwww.weathermanbarnes.com'
    ax.text(0.01, 0.015, copywrite_text, fontsize=6, 
             horizontalalignment='left', verticalalignment='bottom', 
             transform=ax.transAxes,
             bbox=dict(facecolor='white', alpha=0.95),zorder=10)
    
    outfile=outpath+model_name+'_'+name+'_DT_'+str(fignum)+'.jpg'
    plt.savefig(outfile, dpi=dpi)
    crop(outfile)
    plt.close('all') 






