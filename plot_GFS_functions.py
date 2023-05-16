import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '/home/ubuntu/testing/functions')
import os
import numpy as np
import xarray as xr
import netCDF4 as nc
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, BoundaryNorm
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
import wind_spd_dir as wind
import ascii_functions as asc
from crop import crop
import pykdtree

def get_domain_settings(name):
    if name=='Australia':
        plot_extent=[100,180,-60,-5]
        centlon=0
        figsize=(10,8)
        barblength=7
    if name=='SouthernAfrica':
        plot_extent=[-20,60,-60,-5]
        centlon=0
        figsize=(10,8)
        barblength=7
    if name=='SouthAmerica':
        plot_extent=[-100,-20,-60,-5]
        centlon=0
        figsize=(10,8)
        barblength=7
    if name=='IndianOcean':
        plot_extent=[40,120,-60,-5]
        centlon=0
        figsize=(10,8)
        barblength=7
    #if name=='SouthAtlantic':
    #    plot_extent=[-50,20,-50,5]
    #    centlon=0
    #    figsize=(10,8)
    if name=='PacificOcean':
        plot_extent=[170,290,-60,-5]
        centlon=180
        figsize=(15,8)
        barblength=7
    if name=='SH':
        #plot_extent=[0,360,-60,-5]
        plot_extent=[-180,180,-60,-5]
        centlon=0
        figsize=(20,4)
        barblength=5
        
    return plot_extent,centlon,figsize,barblength
        
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

def plot_precip6h(inpath,outpath,dt,init_dt,fignum,data,name='Australia',cbar='on',**kwargs):
    dstr=dt.strftime('%Y%m%d_%H')
    dstr_long=dt.strftime('%H%M UTC %d %b %Y')
    dstr_init_long=init_dt.strftime('%H%M UTC %d %b %Y')
    
    plot_extent,centlon,figsize,barblength=get_domain_settings(name)
    
    fig=plt.figure(figsize=figsize)
    ax=plt.axes(projection=ccrs.PlateCarree(central_longitude=centlon))
    ax.set_extent(plot_extent,crs=ccrs.PlateCarree())
    rain_levels =  [0.2,0.5,1,2,5,15,20,25,30,35,40,45,50,60,70,80,90,100,125,150,200]
    cmap = plt.cm.get_cmap('gist_ncar', len(rain_levels))
    cf=ax.contourf(data['lons'],data['lats'],data['precip'],levels=rain_levels,
            norm=BoundaryNorm(rain_levels,len(rain_levels)),cmap=cmap,extend='max',
            transform=ccrs.PlateCarree())

    nvec=10
    q=plt.barbs(data['lons'][::nvec], data['lats'][::nvec], 
                data['u10'][::nvec,::nvec], data['v10'][::nvec,::nvec],
            length=4, fill_empty=True, pivot='middle',#flagcolor=[.3,.3,.3],barbcolor=[.3,.3,.3],
            sizes=dict(emptybarb=0.05, spacing=0.2, height=0.3),linewidth=0.6,
            transform=ccrs.PlateCarree())
            
    plot_levels = list(range(800,1400,4))
    c=ax.contour(data['lons'],data['lats'],data['mslp'],
                 levels=plot_levels,colors='black',linewidths=1.5,
                 transform=ccrs.PlateCarree())
    fmt = '%i'
    ax.clabel(c, c.levels, inline=True, fmt=fmt, fontsize=8)  
    
    ax.add_feature(LAND,facecolor='lightgrey')
    ax.coastlines(linewidths=0.4)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=0.4, color='lightgrey', alpha=0.5, linestyle='--')
    xlocators=list(np.arange(-180,190,10))
    gl.xlocator = mticker.FixedLocator(xlocators)
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = gl.ylabel_style ={'size': 8, 'color': 'black'}
    
    plt.title('MSLP (black contours) | 10m Wind (arrows) | Precip (accumlated, shaded) \n GFS forecast | Init: ' + dstr_init_long +' | Valid: '+dstr_long, 
              fontsize=10)
    if cbar=='on':
        ax_pos=[0.15,0.12,0.7,0.015] #[left, bottom, width, height]
        fig=plt.gcf()
        cbar_ax = fig.add_axes(ax_pos)
        cb=fig.colorbar(cf,orientation="horizontal", cax=cbar_ax, ticks=rain_levels)
        cb.ax.tick_params(labelsize=8)
        cb.set_label('Accumulated Rainfall [ mm ]', rotation=0, fontsize=8)

    outfile=outpath+'GFS_'+name+'_Precip6H_'+str(fignum)+'.jpg'
    plt.savefig(outfile, dpi=150)
    crop(outfile)
    #plt.show()
    #plt.gcf()
    #plt.close() 

def plot_PTe(inpath,outpath,dt,init_dt,fignum,data,name='Australia',**kwargs):
    
    plot_extent,centlon,figsize,barblength=get_domain_settings(name)
    
    dstr=dt.strftime('%Y%m%d_%H')
    dstr_long=dt.strftime('%H%M UTC %d %b %Y')
    dstr_init_long=init_dt.strftime('%H%M UTC %d %b %Y')
    
    fig=plt.figure(figsize=figsize)
    ax=plt.axes(projection=ccrs.PlateCarree(central_longitude=centlon))
    ax.set_extent(plot_extent,crs=ccrs.PlateCarree())
    plot_levels = list(range(270,360,6))
    cf=ax.contourf(data['pte'].longitude,data['pte'].latitude,data['pte'],
                   levels=plot_levels,cmap='RdYlBu_r',extend='both',
                   transform=ccrs.PlateCarree())
    
    plot_levels = list(range(1100,1600,30))
    c=ax.contour(data['hgt'].longitude,data['hgt'].latitude,data['hgt'],
                 levels=plot_levels,colors='black',
                 linestyles='-',linewidths=0.8,
                 transform=ccrs.PlateCarree(central_longitude=0))
    c_lab=ax.clabel(c, c.levels, inline=True, fontsize=10)
    nvec=20
    uu=data['u'][::nvec,::nvec].data
    vv=data['v'][::nvec,::nvec].data
    q=ax.barbs(data['u'].longitude[::nvec], data['u'].latitude[::nvec],uu,vv,
                  fill_empty=True,sizes=dict(emptybarb=0),
                  length=barblength,
                  #scale=15,scale_units='xy',width=0.005,minshaft=1.5,
                  transform=ccrs.PlateCarree())
                  
    ax.add_feature(LAND,facecolor='lightgrey')
    ax.coastlines(linewidths=1.2)
    gl = ax.gridlines(transform=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=0.4, color='lightgrey', alpha=0.5, linestyle='--')
    xlocators=list(np.arange(-180,190,10))
    gl.xlocator = mticker.FixedLocator(xlocators)
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = gl.ylabel_style ={'size': 8, 'color': 'black'}
    
    # cb=plt.colorbar(cf,orientation="horizontal")
    plt.title('850 equiv. PT, wind and geopotential height (black contours)\n GFS Forecast | Init: ' + dstr_init_long +' | Valid: '+dstr_long, 
              fontsize=10)
    
    ax_pos=[0.25,0.12,0.5,0.015] #[left, bottom, width, height]
    fig=plt.gcf()
    cbar_ax = fig.add_axes(ax_pos)
    cb=fig.colorbar(cf,orientation="horizontal", cax=cbar_ax)
    cb.ax.tick_params(labelsize=8)
    cb.set_label('Equivalent Potential Temperature [K]', rotation=0, fontsize=8)
    
    outfile=outpath+'GFS_'+name+'_PTE850_'+str(fignum)+'.jpg'
    #fig.tight_layout()
    plt.savefig(outfile, dpi=150)
    crop(outfile)
    #plt.show()
    
    
def plot_DT(inpath,outpath,dt,init_dt,fignum,data,name='Australia',**kwargs):
    
    plot_extent,centlon,figsize,barblength=get_domain_settings(name)
    
    dstr=dt.strftime('%Y%m%d_%H')
    dstr_long=dt.strftime('%H%M UTC %d %b %Y')
    dstr_init_long=init_dt.strftime('%H%M UTC %d %b %Y')
    
    fig=plt.figure(figsize=figsize)
    ax=plt.axes(projection=ccrs.PlateCarree(central_longitude=centlon))
    ax.set_extent(plot_extent,crs=ccrs.PlateCarree())
    plot_levels = list(range(270,400,6))
    cf=ax.contourf(data['theta'].longitude,data['theta'].latitude,data['theta'],
                   levels=plot_levels,cmap='RdYlBu_r',extend='max',
                   transform=ccrs.PlateCarree())
    
    plot_levels = [-1.5e-4,-1.0e-4,-0.65e-4]
    c=ax.contour(data['vort'].longitude,data['vort'].latitude,data['vort'],
                 levels=plot_levels,colors='black',
                 linestyles='-',linewidths=0.8,transform=ccrs.PlateCarree())
    nvec=20
    uu=data['u'][::nvec,::nvec].data
    vv=data['v'][::nvec,::nvec].data
    q=plt.barbs(data['u'].longitude[::nvec], data['u'].latitude[::nvec],uu,vv,
                  fill_empty=True,sizes=dict(emptybarb=0),#scale=15,scale_units='xy',width=0.005,minshaft=1.5,
                  transform=ccrs.PlateCarree())
                  
    ax.add_feature(LAND,facecolor='lightgrey')
    ax.coastlines(linewidths=1.2)
    gl = ax.gridlines(transform=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=0.4, color='lightgrey', alpha=0.5, linestyle='--')
    xlocators=list(np.arange(-180,190,10))
    gl.xlocator = mticker.FixedLocator(xlocators)
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = gl.ylabel_style ={'size': 8, 'color': 'black'}
    
    # cb=plt.colorbar(cf,orientation="horizontal")
    plt.title('DT temperature (shading) | wind (vectors) | 850-900hPa rot. vorticity (black contours)\n GFS Forecast | Init: ' + dstr_init_long +' | Valid: '+dstr_long, 
              fontsize=10)
    
    ax_pos=[0.25,0.12,0.5,0.015] #[left, bottom, width, height]
    fig=plt.gcf()
    cbar_ax = fig.add_axes(ax_pos)
    cb=fig.colorbar(cf,orientation="horizontal", cax=cbar_ax)
    cb.ax.tick_params(labelsize=8)
    cb.set_label('Potential Temperature', rotation=0, fontsize=8)
    
    outfile=outpath+'GFS_'+name+'_DT_'+str(fignum)+'.jpg'
    plt.savefig(outfile, dpi=150)
    crop(outfile)
    #plt.show()
    #plt.clf()

def plot_upper(inpath,outpath,dt,init_dt,fignum,data,name='Australia',**kwargs):
    
    plot_extent,centlon,figsize,barblength=get_domain_settings(name)
    
    dstr=dt.strftime('%Y%m%d_%H')
    dstr_long=dt.strftime('%H%M UTC %d %b %Y')
    dstr_init_long=init_dt.strftime('%H%M UTC %d %b %Y')
    
    fig=plt.figure(figsize=figsize)
    ax=plt.axes(projection=ccrs.PlateCarree(central_longitude=centlon))
    ax.set_extent(plot_extent,crs=ccrs.PlateCarree())
    plot_levels = [50,60,70,80,90,100,120,140,160]
    cf=ax.contourf(data['lons'],data['lats'],data['jet300']*1.94384,
                   levels=plot_levels,cmap='Blues',extend='max',
                   transform=ccrs.PlateCarree())
    
    plot_levels = list(range(800,1400,4))
    cmslp=ax.contour(data['lons'],data['lats'],data['mslp'],
                     levels=plot_levels,colors='grey',linewidths=1,linestyles='--',
                     transform=ccrs.PlateCarree())
    
    plot_levels = [-30e-3 -25e-3 -20e-3 -15e-3, -10e-3, -5e-3]
    c1=ax.contourf(data['lons'],data['lats'],data['wMID'],levels=plot_levels,cmap='Reds',
                   transform=ccrs.PlateCarree())
    plot_levels = list(range(4000,6100,50))
    
    c=ax.contour(data['lons'],data['lats'],data['z500'],levels=plot_levels,colors='black',
                 linewidths=0.8,transform=ccrs.PlateCarree())
    nvec=10
    q=plt.quiver(data['lons'][::nvec], data['lats'][::nvec], 
                 data['u300'][::nvec,::nvec], data['v300'][::nvec,::nvec],
                  scale=15,scale_units='xy',width=0.005,minshaft=1.5,
                  transform=ccrs.PlateCarree())
                  
    ax.add_feature(LAND,facecolor='lightgrey')
    ax.coastlines(linewidths=0.4)
    gl = ax.gridlines(transform=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=0.4, color='lightgrey', alpha=0.5, linestyle='--')
    xlocators=list(np.arange(-180,190,10))
    gl.xlocator = mticker.FixedLocator(xlocators)
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = gl.ylabel_style ={'size': 8, 'color': 'black'}
    
    # cb=plt.colorbar(cf,orientation="horizontal")
    plt.title('500hPa GPH (black contours) | 400-600hPa Vert. Vel. (red shading) | MSLP (grey dashed) | 300hPa Jet (blue shading)\n GFS Forecast | Init: ' + dstr_init_long +' | Valid: '+dstr_long, 
              fontsize=10)
    
    ax_pos=[0.25,0.12,0.5,0.015] #[left, bottom, width, height]
    fig=plt.gcf()
    cbar_ax = fig.add_axes(ax_pos)
    cb=fig.colorbar(cf,orientation="horizontal", cax=cbar_ax)
    cb.ax.tick_params(labelsize=8)
    cb.set_label('Wind Speed [m/s]', rotation=0, fontsize=8)
    
    outfile=outpath+'GFS_'+name+'_UpperLevel_'+str(fignum)+'.jpg'
    plt.savefig(outfile, dpi=150)
    crop(outfile)
    #plt.show()
    #plt.clf()
    
def plot_irrotPV(inpath,outpath,dt,init_dt,fignum,data,name='Australia',**kwargs):
    
    plot_extent,centlon,figsize,barblength=get_domain_settings(name)
    
    dstr=dt.strftime('%Y%m%d_%H')
    dstr_long=dt.strftime('%H%M UTC %d %b %Y')
    dstr_init_long=init_dt.strftime('%H%M UTC %d %b %Y')
    
    fig=plt.figure(figsize=figsize)
    ax=plt.axes(projection=ccrs.PlateCarree(central_longitude=centlon))
    ax.set_extent(plot_extent,crs=ccrs.PlateCarree())
    
    plot_levels = list(range(20,80,5))
    cf_pw=ax.contourf(data['lons'],data['lats'],data['pwat'],
                   levels=plot_levels,cmap='GnBu',extend='max',
                   transform=ccrs.PlateCarree())
    
    jet_cmap = plt.cm.get_cmap('RdPu', 16)(range(5,16))
    jet_cmap = ListedColormap(jet_cmap)
    plot_levels = [30,35,40,45,50,60,70,80,90,100,120]
    cf=ax.contourf(data['lons'],data['lats'],data['jet300'],
                   levels=plot_levels,cmap=jet_cmap,extend='max',
                   transform=ccrs.PlateCarree())
    
    plot_levels = list(range(-15,0,1))
    cpv=ax.contour(data['lons'],data['lats'],gaussian_filter(data['pv_iso_upper']*1e6,sigma=2),
                     levels=plot_levels,colors='black',linestyles='-',
                     linewidths=1.5,transform=ccrs.PlateCarree())
    fmt = '%i'
    ax.clabel(cpv, cpv.levels[1::2], inline=True, fmt=fmt, fontsize=14)  
    
    nvec=10
    q=plt.quiver(data['lons'][::nvec], np.flip(data['lats'][::nvec]), 
                 data['uchi'][::nvec,::nvec],
                 data['vchi'][::nvec,::nvec],
                 scale=2,scale_units='xy',width=0.005,minshaft=1,minlength=1e-1000,
                 transform=ccrs.PlateCarree())
                  
    ax.add_feature(LAND,facecolor='lightgrey')
    ax.coastlines(linewidths=0.4,zorder=1e4)
    gl = ax.gridlines(transform=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=0.4, color='lightgrey', alpha=0.5, linestyle='--')
    xlocators=list(np.arange(-180,190,10))
    gl.xlocator = mticker.FixedLocator(xlocators)
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = gl.ylabel_style ={'size': 8, 'color': 'black'}
    
    # cb=plt.colorbar(cf,orientation="horizontal")
    plt.title('200-300hPa PV (black contours) and irrot. wind (quiver) | 300hPa Jet (magenta shading)\n GFS Forecast | Init: ' + dstr_init_long +' | Valid: '+dstr_long, 
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
    
    outfile=outpath+'GFS_'+name+'_IrrotPV_'+str(fignum)+'.jpg'
    plt.savefig(outfile, dpi=150)
    crop(outfile)
    #plt.show()
    #plt.clf()
    
def plot_IVT(inpath,outpath,dt,init_dt,fignum,data,name='Australia',**kwargs):
    
    plot_extent,centlon,figsize,barblength=get_domain_settings(name)
    
    dstr=dt.strftime('%Y%m%d_%H')
    dstr_long=dt.strftime('%H%M UTC %d %b %Y')
    dstr_init_long=init_dt.strftime('%H%M UTC %d %b %Y')
    
    fig=plt.figure(figsize=figsize)
    ax=plt.axes(projection=ccrs.PlateCarree(central_longitude=centlon))
    ax.set_extent(plot_extent,crs=ccrs.PlateCarree())
    plot_levels = [250,300,400,500,600,700,800,1000,1200,1400,1600,1800]
    ivt_cmap = plt.cm.get_cmap('cool', len(plot_levels))
    cf=ax.contourf(data['lons'],data['lats'],data['IVT'],levels=plot_levels,
            norm=BoundaryNorm(plot_levels,len(plot_levels)),cmap=ivt_cmap,extend='max',
            transform=ccrs.PlateCarree())
    c=ax.contour(data['lons'],data['lats'],data['IVT'],levels=plot_levels,colors='grey',
                 linewidths=0.4,transform=ccrs.PlateCarree())
    nvec=10
    q=plt.quiver(data['lons'][::nvec], data['lats'][::nvec], 
                 data['uIVT'][::nvec,::nvec], data['vIVT'][::nvec,::nvec],\
                 scale=250,scale_units='xy',width=0.0105,minshaft=2,minlength=0.01,
                 transform=ccrs.PlateCarree())
    plot_levels = list(range(0,6000,30))
    c=ax.contour(data['lons'],data['lats'],data['z700'],levels=plot_levels,
                 colors='black',linewidths=0.6,
                 transform=ccrs.PlateCarree())
    fmt = '%i'
    ax.clabel(c, c.levels, inline=True, fmt=fmt, fontsize=8)
    
    ax.add_feature(LAND,facecolor='lightgrey')
    ax.coastlines(linewidths=0.4)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=0.4, color='lightgrey', alpha=0.5, linestyle='--')
    xlocators=list(np.arange(-180,190,10))
    gl.xlocator = mticker.FixedLocator(xlocators)
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = gl.ylabel_style ={'size': 8, 'color': 'black'}

    plt.title('700hPa GPH (black contours) | IVT (shaded, arrows) \n GFS forecast | Init: ' + dstr_init_long +' | Valid: '+dstr_long, 
              fontsize=10)
    
    ax_pos=[0.25,0.12,0.5,0.015] #[left, bottom, width, height]
    fig=plt.gcf()
    cbar_ax = fig.add_axes(ax_pos)
    cb=fig.colorbar(cf,orientation="horizontal", cax=cbar_ax)
    cb.ax.tick_params(labelsize=8)
    cb.set_label('IVT', rotation=0, fontsize=8)
    
    outfile=outpath+'GFS_'+name+'_IVT_'+str(fignum)+'.jpg'
    plt.savefig(outfile, dpi=150)
    crop(outfile)
    #plt.clf()
    #plt.show()
    
def plot_IPV(outpath,dt,init_dt,fignum,plot_level,data,name='Australia',**kwargs):
    
    plot_extent,centlon,figsize,barblength=get_domain_settings(name)
    
    dstr=dt.strftime('%Y%m%d_%H')
    dstr_long=dt.strftime('%H%M UTC %d %b %Y')
    dstr_init_long=init_dt.strftime('%H%M UTC %d %b %Y')
    
    pv_clevs=[-12,-10,-8,-6,-4,-3,-2,-1.5,-1,-0.5]
    pv_cmap = plt.cm.get_cmap('RdPu_r', 10)(range(0,6))
    pv_cmap = np.concatenate((pv_cmap,plt.cm.get_cmap('Blues_r', 6)(range(2,5))))
    #pv_cmap = np.concatenate((pv_cmap,plt.cm.get_cmap('Greys', 255)(range(0,1))))
    pv_cmap = ListedColormap(pv_cmap)
    
    fig=plt.figure(figsize=figsize)
    ax=plt.axes(projection=ccrs.PlateCarree(central_longitude=centlon))
    ax.set_extent(plot_extent,crs=ccrs.PlateCarree())
    cf=plt.contourf(data['pv'+str(plot_level)].longitude,data['pv'+str(plot_level)].latitude,
                    gaussian_filter(data['pv'+str(plot_level)]/1e-6,sigma=1),
                    levels=pv_clevs,cmap=pv_cmap,norm=BoundaryNorm(pv_clevs,len(pv_clevs)),
                    transform=ccrs.PlateCarree(),extend='min')
    c=ax.contour(data['pv'+str(plot_level)].longitude,data['pv'+str(plot_level)].latitude,
                 gaussian_filter(data['pv'+str(plot_level)]/1e-6,sigma=1),
                 levels=pv_clevs[0:-1],colors='grey',linewidths=0.4)
    
    nvec=10
    q=plt.quiver(data['u'+str(plot_level)].longitude[::nvec], data['u'+str(plot_level)].latitude[::nvec], 
                 data['u'+str(plot_level)][::nvec,::nvec], data['v'+str(plot_level)][::nvec,::nvec],\
                 scale=25,scale_units='xy',width=0.0205,minshaft=1.3,minlength=0.01,
                 transform=ccrs.PlateCarree())
    
    ax.add_feature(LAND,facecolor='lightgrey')
    ax.coastlines(linewidths=1)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=0.4, color='lightgrey', alpha=0.5, linestyle='--')
    xlocators=list(np.arange(-180,190,10))
    gl.xlocator = mticker.FixedLocator(xlocators)
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = gl.ylabel_style ={'size': 8, 'color': 'black'}

    plt.title(str(plot_level)+'K PV (shaded) and wind (quivers)\n GFS Forecast | Init: ' + dstr_init_long +' | Valid: '+dstr_long,
              fontsize=10)
    
    ax_pos=[0.25,0.08,0.5,0.02] #[left, bottom, width, height]
    fig=plt.gcf()
    cbar_ax = fig.add_axes(ax_pos)
    cb=fig.colorbar(cf,orientation="horizontal", cax=cbar_ax)
    cb.ax.tick_params(labelsize=8)
    cb.set_label('Potential Vorticity [PVU]', rotation=0, fontsize=8)
    
    outfile=outpath+'GFS_'+name+'_IPV-'+str(plot_level)+'K_'+str(fignum)+'.jpg'
    plt.savefig(outfile, dpi=150)
    crop(outfile)
    #plt.clf()
    #plt.show()




