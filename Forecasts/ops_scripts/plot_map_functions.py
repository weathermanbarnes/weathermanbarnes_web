import nclcmaps
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
from cartopy.util import add_cyclic
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from windspharm.xarray import VectorWind
from windspharm.tools import prep_data, recover_data, order_latdim
from metpy.calc import equivalent_potential_temperature,dewpoint_from_relative_humidity,potential_vorticity_baroclinic
from metpy.calc import potential_temperature,isentropic_interpolation_as_dataset
from metpy.units import units
import pykdtree
import gc

from PIL import Image,ImageOps # pip install Pillow
import glob
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

def expand_xr_longitudes(inds,periodic_add=70):
    # expand field for periodicity
    dsW=inds.isel({"longitude": slice(0, int(periodic_add/0.25))})
    dsW["longitude"]=dsW.longitude+360

    dsE=inds.isel({"longitude": slice(len(inds.longitude)-int(periodic_add / 0.25), len(inds.longitude))})
    dsE["longitude"]=dsE.longitude-360
    
    return xr.concat(
        [
            dsE,
            inds,
            dsW,
        ],
        dim="longitude",
    )

def get_domain_settings(name):
    if name=='Australia':
        plot_extent=[90,180,-60,-5]
        centlon=0
        figsize=(10,8)
        barblength=7
        proj=ccrs.Robinson(central_longitude=plot_extent[0]+(abs(plot_extent[1]-plot_extent[0])/2))
        southern_hemisphere=True
        regional_grid=True
    if name=='SouthernAfrica':
        plot_extent=[-20,60,-55,-5]
        centlon=0
        figsize=(11,9)
        barblength=7
        proj=ccrs.Robinson(central_longitude=plot_extent[0]+(abs(plot_extent[1]-plot_extent[0])/2))
        #proj=ccrs.PlateCarree()
        southern_hemisphere=True
        regional_grid=True
    if name=='SouthAmerica':
        plot_extent=[-115,10,-70,15]
        centlon=0
        figsize=(11,10)
        barblength=7
        proj=ccrs.Robinson(central_longitude=plot_extent[0]+(abs(plot_extent[1]-plot_extent[0])/2))
        southern_hemisphere=True
        regional_grid=True
    if name=='IndianOcean':
        plot_extent=[40,120,-55,-5]
        centlon=0
        figsize=(10,8)
        barblength=7
        proj=ccrs.Robinson(central_longitude=plot_extent[0]+(abs(plot_extent[1]-plot_extent[0])/2))
        southern_hemisphere=True
        regional_grid=True
    if name=='PacificOcean':
        plot_extent=[170,290,-60,-5]
        centlon=180
        figsize=(15,8)
        barblength=7
        proj=ccrs.Robinson(central_longitude=plot_extent[0]+(abs(plot_extent[1]-plot_extent[0])/2))
        southern_hemisphere=True
        regional_grid=True
    if name=='SH':
        plot_extent=[-180,179.75,-90,-5]
        centlon=130
        figsize=(11,12)
        barblength=5
        proj=ccrs.SouthPolarStereo(central_longitude=centlon)
        southern_hemisphere=True
        regional_grid=False
    if name=='Europe':
        plot_extent=[-40,45,25,80]
        centlon=0
        figsize=(12,11)
        barblength=7
        proj=ccrs.LambertConformal(central_longitude=plot_extent[0]+(abs(plot_extent[1]-plot_extent[0])/2))
        southern_hemisphere=False
        regional_grid=True
    if name=='NorthAmerica':
        plot_extent=[-140,-60,10,65]
        centlon=0
        figsize=(10,8)
        barblength=7
        proj=ccrs.LambertConformal(central_longitude=plot_extent[0]+(abs(plot_extent[1]-plot_extent[0])/2))
        southern_hemisphere=False
        regional_grid=True
    if name=='NorthAtlantic':
        plot_extent=[-90,0,10,65]
        centlon=0
        figsize=(11,8)
        barblength=7
        proj=ccrs.LambertConformal(central_longitude=plot_extent[0]+(abs(plot_extent[1]-plot_extent[0])/2))
        southern_hemisphere=False
        regional_grid=True
    if name=='NorthAfrica':
        plot_extent=[-30,60,-5,45]
        centlon=0
        figsize=(10,7)
        barblength=7
        proj=ccrs.LambertConformal(central_longitude=plot_extent[0]+(abs(plot_extent[1]-plot_extent[0])/2))
        southern_hemisphere=False
        regional_grid=True
    if name=='Asia':
        plot_extent=[70,180,0,60]
        centlon=0
        figsize=(12,8)
        barblength=7
        proj=ccrs.LambertConformal(central_longitude=plot_extent[0]+(abs(plot_extent[1]-plot_extent[0])/2))
        southern_hemisphere=False
        regional_grid=True
    if name=='NH':
        plot_extent=[-180,179.75,5,90]
        centlon=0
        figsize=(11,12)
        barblength=5
        proj=ccrs.NorthPolarStereo()
        southern_hemisphere=False
        regional_grid=False
        
    return plot_extent,centlon,figsize,barblength,proj,southern_hemisphere,regional_grid

def replace_values_above_equator(da1: xr.DataArray, da2: xr.DataArray, lat_name: str = 'lat') -> xr.DataArray:
    """
    Replace values in da1 with values from da2 where latitude > 0.
    
    Parameters:
    - da1: xarray.DataArray
        The original data array where values will be replaced.
    - da2: xarray.DataArray
        The data array containing replacement values.
    - lat_name: str, optional
        The name of the latitude coordinate in the data arrays (default is 'lat').
        
    Returns:
    - xarray.DataArray
        The modified data array with values replaced above the equator.
    """
    # Ensure both arrays are aligned based on their coordinates (especially latitude)
    da1, da2 = xr.align(da1, da2)
    
    # Create a mask for latitudes greater than 0
    mask = da1[lat_name] > 0
    
    # Replace values in da1 where the latitude is greater than 0 with those from da2
    da1 = da1.where(~mask, da2)
    
    return da1

        
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

def plot_fai(outpath,dt,init_dt,fignum,indata,name='Australia',model_name='GFS',dpi=150,**kwargs):
    plot_extent,centlon,figsize,barblength,proj,southern_hemisphere,regional_grid=get_domain_settings(name)
    data=indata.copy()
    
    for k in data.keys():
        if k=='lats':
            data[k]=data[k].sel(latitude=slice(plot_extent[3]+10,plot_extent[2]-5))
        elif k=='lons':
            None
        else:
            data[k]=data[k].sel(latitude=slice(plot_extent[3]+10,plot_extent[2]-5))
    
    dstr=dt.strftime('%Y%m%d_%H')
    dstr_long=dt.strftime('%H%M UTC %d %b %Y')
    dstr_init_long=init_dt.strftime('%H%M UTC %d %b %Y')
    
    fig=plt.figure(figsize=figsize)
    data_crs=ccrs.PlateCarree()
    
    ax=plt.axes(projection=proj)
    ax.set_extent(plot_extent,crs=data_crs)
    if not regional_grid:
        ax.set_boundary(map_circle, transform=ax.transAxes)
    
    plot_levels = np.arange(2,20,2)
    lon2d, lat2d = np.meshgrid(data['fai850'].longitude.values, data['fai850'].latitude.values)
    cyclic_data, cyclic_lon2d, cyclic_lat2d = add_cyclic((data['fai850'].values*1e10), x=lon2d, y=lat2d)
    if not regional_grid:
        cf=ax.pcolormesh(cyclic_lon2d, cyclic_lat2d, cyclic_data, vmin=2, vmax=20, #levels=plot_levels,
                cmap='ocean_r',#extend='max',
                transform=data_crs)
    else:
        cf=ax.pcolormesh(cyclic_lon2d, cyclic_lat2d, cyclic_data, vmin=2, vmax=20, #levels=plot_levels,
                cmap='ocean_r',#extend='max',
                transform=data_crs)
    
    plot_levels = list(range(1000,2500,30))
    lon2d, lat2d = np.meshgrid(data['z850'].longitude.values, data['z850'].latitude.values)
    cyclic_data, cyclic_lon2d, cyclic_lat2d = add_cyclic(data['z850'].values, x=lon2d, y=lat2d)
    c=ax.contour(cyclic_lon2d, cyclic_lat2d, cyclic_data,
                 levels=plot_levels,colors='black',
                 linewidths=0.8,transform=data_crs)
    
    fmt = '%i'
    ax.clabel(c, c.levels, inline=True, fmt=fmt, fontsize=10, inline_spacing=0)
    
    ax.add_feature(LAND,facecolor='lightgrey')
    ax.coastlines(linewidths=0.4)
    
    if not regional_grid:
        gl = ax.gridlines(crs=data_crs, draw_labels=True,
                          linewidth=0.6, color='lightgrey', alpha=0.5, linestyle='--')
    else:
        gl = ax.gridlines(crs=data_crs, draw_labels=True, x_inline=False, 
                          y_inline=False, linewidth=0.33, color='k',alpha=0.5)
    if not regional_grid:
        xlocators=list(np.arange(-180,190,30))
    else:
        xlocators=list(np.arange(-180,190,10))
    gl.xlocator = mticker.FixedLocator(xlocators)
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.top_labels = False
    gl.right_labels = False
    if not regional_grid:
        gl.bottom_labels = False 
    if regional_grid:
        gl.xlabel_style = gl.ylabel_style ={'size': 8, 'color': 'black','rotation': 0}
    else:
        gl.xlabel_style = gl.ylabel_style ={'size': 8, 'color': 'black'}
    
    plt.title('850hPa GPH (black) | 850hPa Frontal Activity Index (shading)\n'+model_name +' Forecast | Init: ' + dstr_init_long +' | Valid: '+dstr_long, 
              fontsize=10)
    
    ax_pos=[0.25,0.12,0.5,0.015] #[left, bottom, width, height]
    fig=plt.gcf()
    cbar_ax = fig.add_axes(ax_pos)
    cb=fig.colorbar(cf,orientation="horizontal", cax=cbar_ax)
    cb.ax.tick_params(labelsize=8)
    cb.set_label('Frontal Activity Index [$K.m^{-1}.s^{-1} * 1e^{10}$]', rotation=0, fontsize=8)
    #cb.set_label(incapetype+' [$J.kg^{-1}$]', rotation=0, fontsize=8)
    
    copywrite_text='\xa9 Michael A. Barnes\nwww.weathermanbarnes.com'
    ax.text(0.01, 0.015, copywrite_text, fontsize=6, 
             horizontalalignment='left', verticalalignment='bottom', 
             transform=ax.transAxes,
             bbox=dict(facecolor='white', alpha=0.95),zorder=10)
    
    outfile=outpath+model_name+'_'+name+'_FAI_'+str(fignum)+'.jpg'
    plt.savefig(outfile, dpi=dpi)
    crop(outfile)
    plt.close('all') 
    #plt.show()

def plot_precip6h(outpath,dt,init_dt,fignum,indata,name='Australia',model_name='GFS',dpi=150,**kwargs):
    
    plot_extent,centlon,figsize,barblength,proj,southern_hemisphere,regional_grid=get_domain_settings(name)
    data=indata.copy()
    
    for k in data.keys():
        if k=='lats':
            data[k]=data[k].sel(latitude=slice(plot_extent[3]+10,plot_extent[2]-5))
        elif k=='lons':
            None
        else:
            data[k]=data[k].sel(latitude=slice(plot_extent[3]+10,plot_extent[2]-5))
    
    dstr=dt.strftime('%Y%m%d_%H')
    dstr_long=dt.strftime('%H%M UTC %d %b %Y')
    dstr_init_long=init_dt.strftime('%H%M UTC %d %b %Y')
    
    fig=plt.figure(figsize=figsize)
    data_crs=ccrs.PlateCarree()
    
    ax=plt.axes(projection=proj)
    ax.set_extent(plot_extent,crs=data_crs)
    if not regional_grid:
        ax.set_boundary(map_circle, transform=ax.transAxes)
    
    lon2d, lat2d = np.meshgrid(data['precip'].longitude.values, data['precip'].latitude.values)
    cyclic_data, cyclic_lon2d, cyclic_lat2d = add_cyclic(data['precip'].values, x=lon2d, y=lat2d)
    
    rain_levels =  [0.2,0.5,1,2,5,15,20,25,30,35,40,45,50,60,70,80,90,100,125,150,200]
    cmap = nclcmaps.cmap('WhiteBlueGreenYellowRed')(range(26,256,int(np.floor(256/len(rain_levels)+2))))
    cmap = ListedColormap(np.concatenate([cmap,nclcmaps.cmap('MPL_gist_nca')(range(102,129,7))]))
    #cdata, clon, clat = cutil.add_cyclic(data['precip'].values,
    #                                     data['precip'].longitude.values,
    #                                     data['precip'].latitude.values)
    #cf=ax.contourf(data['precip'].longitude.values,data['precip'].latitude.values,
    #               data['precip'],levels=rain_levels,
    #               norm=BoundaryNorm(rain_levels,len(rain_levels)),cmap=cmap,extend='max',
    #               transform=data_crs)
    cf=ax.contourf(cyclic_lon2d, cyclic_lat2d,cyclic_data,
                   levels=rain_levels,
                   norm=BoundaryNorm(rain_levels,len(rain_levels)),cmap=cmap,extend='max',
                   transform=data_crs)#, transform_first=True)
    
    if not regional_grid:
        lon2d, lat2d = np.meshgrid(data['u10'].longitude.values, data['u10'].latitude.values)
        cyclic_u, cyclic_lon2d, cyclic_lat2d = add_cyclic(data['u10'].values, x=lon2d, y=lat2d)
        cyclic_v, cyclic_lon2d, cyclic_lat2d = add_cyclic(data['v10'].values, x=lon2d, y=lat2d)
        q=plt.barbs(cyclic_lon2d, cyclic_lat2d,  
                cyclic_u*1.94384, cyclic_v*1.94384,
                length=4, fill_empty=True, pivot='middle',#flagcolor=[.3,.3,.3],barbcolor=[.3,.3,.3],
                sizes=dict(emptybarb=0.05, spacing=0.2, height=0.3), linewidth=0.6, regrid_shape=40,
                transform=data_crs)
    else:
        expanded_u=expand_xr_longitudes(data['u10'],periodic_add=60)
        expanded_v=expand_xr_longitudes(data['v10'],periodic_add=60)
        q=plt.barbs(expanded_u.longitude, expanded_u.latitude,  
                    expanded_u*1.94384, expanded_v*1.94384, regrid_shape=20,
                    length=5, fill_empty=True, pivot='middle',#flagcolor=[.3,.3,.3],barbcolor=[.3,.3,.3],
                    sizes=dict(emptybarb=0.05, spacing=0.2, height=0.3),linewidth=0.9,
                    transform=data_crs)
            
    plot_levels = list(range(800,1400,4))
    lon2d, lat2d = np.meshgrid(data['mslp'].longitude.values, data['mslp'].latitude.values)
    cyclic_data, cyclic_lon2d, cyclic_lat2d = add_cyclic(data['mslp'].values, x=lon2d, y=lat2d)
    c=ax.contour(cyclic_lon2d, cyclic_lat2d, cyclic_data,
                 levels=plot_levels,colors='black',linewidths=1.5,
                 transform=data_crs)
    fmt = '%i'
    ax.clabel(c, c.levels, inline=True, fmt=fmt, fontsize=8) 
    
    ax.add_feature(LAND,facecolor='lightgrey')
    ax.coastlines(linewidths=0.4)
    
    if not regional_grid:
        gl = ax.gridlines(crs=data_crs, draw_labels=True,
                          linewidth=0.6, color='lightgrey', alpha=0.5, linestyle='--')
    else:
        gl = ax.gridlines(crs=data_crs, draw_labels=True, x_inline=False, 
                          y_inline=False, linewidth=0.33, color='k',alpha=0.5)
    if not regional_grid:
        xlocators=list(np.arange(-180,190,30))
    else:
        xlocators=list(np.arange(-180,190,10))
    gl.xlocator = mticker.FixedLocator(xlocators)
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.top_labels = False
    gl.right_labels = False
    if not regional_grid:
        gl.bottom_labels = False 
    if regional_grid:
        gl.xlabel_style = gl.ylabel_style ={'size': 8, 'color': 'black','rotation': 0}
    else:
        gl.xlabel_style = gl.ylabel_style ={'size': 8, 'color': 'black'}
    
    plt.title('MSLP (black contours) | 10m Wind (arrows) | Precip (accumlated, shaded) \n'+model_name +' Forecast | Init: ' + dstr_init_long +' | Valid: '+dstr_long, 
              fontsize=10)
    
    ax_pos=[0.25,0.12,0.5,0.015] #[left, bottom, width, height]
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
    gc.collect()

def plot_LowVortPTE(outpath,dt,init_dt,fignum,indata,name='Australia',model_name='GFS',dpi=150,**kwargs):
    
    plot_extent,centlon,figsize,barblength,proj,southern_hemisphere,regional_grid=get_domain_settings(name)
    data=indata.copy()
    
    for k in data.keys():
        if k=='lats':
            data[k]=data[k].sel(latitude=slice(plot_extent[3]+10,plot_extent[2]-5))
        elif k=='lons':
            None
        else:
            data[k]=data[k].sel(latitude=slice(plot_extent[3]+10,plot_extent[2]-5))
    
    dstr=dt.strftime('%Y%m%d_%H')
    dstr_long=dt.strftime('%H%M UTC %d %b %Y')
    dstr_init_long=init_dt.strftime('%H%M UTC %d %b %Y')
    
    fig=plt.figure(figsize=figsize)
    data_crs=ccrs.PlateCarree()
    
    ax=plt.axes(projection=proj)
    ax.set_extent(plot_extent,crs=data_crs)
    if not regional_grid:
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
    lon2d, lat2d = np.meshgrid(data['pte850'].longitude.values, data['pte850'].latitude.values)
    cyclic_data, cyclic_lon2d, cyclic_lat2d = add_cyclic(data['pte850'].values, x=lon2d, y=lat2d)
    cf=ax.contourf(cyclic_lon2d, cyclic_lat2d, cyclic_data,
                   levels=plot_levels,cmap=jet_purple_cmap,extend='both',
                   transform=data_crs)
    
    plot_levels = list(range(500,3000,30))
    lon2d, lat2d = np.meshgrid(data['z850'].longitude.values, data['z850'].latitude.values)
    cyclic_data, cyclic_lon2d, cyclic_lat2d = add_cyclic(data['z850'].values, x=lon2d, y=lat2d)
    c=ax.contour(cyclic_lon2d, cyclic_lat2d, cyclic_data,
                 levels=plot_levels,colors='black',linewidths=1.5,
                 transform=data_crs)
    fmt = '%i'
    ax.clabel(c, c.levels, inline=True, fmt=fmt, fontsize=8)  

    if southern_hemisphere:
        plot_levels = np.arange(-7,-4.5,0.5)
    else:
        plot_levels = np.arange(5,7.5,0.5)
    lon2d, lat2d = np.meshgrid(data['vortLOW'].longitude.values, data['vortLOW'].latitude.values)
    cyclic_data, cyclic_lon2d, cyclic_lat2d = add_cyclic(data['vortLOW'].values, x=lon2d, y=lat2d)
    c=ax.contour(cyclic_lon2d, cyclic_lat2d,cyclic_data*1e5,
                 levels=plot_levels,colors='grey',linestyles='-',linewidths=0.8,
                 transform=data_crs)

    if not regional_grid:
        lon2d, lat2d = np.meshgrid(data['u850'].longitude.values, data['u850'].latitude.values)
        cyclic_u, cyclic_lon2d, cyclic_lat2d = add_cyclic(data['u850'].values, x=lon2d, y=lat2d)
        cyclic_v, cyclic_lon2d, cyclic_lat2d = add_cyclic(data['v850'].values, x=lon2d, y=lat2d)
        q=plt.barbs(cyclic_lon2d, cyclic_lat2d,  
                cyclic_u, cyclic_v,
                length=4, fill_empty=True, pivot='middle',#flagcolor=[.3,.3,.3],barbcolor=[.3,.3,.3],
                sizes=dict(emptybarb=0.05, spacing=0.2, height=0.3), linewidth=0.6, regrid_shape=40,
                transform=data_crs)
    else:
        expanded_u=expand_xr_longitudes(data['u850'],periodic_add=60)
        expanded_v=expand_xr_longitudes(data['v850'],periodic_add=60)
        q=plt.barbs(expanded_u.longitude, expanded_u.latitude,  
                    expanded_u, expanded_v, regrid_shape=20,
                    length=5, fill_empty=True, pivot='middle',#flagcolor=[.3,.3,.3],barbcolor=[.3,.3,.3],
                    sizes=dict(emptybarb=0.05, spacing=0.2, height=0.3),linewidth=0.9,
                    transform=data_crs)
                  
    #ax.add_feature(LAND,facecolor='lightgrey')
    ax.coastlines(linewidth=2,color='black')
    
    if not regional_grid:
        gl = ax.gridlines(crs=data_crs, draw_labels=True,
                          linewidth=0.6, color='lightgrey', alpha=0.5, linestyle='--')
    else:
        gl = ax.gridlines(crs=data_crs, draw_labels=True, x_inline=False, 
                          y_inline=False, linewidth=0.33, color='k',alpha=0.5)
    if not regional_grid:
        xlocators=list(np.arange(-180,190,30))
    else:
        xlocators=list(np.arange(-180,190,10))
    gl.xlocator = mticker.FixedLocator(xlocators)
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.top_labels = False
    gl.right_labels = False
    if not regional_grid:
        gl.bottom_labels = False 
    if regional_grid:
        gl.xlabel_style = gl.ylabel_style ={'size': 8, 'color': 'black','rotation': 0}
    else:
        gl.xlabel_style = gl.ylabel_style ={'size': 8, 'color': 'black'}
    
    plt.title('850hPa GPH (black contours) | 850-925hPa Cyc. Rel. Vort. (grey) | Equiv. Pot. Temp (shading)\n'+model_name +' Forecast | Init: ' + dstr_init_long +' | Valid: '+dstr_long, 
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
    
    outfile=outpath+model_name+'_'+name+'_VortPTE850_'+str(fignum)+'.jpg'
    plt.savefig(outfile, dpi=dpi)
    crop(outfile,padding=30)
    plt.close('all')
    #plt.show()

def plot_QvectPTE(outpath,dt,init_dt,fignum,indata,name='Australia',model_name='GFS',dpi=150,**kwargs):
    
    plot_extent,centlon,figsize,barblength,proj,southern_hemisphere,regional_grid=get_domain_settings(name)
    data=indata.copy()
    
    for k in data.keys():
        if k=='lats':
            data[k]=data[k].sel(latitude=slice(plot_extent[3]+10,plot_extent[2]-5))
        elif k=='lons':
            None
        else:
            data[k]=data[k].sel(latitude=slice(plot_extent[3]+10,plot_extent[2]-5))
    
    dstr=dt.strftime('%Y%m%d_%H')
    dstr_long=dt.strftime('%H%M UTC %d %b %Y')
    dstr_init_long=init_dt.strftime('%H%M UTC %d %b %Y')
    
    fig=plt.figure(figsize=figsize)
    data_crs=ccrs.PlateCarree()
    
    ax=plt.axes(projection=proj)
    ax.set_extent(plot_extent,crs=data_crs)
    if not regional_grid:
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
    lon2d, lat2d = np.meshgrid(data['pte850'].longitude.values, data['pte850'].latitude.values)
    cyclic_data, cyclic_lon2d, cyclic_lat2d = add_cyclic(data['pte850'].values, x=lon2d, y=lat2d)
    cf=ax.contourf(cyclic_lon2d, cyclic_lat2d, cyclic_data,
                   levels=plot_levels,cmap=jet_purple_cmap,extend='both',
                   transform=data_crs)
    
    plot_levels = list(range(1000,3000,30))
    lon2d, lat2d = np.meshgrid(data['z850'].longitude.values, data['z850'].latitude.values)
    cyclic_data, cyclic_lon2d, cyclic_lat2d = add_cyclic(data['z850'].values, x=lon2d, y=lat2d)
    c=ax.contour(cyclic_lon2d, cyclic_lat2d, cyclic_data,
                 levels=plot_levels,colors='black',linewidths=1.5,
                 transform=data_crs)
    fmt = '%i'
    ax.clabel(c, c.levels, inline=True, fmt=fmt, fontsize=8)  

    plot_levels = [-1,-0.75,-0.5]
    lon2d, lat2d = np.meshgrid(data['wMID'].longitude.values, data['wMID'].latitude.values)
    cyclic_data, cyclic_lon2d, cyclic_lat2d = add_cyclic(data['wMID'].values, x=lon2d, y=lat2d)
    c1=ax.contour(cyclic_lon2d, cyclic_lat2d, cyclic_data,
                  levels=plot_levels,colors='grey',linestyles='-',
                  transform=data_crs)
    
    if not regional_grid:
        lon2d, lat2d = np.meshgrid(data['uqvect850'].longitude.values, data['uqvect850'].latitude.values)
        cyclic_u, cyclic_lon2d, cyclic_lat2d = add_cyclic(data['uqvect850'].values, x=lon2d, y=lat2d)
        cyclic_v, cyclic_lon2d, cyclic_lat2d = add_cyclic(data['vqvect850'].values, x=lon2d, y=lat2d)
        q=ax.quiver(cyclic_lon2d, cyclic_lat2d, cyclic_u, cyclic_v,
                    angles="xy", regrid_shape=40,
                    transform=data_crs, zorder=1e2)
    else:
        expanded_u=expand_xr_longitudes(data['ujet300'],periodic_add=60)
        expanded_v=expand_xr_longitudes(data['vjet300'],periodic_add=60)
        q=ax.quiver(expanded_u.longitude.values, expanded_u.latitude.values, 
                     expanded_u, expanded_v, regrid_shape=30,
                      scale=5e-18,scale_units='xy',width=0.004,#minshaft=2,minlength=0.04,
                      #headaxislength=10,#minshaft=2,#minlength=0.04,
                      transform=data_crs,zorder=3)
                  
    ax.add_feature(LAND,facecolor='lightgrey')
    ax.coastlines(linewidth=1.75,color='black')
    
    if not regional_grid:
        gl = ax.gridlines(crs=data_crs, draw_labels=True,
                          linewidth=0.6, color='lightgrey', alpha=0.5, linestyle='--')
    else:
        gl = ax.gridlines(crs=data_crs, draw_labels=True, x_inline=False, 
                          y_inline=False, linewidth=0.33, color='k',alpha=0.5)
    if not regional_grid:
        xlocators=list(np.arange(-180,190,30))
    else:
        xlocators=list(np.arange(-180,190,10))
    gl.xlocator = mticker.FixedLocator(xlocators)
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.top_labels = False
    gl.right_labels = False
    if not regional_grid:
        gl.bottom_labels = False 
    if regional_grid:
        gl.xlabel_style = gl.ylabel_style ={'size': 8, 'color': 'black','rotation': 0}
    else:
        gl.xlabel_style = gl.ylabel_style ={'size': 8, 'color': 'black'}
    
    plt.title('850hPa GPH (black contours), Q-vectors (quivers), Equiv. Pot. Temp (shading) | 400-700hPa Up. Vert. Vel. (grey contours)\n'+model_name +' Forecast | Init: ' + dstr_init_long +' | Valid: '+dstr_long, 
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
    #plt.show()

def plot_upper(outpath,dt,init_dt,fignum,indata,name='Australia',model_name='GFS',dpi=150,**kwargs):
    
    plot_extent,centlon,figsize,barblength,proj,southern_hemisphere,regional_grid=get_domain_settings(name)
    data=indata.copy()
    
    for k in data.keys():
        print(k)
        if k=='lats':
            data[k]=data[k].sel(latitude=slice(plot_extent[3]+10,plot_extent[2]-5))
        elif k=='lons':
            None
        else:
            data[k]=data[k].sel(latitude=slice(plot_extent[3]+10,plot_extent[2]-5))
    
    dstr=dt.strftime('%Y%m%d_%H')
    dstr_long=dt.strftime('%H%M UTC %d %b %Y')
    dstr_init_long=init_dt.strftime('%H%M UTC %d %b %Y')
    
    fig=plt.figure(figsize=figsize)
    data_crs=ccrs.PlateCarree()
    
    ax=plt.axes(projection=proj)
    ax.set_extent(plot_extent,crs=data_crs)
    
    if not regional_grid:
        ax.set_boundary(map_circle, transform=ax.transAxes)
        
    plot_levels = [50,60,70,80,90,100,120,140,160]
    lon2d, lat2d = np.meshgrid(data['jet300'].longitude.values, data['jet300'].latitude.values)
    cyclic_data, cyclic_lon2d, cyclic_lat2d = add_cyclic(data['jet300'].values, x=lon2d, y=lat2d)
    cf=ax.contourf(cyclic_lon2d, cyclic_lat2d, cyclic_data*1.94384,
                   levels=plot_levels,cmap='Blues',extend='max',
                   transform=data_crs)
    
    plot_levels = list(range(800,1400,4))
    lon2d, lat2d = np.meshgrid(data['mslp'].longitude.values, data['mslp'].latitude.values)
    cyclic_data, cyclic_lon2d, cyclic_lat2d = add_cyclic(data['mslp'].values, x=lon2d, y=lat2d)
    cmslp=ax.contour(cyclic_lon2d, cyclic_lat2d, cyclic_data,
                     levels=plot_levels,colors='grey',linewidths=1,linestyles='--',
                     transform=data_crs)
    
    #plot_levels = [-30e-1 -25e-1 -20e-1 -15e-1, -10e-1, -5e-1]
    plot_levels = [-0.5,-0.4,-0.3,-0.2]
    lon2d, lat2d = np.meshgrid(data['wMID'].longitude.values, data['wMID'].latitude.values)
    cyclic_data, cyclic_lon2d, cyclic_lat2d = add_cyclic(data['wMID'].values, x=lon2d, y=lat2d)
    c1=ax.contourf(cyclic_lon2d, cyclic_lat2d, cyclic_data, 
                   levels=plot_levels, cmap='Reds_r',extend='min',
                   transform=data_crs)
    
    plot_levels = list(range(4000,6100,50))
    lon2d, lat2d = np.meshgrid(data['z500'].longitude.values, data['z500'].latitude.values)
    cyclic_data, cyclic_lon2d, cyclic_lat2d = add_cyclic(data['z500'].values, x=lon2d, y=lat2d)
    c=ax.contour(cyclic_lon2d, cyclic_lat2d, cyclic_data, 
                 levels=plot_levels,colors='black',
                 linewidths=0.8,transform=data_crs)
    
    if not regional_grid:
        lon2d, lat2d = np.meshgrid(data['ujet300'].longitude.values, data['ujet300'].latitude.values)
        cyclic_u, cyclic_lon2d, cyclic_lat2d = add_cyclic(data['ujet300'].values, x=lon2d, y=lat2d)
        cyclic_v, cyclic_lon2d, cyclic_lat2d = add_cyclic(data['vjet300'].values, x=lon2d, y=lat2d)
        q=ax.quiver(cyclic_lon2d, cyclic_lat2d, cyclic_u, cyclic_v,
                    angles="xy", transform=data_crs, regrid_shape=40)
    else:
        expanded_u=expand_xr_longitudes(data['ujet300'],periodic_add=60)
        expanded_v=expand_xr_longitudes(data['vjet300'],periodic_add=60)
        q=ax.quiver(expanded_u.longitude.values, expanded_u.latitude.values, 
                     expanded_u, expanded_v, regrid_shape=20,
                      scale=1e-4,scale_units='xy',width=0.004,#minshaft=2,# minlength=1,
                      transform=data_crs)
                  
    ax.add_feature(LAND,facecolor='lightgrey')
    ax.coastlines(linewidths=0.4)
    
    if not regional_grid:
        gl = ax.gridlines(crs=data_crs, draw_labels=True,
                          linewidth=0.6, color='lightgrey', alpha=0.5, linestyle='--')
    else:
        gl = ax.gridlines(crs=data_crs, draw_labels=True, x_inline=False, 
                          y_inline=False, linewidth=0.33, color='k',alpha=0.5)
    if not regional_grid:
        xlocators=list(np.arange(-180,190,30))
    else:
        xlocators=list(np.arange(-180,190,10))
        ylocators=list(np.arange(-80,80,10))
        gl.ylocator = mticker.FixedLocator(ylocators)
    gl.xlocator = mticker.FixedLocator(xlocators)
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.top_labels = False
    gl.right_labels = False
    if not regional_grid:
        gl.bottom_labels = False 
    if regional_grid:
        gl.xlabel_style = gl.ylabel_style ={'size': 8, 'color': 'black','rotation': 0}
    else:
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
    gc.collect()
    
def plot_irrotPV(outpath,dt,init_dt,fignum,indata,name='Australia',model_name='GFS',dpi=150,**kwargs):
    
    plot_extent,centlon,figsize,barblength,proj,southern_hemisphere,regional_grid=get_domain_settings(name)
    data=indata.copy()
    
    for k in data.keys():
        if k=='lats':
            data[k]=data[k].sel(latitude=slice(plot_extent[3]+10,plot_extent[2]-5))
        elif k=='lons' or k=='levs':
            None
        else:
            data[k]=data[k].sel(latitude=slice(plot_extent[3]+10,plot_extent[2]-5))
    
    dstr=dt.strftime('%Y%m%d_%H')
    dstr_long=dt.strftime('%H%M UTC %d %b %Y')
    dstr_init_long=init_dt.strftime('%H%M UTC %d %b %Y')
    
    fig=plt.figure(figsize=figsize)
    data_crs=ccrs.PlateCarree()
    
    ax=plt.axes(projection=proj)
    ax.set_extent(plot_extent,crs=data_crs)
    
    if not regional_grid:
        ax.set_boundary(map_circle, transform=ax.transAxes)
        
    plot_levels = list(range(20,80,5))
    lon2d, lat2d = np.meshgrid(data['pwat'].longitude.values, data['pwat'].latitude.values)
    cyclic_data, cyclic_lon2d, cyclic_lat2d = add_cyclic(data['pwat'].values, x=lon2d, y=lat2d)
    cf_pw=ax.contourf(cyclic_lon2d, cyclic_lat2d, cyclic_data,
                   levels=plot_levels,cmap='GnBu',extend='max',
                   transform=data_crs)
    
    jet_cmap = plt.cm.get_cmap('RdPu', 16)(range(5,16))
    jet_cmap = ListedColormap(jet_cmap)
    plot_levels = [30,35,40,45,50,60,70,80,90,100,120]
    lon2d, lat2d = np.meshgrid(data['jet300'].longitude.values, data['jet300'].latitude.values)
    cyclic_data, cyclic_lon2d, cyclic_lat2d = add_cyclic(data['jet300'].values, x=lon2d, y=lat2d)
    cf=ax.contourf(cyclic_lon2d, cyclic_lat2d, cyclic_data,
                   levels=plot_levels,cmap=jet_cmap,extend='max',
                   transform=data_crs)

    plot_levels = [-1,-0.75,-0.5]
    lon2d, lat2d = np.meshgrid(data['wMID'].longitude.values, data['wMID'].latitude.values)
    cyclic_data, cyclic_lon2d, cyclic_lat2d = add_cyclic(data['wMID'].values, x=lon2d, y=lat2d)
    c1=ax.contour(cyclic_lon2d, cyclic_lat2d, cyclic_data,
                  levels=plot_levels,colors='red',linestyles='-',
                   transform=data_crs)

    if southern_hemisphere:
        plot_levels = list(range(-15,0,1))
    else:
        plot_levels = list(range(1,16,1))
    lon2d, lat2d = np.meshgrid(data['pv_iso_upper'].longitude.values, data['pv_iso_upper'].latitude.values)
    cyclic_data, cyclic_lon2d, cyclic_lat2d = add_cyclic(data['pv_iso_upper'].values, x=lon2d, y=lat2d)
    cpv=ax.contour(cyclic_lon2d, cyclic_lat2d, gaussian_filter(cyclic_data*1e6,sigma=2),
                     levels=plot_levels,colors='black',linestyles='-',
                     linewidths=1.5,transform=data_crs)
    fmt = '%i'
    ax.clabel(cpv, cpv.levels[1::2], inline=True, fmt=fmt, fontsize=14)  

    if not regional_grid:
        lon2d, lat2d = np.meshgrid(data['uchi'].longitude.values, data['uchi'].latitude.values)
        cyclic_u, cyclic_lon2d, cyclic_lat2d = add_cyclic(data['uchi'].values, x=lon2d, y=lat2d)
        cyclic_v, cyclic_lon2d, cyclic_lat2d = add_cyclic(data['vchi'].values, x=lon2d, y=lat2d)
        q=plt.quiver(cyclic_lon2d, cyclic_lat2d, cyclic_u, cyclic_v, 
                 angles="xy", transform=data_crs, regrid_shape=40)
    else:
        expanded_u=expand_xr_longitudes(data['uchi'],periodic_add=60)
        expanded_v=expand_xr_longitudes(data['vchi'],periodic_add=60)
        q=plt.quiver(expanded_u.longitude, expanded_u.latitude, expanded_u, expanded_v,
                     regrid_shape=22,scale=20e-6,scale_units='xy',#width=0.0035,#minshaft=2,#minlength=2,
                     width=0.0405,minshaft=2,minlength=0.04,
                     transform=data_crs)
                  
    ax.add_feature(LAND,facecolor='lightgrey')
    ax.coastlines(linewidths=0.4,zorder=1e4)
    
    if not regional_grid:
        gl = ax.gridlines(crs=data_crs, draw_labels=True,
                          linewidth=0.6, color='lightgrey', alpha=0.5, linestyle='--')
    else:
        gl = ax.gridlines(crs=data_crs, draw_labels=True, x_inline=False, 
                          y_inline=False, linewidth=0.33, color='k',alpha=0.5)
    if not regional_grid:
        xlocators=list(np.arange(-180,190,30))
    else:
        xlocators=list(np.arange(-180,190,10))
    gl.xlocator = mticker.FixedLocator(xlocators)
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.top_labels = False
    gl.right_labels = False
    if not regional_grid:
        gl.bottom_labels = False 
    if regional_grid:
        gl.xlabel_style = gl.ylabel_style ={'size': 8, 'color': 'black','rotation': 0}
    else:
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
    
def plot_IVT(outpath,dt,init_dt,fignum,indata,name='Australia',model_name='GFS',dpi=150,**kwargs):
    
    plot_extent,centlon,figsize,barblength,proj,southern_hemisphere,regional_grid=get_domain_settings(name)
    data=indata.copy()
    
    for k in data.keys():
        if k=='lats':
            data[k]=data[k].sel(latitude=slice(plot_extent[3]+10,plot_extent[2]-5))
        elif k=='lons':
            None
        else:
            data[k]=data[k].sel(latitude=slice(plot_extent[3]+10,plot_extent[2]-5))
    
    dstr=dt.strftime('%Y%m%d_%H')
    dstr_long=dt.strftime('%H%M UTC %d %b %Y')
    dstr_init_long=init_dt.strftime('%H%M UTC %d %b %Y')
    
    fig=plt.figure(figsize=figsize)
    data_crs=ccrs.PlateCarree()
    
    ax=plt.axes(projection=proj)
    ax.set_extent(plot_extent,crs=data_crs)
    if not regional_grid:
        ax.set_boundary(map_circle, transform=ax.transAxes)

    if southern_hemisphere:
        plot_levels = np.arange(-7,-4.5,0.5)
    else:
        plot_levels = np.arange(5,7.5,0.5)
    
    plot_levels = [250,300,400,500,600,700,800,1000,1200,1400,1600,1800]
    ivt_cmap = plt.cm.get_cmap('cool', len(plot_levels))
    lon2d, lat2d = np.meshgrid(data['IVT'].longitude.values, data['IVT'].latitude.values)
    cyclic_data, cyclic_lon2d, cyclic_lat2d = add_cyclic(data['IVT'].values, x=lon2d, y=lat2d)
    cf=ax.contourf(cyclic_lon2d, cyclic_lat2d, cyclic_data,
                   levels=plot_levels, norm=BoundaryNorm(plot_levels,len(plot_levels)),
                   cmap=ivt_cmap,extend='max',
                   transform=data_crs,zorder=1)
    c=ax.contour(cyclic_lon2d, cyclic_lat2d, cyclic_data,
                 levels=plot_levels,colors='darkgrey',
                 linewidths=0.4,transform=data_crs,zorder=1)
    
    if not regional_grid:
        lon2d, lat2d = np.meshgrid(data['uIVT'].longitude.values, data['uIVT'].latitude.values)
        cyclic_u, cyclic_lon2d, cyclic_lat2d = add_cyclic(data['uIVT'].values, x=lon2d, y=lat2d)
        cyclic_v, cyclic_lon2d, cyclic_lat2d = add_cyclic(data['vIVT'].values, x=lon2d, y=lat2d)
        q=ax.quiver(cyclic_lon2d, cyclic_lat2d, cyclic_u, cyclic_v,
                    angles="xy", transform=data_crs, regrid_shape=40,zorder=3)
    else:
        expanded_u=expand_xr_longitudes(data['uIVT'],periodic_add=60)
        expanded_v=expand_xr_longitudes(data['vIVT'],periodic_add=60)
        q=plt.quiver(expanded_u.longitude, expanded_u.latitude, expanded_u, expanded_v,
                     regrid_shape=20,scale=1.25e-3,scale_units='xy',width=0.0405,minshaft=2,minlength=0.04,
                     transform=data_crs,zorder=3)
    
    plot_levels = list(range(0,6000,30))
    lon2d, lat2d = np.meshgrid(data['z700'].longitude.values, data['z700'].latitude.values)
    cyclic_data, cyclic_lon2d, cyclic_lat2d = add_cyclic(data['z700'].values, x=lon2d, y=lat2d) 
    c=ax.contour(cyclic_lon2d, cyclic_lat2d, cyclic_data,
                 levels=plot_levels,colors='black',linewidths=0.6,
                 transform=data_crs,zorder=3)
    fmt = '%i'
    ax.clabel(c, c.levels, inline=True, fmt=fmt, fontsize=8)
    
    ax.add_feature(LAND,facecolor='lightgrey')
    ax.coastlines(linewidths=0.4)
    
    if not regional_grid:
        gl = ax.gridlines(crs=data_crs, draw_labels=True,
                          linewidth=0.6, color='lightgrey', alpha=0.5, linestyle='--')
    else:
        gl = ax.gridlines(crs=data_crs, draw_labels=True, x_inline=False, 
                          y_inline=False, linewidth=0.33, color='k',alpha=0.5)
    if not regional_grid:
        xlocators=list(np.arange(-180,190,30))
    else:
        xlocators=list(np.arange(-180,190,10))
    gl.xlocator = mticker.FixedLocator(xlocators)
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.top_labels = False
    gl.right_labels = False
    if not regional_grid:
        gl.bottom_labels = False 
    if regional_grid:
        gl.xlabel_style = gl.ylabel_style ={'size': 8, 'color': 'black','rotation': 0}
    else:
        gl.xlabel_style = gl.ylabel_style ={'size': 8, 'color': 'black'}

    plt.title('700hPa GPH (black contours) | IVT (shaded, arrows) \n'+model_name +' Forecast | Init: ' + dstr_init_long +' | Valid: '+dstr_long, 
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
             bbox=dict(facecolor='white', alpha=0.95),zorder=10)
    
    outfile=outpath+model_name+'_'+name+'_IVT_'+str(fignum)+'.jpg'
    plt.savefig(outfile, dpi=dpi)
    crop(outfile)
    plt.close('all') 
    
def plot_IPV(outpath,dt,init_dt,fignum,plot_level,indata,name='Australia',model_name='GFS',dpi=150,**kwargs):
    
    plot_extent,centlon,figsize,barblength,proj,southern_hemisphere,regional_grid=get_domain_settings(name)
    data=indata.copy()
    
    for k in data.keys():
        if k=='lats':
            data[k]=data[k].sel(latitude=slice(plot_extent[3]+10,plot_extent[2]-5))
        elif k=='lons' or k=='levs':
            None
        else:
            data[k]=data[k].sel(latitude=slice(plot_extent[3]+10,plot_extent[2]-5))
    
    dstr=dt.strftime('%Y%m%d_%H')
    dstr_long=dt.strftime('%H%M UTC %d %b %Y')
    dstr_init_long=init_dt.strftime('%H%M UTC %d %b %Y')
    
    fig=plt.figure(figsize=figsize)
    data_crs=ccrs.PlateCarree()
    
    ax=plt.axes(projection=proj)
    ax.set_extent(plot_extent,crs=data_crs)
    if not regional_grid:
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

    lon2d, lat2d = np.meshgrid(data['pv'+str(plot_level)].longitude.values, data['pv'+str(plot_level)].latitude.values)
    cyclic_data, cyclic_lon2d, cyclic_lat2d = add_cyclic(data['pv'+str(plot_level)].values, x=lon2d, y=lat2d)
    cf=plt.contourf(cyclic_lon2d, cyclic_lat2d, gaussian_filter(cyclic_data/1e-6,sigma=1),
                    levels=pv_clevs,cmap=pv_cmap,norm=BoundaryNorm(pv_clevs,len(pv_clevs)),
                    transform=data_crs,extend=PVextend)
    c=ax.contour(cyclic_lon2d, cyclic_lat2d, gaussian_filter(cyclic_data/1e-6,sigma=1),
                 levels=pv_clevs[0:-1],colors='grey',linewidths=0.4,linestyles='-',
                 transform=data_crs)
    
    if not regional_grid:
        lon2d, lat2d = np.meshgrid(data['u'+str(plot_level)].longitude.values, data['u'+str(plot_level)].latitude.values)
        cyclic_u, cyclic_lon2d, cyclic_lat2d = add_cyclic(data['u'+str(plot_level)].values, x=lon2d, y=lat2d)
        cyclic_v, cyclic_lon2d, cyclic_lat2d = add_cyclic(data['v'+str(plot_level)].values, x=lon2d, y=lat2d)
        q=ax.quiver(cyclic_lon2d, cyclic_lat2d, cyclic_u, cyclic_v,
                    angles="xy", transform=data_crs, regrid_shape=40)
    else:
        expanded_u=expand_xr_longitudes(data['u'+str(plot_level)],periodic_add=60)
        expanded_v=expand_xr_longitudes(data['v'+str(plot_level)],periodic_add=60)
        q=plt.quiver(expanded_u.longitude, expanded_u.latitude, expanded_u, expanded_v,
                     regrid_shape=22,scale=135e-6,scale_units='xy',width=0.004,#minshaft=2,#minlength=2,
                     transform=data_crs)
    
    ax.add_feature(LAND,facecolor='lightgrey')
    ax.coastlines(linewidths=0.4)
    
    if not regional_grid:
        gl = ax.gridlines(crs=data_crs, draw_labels=True,
                          linewidth=0.6, color='lightgrey', alpha=0.5, linestyle='--')
    else:
        gl = ax.gridlines(crs=data_crs, draw_labels=True, x_inline=False, 
                          y_inline=False, linewidth=0.33, color='k',alpha=0.5)
    if not regional_grid:
        xlocators=list(np.arange(-180,190,30))
    else:
        xlocators=list(np.arange(-180,190,10))
    gl.xlocator = mticker.FixedLocator(xlocators)
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.top_labels = False
    gl.right_labels = False
    if not regional_grid:
        gl.bottom_labels = False 
    if regional_grid:
        gl.xlabel_style = gl.ylabel_style ={'size': 8, 'color': 'black','rotation': 0}
    else:
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
    plt.close('all') 

def plot_thickness(outpath,dt,init_dt,fignum,indata,name='Australia',model_name='GFS',dpi=150,**kwargs):

    plot_extent,centlon,figsize,barblength,proj,southern_hemisphere,regional_grid=get_domain_settings(name)
    data=indata.copy()
    
    for k in data.keys():
        if k=='lats':
            data[k]=data[k].sel(latitude=slice(plot_extent[3]+10,plot_extent[2]-5))
        elif k=='lons':
            None
        else:
            data[k]=data[k].sel(latitude=slice(plot_extent[3]+10,plot_extent[2]-5))
    
    dstr=dt.strftime('%Y%m%d_%H')
    dstr_long=dt.strftime('%H%M UTC %d %b %Y')
    dstr_init_long=init_dt.strftime('%H%M UTC %d %b %Y')
    
    fig=plt.figure(figsize=figsize)
    data_crs=ccrs.PlateCarree()
    
    ax=plt.axes(projection=proj)
    ax.set_extent(plot_extent,crs=data_crs)
    if not regional_grid:
        ax.set_boundary(map_circle, transform=ax.transAxes)

    if southern_hemisphere:
        plot_levels = np.arange(-30,-4,2)
        extend_cmap='min'
    else:
        plot_levels = np.arange(6,32,2)
        extend_cmap='max'

    cmap = plt.cm.get_cmap('BuPu', len(plot_levels)+10)(range(10,len(plot_levels)+10))
    if southern_hemisphere:
        cmap=cmap[::-1]
    cmap = ListedColormap(cmap)

    lon2d, lat2d = np.meshgrid(data['vort500'].longitude.values, data['vort500'].latitude.values)
    cyclic_data, cyclic_lon2d, cyclic_lat2d = add_cyclic(data['vort500'].values, x=lon2d, y=lat2d)
    if not regional_grid:
        cf=ax.contourf(cyclic_lon2d, cyclic_lat2d, gaussian_filter(cyclic_data*1e5,sigma=2),levels=plot_levels,
                norm=BoundaryNorm(plot_levels,len(plot_levels)),cmap=cmap,extend=extend_cmap,
                transform=data_crs)
    else:
        cf=ax.contourf(cyclic_lon2d, cyclic_lat2d, cyclic_data*1e5, levels=plot_levels,
                norm=BoundaryNorm(plot_levels,len(plot_levels)),cmap=cmap,extend=extend_cmap,
                transform=data_crs)
    
    plot_levels = list(range(4000,6100,50))
    lon2d, lat2d = np.meshgrid(data['z500'].longitude.values, data['z500'].latitude.values)
    cyclic_data, cyclic_lon2d, cyclic_lat2d = add_cyclic(data['z500'].values, x=lon2d, y=lat2d)
    c=ax.contour(cyclic_lon2d, cyclic_lat2d, cyclic_data,
                 levels=plot_levels,colors='black',
                 linewidths=0.8,transform=data_crs)
    
    plot_levels = [-0.8,-0.7,-0.6,-0.5,-0.4,-0.3]
    lon2d, lat2d = np.meshgrid(data['wMID'].longitude.values, data['wMID'].latitude.values)
    cyclic_data, cyclic_lon2d, cyclic_lat2d = add_cyclic(data['wMID'].values, x=lon2d, y=lat2d)
    c1=ax.contour(cyclic_lon2d, cyclic_lat2d, cyclic_data,
                  levels=plot_levels,colors='darkorange',linewidths=0.8,linestyles='-',
                  transform=data_crs)
    
    plot_levels = list(range(546,700,6))
    lon2d, lat2d = np.meshgrid(data['thickness'].longitude.values, data['thickness'].latitude.values)
    cyclic_data, cyclic_lon2d, cyclic_lat2d = add_cyclic(data['thickness'].values, x=lon2d, y=lat2d)
    c=ax.contour(cyclic_lon2d, cyclic_lat2d, cyclic_data,
                 levels=plot_levels,colors='red',linestyles='--',linewidths=1.5,
                 transform=data_crs)
    fmt = '%i'
    ax.clabel(c, c.levels, inline=True, fmt=fmt, fontsize=10, inline_spacing=0)
    plot_levels = list(range(546-120,546,6))
    c=ax.contour(cyclic_lon2d, cyclic_lat2d, cyclic_data,
                 levels=plot_levels,colors='blue',linestyles='--',linewidths=1.5,
                 transform=data_crs)
    fmt = '%i'
    ax.clabel(c, c.levels, inline=True, fmt=fmt, fontsize=10, inline_spacing=0)
    
    ax.add_feature(LAND,facecolor='lightgrey')
    ax.coastlines(linewidths=0.4)
    
    if not regional_grid:
        gl = ax.gridlines(crs=data_crs, draw_labels=True,
                          linewidth=0.6, color='lightgrey', alpha=0.5, linestyle='--')
    else:
        gl = ax.gridlines(crs=data_crs, draw_labels=True, x_inline=False, 
                          y_inline=False, linewidth=0.33, color='k',alpha=0.5)
    if not regional_grid:
        xlocators=list(np.arange(-180,190,30))
    else:
        xlocators=list(np.arange(-180,190,10))
    gl.xlocator = mticker.FixedLocator(xlocators)
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.top_labels = False
    gl.right_labels = False
    if not regional_grid:
        gl.bottom_labels = False 
    if regional_grid:
        gl.xlabel_style = gl.ylabel_style ={'size': 8, 'color': 'black','rotation': 0}
    else:
        gl.xlabel_style = gl.ylabel_style ={'size': 8, 'color': 'black'}
    
    plt.title('500-1000hPa Thickness (red/blue, dam) | 500hPa Cyc. Rel. Vort. (shading), GPH (black) | 400-700hPa Up. Vert. Vel. (orange)\n'+model_name +' Forecast | Init: ' + dstr_init_long +' | Valid: '+dstr_long, 
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
    #plt.show()

def plot_DT(outpath,dt,init_dt,fignum,indata,name='Australia',model_name='GFS',dpi=150,**kwargs):
    
    plot_extent,centlon,figsize,barblength,proj,southern_hemisphere,regional_grid=get_domain_settings(name)
    data=indata.copy()
    
    for k in data.keys():
        if k=='lats':
            data[k]=data[k].sel(latitude=slice(plot_extent[3]+10,plot_extent[2]-5))
        elif k=='lons':
            None
        else:
            data[k]=data[k].sel(latitude=slice(plot_extent[3]+10,plot_extent[2]-5))
    
    dstr=dt.strftime('%Y%m%d_%H')
    dstr_long=dt.strftime('%H%M UTC %d %b %Y')
    dstr_init_long=init_dt.strftime('%H%M UTC %d %b %Y')
    
    fig=plt.figure(figsize=figsize)
    data_crs=ccrs.PlateCarree()
    
    ax=plt.axes(projection=proj)
    ax.set_extent(plot_extent,crs=data_crs)
    if not regional_grid:
        ax.set_boundary(map_circle, transform=ax.transAxes)
    
    plot_levels = np.arange(260,400,10)#[250,300,400,500,600,700,800,1000,1200,1400,1600,1800]
    dtPTsmooth=gaussian_filter(data['dtPT'],sigma=1)
    cmap = plt.cm.get_cmap('plasma', len(plot_levels))
    cf=ax.contourf(data['dtPT'].longitude.values,data['dtPT'].latitude.values,
                   dtPTsmooth,levels=plot_levels,
            norm=BoundaryNorm(plot_levels,len(plot_levels)),cmap=cmap,extend='both',
            transform=data_crs)

    if southern_hemisphere:
        plot_levels = np.arange(-7,-4.5,0.5)
    else:
        plot_levels = np.arange(5,7.5,0.5)
    c=ax.contour(data['vortLOW'].longitude.values,data['vortLOW'].latitude.values,
                 data['vortLOW']*1e5,levels=plot_levels,
            colors='black',linestyles='-',linewidths=0.8,
            #norm=BoundaryNorm(plot_levels,len(plot_levels)),cmap=cmap,extend='min',
            transform=data_crs)
    
    ax.add_feature(LAND,facecolor='lightgrey')
    ax.coastlines(linewidth=1.25,color='grey')
    
    if not regional_grid:
        gl = ax.gridlines(crs=data_crs, draw_labels=True,
                          linewidth=0.6, color='lightgrey', alpha=0.5, linestyle='--')
    else:
        gl = ax.gridlines(crs=data_crs, draw_labels=True, x_inline=False, 
                          y_inline=False, linewidth=0.33, color='k',alpha=0.5)
    if not regional_grid:
        xlocators=list(np.arange(-180,190,30))
    else:
        xlocators=list(np.arange(-180,190,10))
    gl.xlocator = mticker.FixedLocator(xlocators)
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.top_labels = False
    gl.right_labels = False
    
    if not regional_grid:
        gl.bottom_labels = False 
    if regional_grid:
        gl.xlabel_style = gl.ylabel_style ={'size': 8, 'color': 'black','rotation': 0}
    else:
        gl.xlabel_style = gl.ylabel_style ={'size': 8, 'color': 'black'}
    
    plt.title('2PVU Potential Temperature (shading) | 850-925hPa Cyc. Rel. Vort. (black)\n'+model_name +' Forecast | Init: ' + dstr_init_long +' | Valid: '+dstr_long, 
              fontsize=10)
    
    ax_pos=[0.25,0.12,0.5,0.015] #[left, bottom, width, height]
    fig=plt.gcf()
    cbar_ax = fig.add_axes(ax_pos)
    cb=fig.colorbar(cf,orientation="horizontal", cax=cbar_ax)
    cb.ax.tick_params(labelsize=8)
    cb.set_label('Potential Temperature [K]', rotation=0, fontsize=8)
    
    copywrite_text='\xa9 Michael A. Barnes\nwww.weathermanbarnes.com'
    ax.text(0.01, 0.015, copywrite_text, fontsize=6, 
             horizontalalignment='left', verticalalignment='bottom', 
             transform=ax.transAxes,
             bbox=dict(facecolor='white', alpha=0.95),zorder=10)
    
    outfile=outpath+model_name+'_'+name+'_DT_'+str(fignum)+'.jpg'
    plt.savefig(outfile, dpi=dpi)
    crop(outfile)
    plt.close('all') 

def plot_cape(outpath,dt,init_dt,fignum,indata,name='Australia',incapetype='CAPE',model_name='GFS',dpi=150,**kwargs):
    plot_extent,centlon,figsize,barblength,proj,southern_hemisphere,regional_grid=get_domain_settings(name)
    data=indata.copy()
    
    for k in data.keys():
        if k=='lats':
            data[k]=data[k].sel(latitude=slice(plot_extent[3]+10,plot_extent[2]-5))
        elif k=='lons':
            None
        else:
            data[k]=data[k].sel(latitude=slice(plot_extent[3]+10,plot_extent[2]-5))
    
    dstr=dt.strftime('%Y%m%d_%H')
    dstr_long=dt.strftime('%H%M UTC %d %b %Y')
    dstr_init_long=init_dt.strftime('%H%M UTC %d %b %Y')
    
    fig=plt.figure(figsize=figsize)
    data_crs=ccrs.PlateCarree()
    
    ax=plt.axes(projection=proj)
    ax.set_extent(plot_extent,crs=data_crs)
    if not regional_grid:
        ax.set_boundary(map_circle, transform=ax.transAxes)
    
    plot_levels = np.arange(250,4250,250)
    lon2d, lat2d = np.meshgrid(data['cape'].longitude.values, data['cape'].latitude.values)
    cyclic_data, cyclic_lon2d, cyclic_lat2d = add_cyclic(data['cape'].values, x=lon2d, y=lat2d)
    #if not regional_grid:
    cf=ax.contourf(cyclic_lon2d, cyclic_lat2d, cyclic_data, levels=plot_levels,
            cmap='rainbow',extend='max',
            transform=data_crs)
    #else:
    #    cf=ax.contourf(cyclic_lon2d, cyclic_lat2d, cyclic_data, levels=plot_levels,
    #            cmap='rainbow',extend='max',
    #            transform=data_crs)
    
    plot_levels = list(range(1000,2500,30))
    lon2d, lat2d = np.meshgrid(data['z850'].longitude.values, data['z850'].latitude.values)
    cyclic_data, cyclic_lon2d, cyclic_lat2d = add_cyclic(data['z850'].values, x=lon2d, y=lat2d)
    c=ax.contour(cyclic_lon2d, cyclic_lat2d, cyclic_data,
                 levels=plot_levels,colors='black',
                 linewidths=0.8,transform=data_crs)
    
    fmt = '%i'
    ax.clabel(c, c.levels, inline=True, fmt=fmt, fontsize=10, inline_spacing=0)
    
    plot_levels = list(range(250,350,4))
    lon2d, lat2d = np.meshgrid(data['pt850'].longitude.values, data['pt850'].latitude.values)
    cyclic_data, cyclic_lon2d, cyclic_lat2d = add_cyclic(data['pt850'].values, x=lon2d, y=lat2d)
    c=ax.contour(cyclic_lon2d, cyclic_lat2d, cyclic_data,
                 levels=plot_levels,colors='red',linestyles='-',linewidths=0.8,
                 transform=data_crs)
    fmt = '%i'
    ax.clabel(c, c.levels, inline=True, fmt=fmt, fontsize=10, inline_spacing=0)
    plot_levels = list(range(546-120,546,6))
    #c=ax.contour(cyclic_lon2d, cyclic_lat2d, cyclic_data,
    #             levels=plot_levels,colors='blue',linestyles='--',linewidths=1.5,
    #             transform=data_crs)
    #fmt = '%i'
    #ax.clabel(c, c.levels, inline=True, fmt=fmt, fontsize=10, inline_spacing=0)
    
    ax.add_feature(LAND,facecolor='lightgrey')
    ax.coastlines(linewidths=0.4)
    
    if not regional_grid:
        gl = ax.gridlines(crs=data_crs, draw_labels=True,
                          linewidth=0.6, color='lightgrey', alpha=0.5, linestyle='--')
    else:
        gl = ax.gridlines(crs=data_crs, draw_labels=True, x_inline=False, 
                          y_inline=False, linewidth=0.33, color='k',alpha=0.5)
    if not regional_grid:
        xlocators=list(np.arange(-180,190,30))
    else:
        xlocators=list(np.arange(-180,190,10))
    gl.xlocator = mticker.FixedLocator(xlocators)
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.top_labels = False
    gl.right_labels = False
    if not regional_grid:
        gl.bottom_labels = False 
    if regional_grid:
        gl.xlabel_style = gl.ylabel_style ={'size': 8, 'color': 'black','rotation': 0}
    else:
        gl.xlabel_style = gl.ylabel_style ={'size': 8, 'color': 'black'}
    
    plt.title(incapetype+' (shading) | 850hPa GPH (black) | 850hPa Potential Temperature (red, K)\n'+model_name +' Forecast | Init: ' + dstr_init_long +' | Valid: '+dstr_long, 
              fontsize=10)
    
    ax_pos=[0.25,0.12,0.5,0.015] #[left, bottom, width, height]
    fig=plt.gcf()
    cbar_ax = fig.add_axes(ax_pos)
    cb=fig.colorbar(cf,orientation="horizontal", cax=cbar_ax)
    cb.ax.tick_params(labelsize=8)
    cb.set_label(incapetype+' [$J.kg^{-1}$]', rotation=0, fontsize=8)
    
    copywrite_text='\xa9 Michael A. Barnes\nwww.weathermanbarnes.com'
    ax.text(0.01, 0.015, copywrite_text, fontsize=6, 
             horizontalalignment='left', verticalalignment='bottom', 
             transform=ax.transAxes,
             bbox=dict(facecolor='white', alpha=0.95),zorder=10)
    
    outfile=outpath+model_name+'_'+name+'_CAPE_'+str(fignum)+'.jpg'
    plt.savefig(outfile, dpi=dpi)
    crop(outfile)
    plt.close('all') 
    #plt.show()




