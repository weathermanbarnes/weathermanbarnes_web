#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  7 20:57:50 2023

@author: mbar0087
"""

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

def calc_Vchi(u,v):#,lons,lats):
    print('Getting irrotational wind and absolute vorticity')
    #uwnd, uwnd_info = prep_data(u, 'yx')
    #vwnd, vwnd_info = prep_data(v, 'yx')
    #lats_new, uwnd, vwnd = order_latdim(lats, uwnd, vwnd)
    VW = VectorWind(u, v)
    uchi, vchi = VW.irrotationalcomponent()

    uchi=uchi.data
    vchi=vchi.data
    
    return uchi, vchi


def get_precip6h(inpath,dt):
    dstr=dt.strftime('%Y%m%d_%H')

    infile=inpath+'GFS_surface_' + dstr +'.grib2'
    pfile=inpath+'GFS_precip6H_' + dstr + '.nc'

    datafile=xr.load_dataset(infile, engine="cfgrib")
    pdatafile=xr.load_dataset(pfile)
    
    lats=datafile['latitude'].data
    lons=datafile['longitude'].data
    
    u10=datafile['u10'].data*1.94384 #in knots
    v10=datafile['v10'].data*1.94384 #in knots
    precip=pdatafile['tp'].data/10
    mslp=datafile['mslet'].data/100
    
    data={}
    data['lats']=lats
    data['lons']=lons
    data['u10']=u10
    data['v10']=v10
    data['precip']=precip
    data['mslp']=mslp
    
    return data

def get_IPV_data(inpath,dt,**kwargs):
    dstr=dt.strftime('%Y%m%d_%H')
    #dstr_long=(dt+relativedelta(hours=fhour)).strftime('%H%M UTC %d %b %Y')
    #dstr_init_long=dt.strftime('%H%M UTC %d %b %Y')
    
    infile=inpath+'GFS_surface_' + dstr + '.grib2'
    #BoMfile=inpath+dYMD+dYMD+'.grid'
    datafile=xr.load_dataset(infile, engine="cfgrib")
    mslp=datafile['mslet'].data/100
    
    infile=inpath+'GFS_' + dstr + '.grib2'
    datafile=xr.load_dataset(infile, engine="cfgrib",  
                             backend_kwargs={'filter_by_keys': {'typeOfLevel': 'isobaricInhPa'}})
    lats=datafile['latitude']
    lons=datafile['longitude']
    levs=datafile['isobaricInhPa']
    u=datafile['u']
    v=datafile['v']
    hgt=datafile['gh']
    t=datafile['t']
    
    pt=potential_temperature(levs, t)
    pv=potential_vorticity_baroclinic(pt, levs, u, v)
    pv=pv.rename('pv')
    ipv=isentropic_interpolation_as_dataset(np.array([320,330,350])*units.K, t,pv,u,v)
    
    #Subset arrays for plotting
    hgt300 = hgt[np.where(levs==300)[0][0],:,:]
    pv_iso_upper = pv.sel(isobaricInhPa=slice(300,200)).mean(dim='isobaricInhPa') 
    uchi,vchi=calc_Vchi(datafile['u'].sel(isobaricInhPa=slice(300,200)).mean(dim='isobaricInhPa'),
                        datafile['v'].sel(isobaricInhPa=slice(300,200)).mean(dim='isobaricInhPa'))
    
    pv_iso_upper=gaussian_filter(pv_iso_upper,sigma=1)

    data={}
    data['lats']=lats
    data['lons']=lons
    data['levs']=levs
    data['pv']=ipv
    data['theta']=pt
    
    data['pv320']=ipv.pv.sel(isentropic_level=320)
    data['u320']=ipv.u.sel(isentropic_level=320)
    data['v320']=ipv.v.sel(isentropic_level=320)
    
    data['pv330']=ipv.pv.sel(isentropic_level=330)
    data['u330']=ipv.u.sel(isentropic_level=330)
    data['v330']=ipv.v.sel(isentropic_level=330)
    
    data['pv350']=ipv.pv.sel(isentropic_level=350)#gaussian_filter(ipv.pv.sel(isentropic_level=350),sigma=1)
    data['u350']=ipv.u.sel(isentropic_level=350)
    data['v350']=ipv.v.sel(isentropic_level=350)
    
    data['pv_iso_upper']=pv_iso_upper
    data['uchi']=uchi
    data['vchi']=vchi
    
    return data

def get_IVT_data(inpath,dt):
    dstr=dt.strftime('%Y%m%d_%H')
    cp=1004.5
    r=2*cp/7
    
    infile=inpath+'GFS_' + dstr + '.grib2'
    datafile=xr.load_dataset(infile, engine="cfgrib",  
                             backend_kwargs={'filter_by_keys': {'typeOfLevel': 'isobaricInhPa'}})
    lats=datafile['latitude'][:].data
    lons=datafile['longitude'][:].data
    levs=datafile['isobaricInhPa'].data
    u=datafile['u'].data[0:np.where(levs==100)[0][0]]
    v=datafile['v'].data[0:np.where(levs==100)[0][0]]
    z700=datafile['gh'].data[np.where(levs==700)[0][0]]
    q=datafile['q'].data[0:np.where(levs==100)[0][0]]
    
    qu=q*u
    qv=q*v
    uIVT=(1/r)*np.sum(qu,axis=0)*(100000-10000)
    vIVT=(1/r)*np.sum(qv,axis=0)*(100000-10000)
    IVT=wind.wind_uv_to_spd(uIVT,vIVT)
    
    data={}
    data['lats']=lats
    data['lons']=lons
    data['z700']=z700
    data['uIVT']=uIVT
    data['vIVT']=vIVT
    data['IVT']=IVT
    
    return data

def get_upper_data(inpath,dt):
    dstr=dt.strftime('%Y%m%d_%H')
        
    infile=inpath+'GFS_surface_' + dstr +'.grib2'
    datafile=xr.load_dataset(infile, engine="cfgrib")
    #print(datafile)
    mslp=datafile['mslet'].data/100
    pwat=datafile['pwat'].data
    
    infile=inpath+'GFS_' + dstr +'.grib2'
    datafile=xr.load_dataset(infile, engine="cfgrib", 
                             backend_kwargs={'filter_by_keys': {'typeOfLevel': 'isobaricInhPa'}})
    levs=datafile['isobaricInhPa'].data
    z500=datafile['gh'].data[np.where(levs==500)[0][0]]
    u300=datafile['u'].data[np.where(levs==300)[0][0]]
    v300=datafile['v'].data[np.where(levs==300)[0][0]]
    spd300=wind.wind_uv_to_spd(u300,v300)
    
    ujet300=u300
    ujet300[spd300<50*0.514444]=np.nan
    vjet300=v300
    vjet300[spd300<50*0.514444]=np.nan
    jet300=spd300
    jet300[spd300<50*0.514444]=np.nan
    w=datafile['w'].data
    wMID=np.sum(w[np.where(levs==600)[0][0]:np.where(levs==400)[0][0]],axis=0)/300
    wMID[wMID<0]
    
    lats=datafile['latitude'][:].data
    lons=datafile['longitude'][:].data
    
    data={}
    data['lats']=lats
    data['lons']=lons
    data['z500']=z500
    data['u300']=u300
    data['v300']=v300
    data['jet300']=jet300
    data['wMID']=wMID
    data['mslp']=mslp
    data['pwat']=pwat
    
    return data

def get_DT_data(inpath,dt):
    dstr=dt.strftime('%Y%m%d_%H')
    
    infile=inpath+'GFS_' + dstr + '.grib2'
    datafile=xr.load_dataset(infile, engine="cfgrib",  
                             backend_kwargs={'filter_by_keys': {'typeOfLevel': 'potentialVorticity'}})
    uDT=datafile['u']
    vDT=datafile['v']
    tDT=datafile['t']
    pDT=datafile['pres']
    theta=potential_temperature(pDT, tDT)
    
    infile=inpath+'GFS_' + dstr + '.grib2'
    datafile=xr.load_dataset(infile, engine="cfgrib",  
                             backend_kwargs={'filter_by_keys': {'typeOfLevel': 'isobaricInhPa'}})
    
    levs=datafile['isobaricInhPa']
    u=datafile['u'].sel(isobaricInhPa=slice(925,850)).mean(dim='isobaricInhPa')
    v=datafile['v'].sel(isobaricInhPa=slice(925,850)).mean(dim='isobaricInhPa')
    VW = VectorWind(u, v)
    upsi, vpsi = VW.nondivergentcomponent()
    VW = VectorWind(upsi, vpsi)
    vort=VW.vorticity(truncation=196)#truncation=196).data
    
    data={}
    data['vort']=vort
    data['u']=uDT
    data['v']=vDT
    data['theta']=theta
    
    return data

def get_PTe_data(inpath,dt):
    dstr=dt.strftime('%Y%m%d_%H')
    
    infile=inpath+'GFS_' + dstr + '.grib2'
    datafile=xr.load_dataset(infile, engine="cfgrib",  
                             backend_kwargs={'filter_by_keys': {'typeOfLevel': 'isobaricInhPa'}})
    lats=datafile['latitude']
    lons=datafile['longitude']
    t=datafile['t'].sel(isobaricInhPa=850)
    rh=datafile['r'].sel(isobaricInhPa=850)
    hgt=datafile['gh'].sel(isobaricInhPa=850)
    v=datafile['v'].sel(isobaricInhPa=850)
    u=datafile['u'].sel(isobaricInhPa=850)
   
    dwpt=dewpoint_from_relative_humidity(t, rh)
    pte=equivalent_potential_temperature(850*units.hPa,t*units.K,dwpt)
    
    data={}
    data['hgt']=hgt
    data['pte']=pte
    data['u']=u
    data['v']=v
    
    return data
