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
from metpy.calc import wind_speed,potential_temperature,isentropic_interpolation_as_dataset
from metpy.calc import q_vector,lat_lon_grid_deltas,divergence,smooth_n_point
from metpy.calc import vorticity
from metpy.units import units
import pykdtree

def calc_Vchi(u,v):#,lons,lats):
    print('Getting irrotational wind and absolute vorticity')
    VW = VectorWind(u, v)
    uchi, vchi = VW.irrotationalcomponent()
    
    return uchi, vchi

def get_q_pte(datafile):
    lats=datafile['latitude']
    lons=datafile['longitude']
    levs=datafile['isobaricInhPa']
    u850=smooth_n_point(datafile['u'].sel(isobaricInhPa=850), 9, 50)
    v850=smooth_n_point(datafile['v'].sel(isobaricInhPa=850), 9, 50)
    z850=datafile['gh'].sel(isobaricInhPa=850)
    t850=smooth_n_point(datafile['t'].sel(isobaricInhPa=850), 9, 50)
    rh850=datafile['r'].sel(isobaricInhPa=850)
   
    dwpt850=dewpoint_from_relative_humidity(t850, rh850)
    pte850=equivalent_potential_temperature(850*units.hPa,t850,dwpt850)
    
    dx, dy = lat_lon_grid_deltas(lons, lats)
    uqvect, vqvect = q_vector(u850, v850, t850, 850*units.hPa, dx, dy)
    uqvect = uqvect.metpy.dequantify()
    vqvect = vqvect.metpy.dequantify()

    spdqvect = np.sqrt(uqvect**2 + vqvect**2)
    maskval=5e-13
    uqvect=uqvect.where(spdqvect >= maskval)
    vqvect=vqvect.where(spdqvect >= maskval)
    
    def limit_speed(U, V, max_speed):
        speed = np.sqrt(U**2 + V**2)
        mask = speed > max_speed
        scale_factor = np.where(mask, max_speed / speed, 1.0)
        U_limited = U * scale_factor
        V_limited = V * scale_factor
        
        return U_limited, V_limited

    uqvect,vqvect=limit_speed(uqvect,vqvect,3e-12)
    uqvect=uqvect.where(np.abs(uqvect.latitude)<=80)
    vqvect=vqvect.where(np.abs(vqvect.latitude)<=80)

    data={}
    #data['lats']=lats
    #data['lons']=lons
    data['z850']=z850
    data['uqvect850']=uqvect
    data['vqvect850']=vqvect
    data['pte850']=pte850
    
    return data

def get_vort_thick_data(datafile):
    lats=datafile['latitude']
    lons=datafile['longitude']
    levs=datafile['isobaricInhPa']
    t850=datafile['t'].sel(isobaricInhPa=850)
    z500=datafile['gh'].sel(isobaricInhPa=500)
    z1000=datafile['gh'].sel(isobaricInhPa=1000)
    
    vort500=vorticity(datafile['u'].sel(isobaricInhPa=500) * units['m/s'], 
                      datafile['v'].sel(isobaricInhPa=500) * units['m/s'])
    vortLOW=vorticity(datafile['u'].sel(isobaricInhPa=slice(925,850)) * units['m/s'], 
                      datafile['v'].sel(isobaricInhPa=slice(925,850)) * units['m/s']).mean(dim='isobaricInhPa')
    thickness=z500-z1000
    
    data={}
    ##data['lats']=lats
    ##data['lons']=lons
    data['vort500']=vort500.sel(latitude=slice(88,-88)).metpy.dequantify()
    data['vortLOW']=vortLOW.sel(latitude=slice(88,-88)).metpy.dequantify()
    data['z500']=z500
    data['t850']=t850
    data['u850']=datafile['u'].sel(isobaricInhPa=850)
    data['v850']=datafile['v'].sel(isobaricInhPa=850)
    data['thickness']=thickness/10
    
    return data

def get_precip6h(datafile,data_u10,data_v10):
    lats=datafile['latitude']
    lons=datafile['longitude']
    
    u10=data_u10.u10
    v10=data_v10.v10
    precip=datafile.tp*1000#/1000
    mslp=datafile.msl/100
    
    data={}
    #data['lats']=lats
    #data['lons']=lons
    data['u10']=u10
    data['v10']=v10
    data['precip']=precip
    data['mslp']=mslp
    
    return data

def get_IPV_data(datafile):
    mslp=datafile.msl/100

    lats=datafile['latitude']
    lons=datafile['longitude']
    levs=datafile['isobaricInhPa']
    u=datafile['u']
    v=datafile['v']
    hgt=datafile['gh']
    t=datafile['t']
    #tcwv=datafile['tcwv']
    
    pt=potential_temperature(levs, t)
    pv=potential_vorticity_baroclinic(pt, levs, u, v)
    pv=pv.rename('pv')
    ipv=isentropic_interpolation_as_dataset(np.array([320,330,350])*units.K, t,pv,u,v)
    
    #Subset arrays for plotting
    hgt300 = hgt.sel(isobaricInhPa=300)
    pv_iso_upper = pv.sel(isobaricInhPa=slice(300,200)).mean(dim='isobaricInhPa') 
    uchi,vchi=calc_Vchi(datafile['u'].sel(isobaricInhPa=slice(300,200)).mean(dim='isobaricInhPa'),
                        datafile['v'].sel(isobaricInhPa=slice(300,200)).mean(dim='isobaricInhPa'))

    spdchi = np.sqrt(uchi**2 + vchi**2)
    maskval=2
    uchi=uchi.where(spdchi >= maskval)
    vchi=vchi.where(spdchi >= maskval)

    spd320=wind_speed(ipv.u.sel(isentropic_level=320).metpy.dequantify(),
                      ipv.v.sel(isentropic_level=320).metpy.dequantify()).metpy.dequantify()
    spd330=wind_speed(ipv.u.sel(isentropic_level=330).metpy.dequantify(),
                      ipv.v.sel(isentropic_level=330).metpy.dequantify()).metpy.dequantify()
    spd350=wind_speed(ipv.u.sel(isentropic_level=350).metpy.dequantify(),
                      ipv.v.sel(isentropic_level=350).metpy.dequantify()).metpy.dequantify()
    pv_iso_upper=gaussian_filter(pv_iso_upper,sigma=1)

    data={}
    #data['lats']=lats
    #data['lons']=lons
    #data['levs']=levs
    data['pv']=ipv.metpy.dequantify()
    data['theta']=pt.metpy.dequantify()
    
    data['pv320']=ipv.pv.sel(isentropic_level=320).sel(latitude=slice(89,-89)).metpy.dequantify()
    data['u320']=ipv.u.sel(isentropic_level=320).metpy.dequantify().where(spd320>10*0.514444)
    data['v320']=ipv.v.sel(isentropic_level=320).metpy.dequantify().where(spd320>10*0.514444)
    
    data['pv330']=ipv.pv.sel(isentropic_level=330).sel(latitude=slice(89,-89)).metpy.dequantify()
    data['u330']=ipv.u.sel(isentropic_level=330).metpy.dequantify().where(spd330>10*0.514444)
    data['v330']=ipv.v.sel(isentropic_level=330).metpy.dequantify().where(spd330>10*0.514444)
    
    data['pv350']=ipv.pv.sel(isentropic_level=350).sel(latitude=slice(89,-89)).metpy.dequantify()
    data['u350']=ipv.u.sel(isentropic_level=350).metpy.dequantify().where(spd350>10*0.514444)
    data['v350']=ipv.v.sel(isentropic_level=350).metpy.dequantify().where(spd350>10*0.514444)
    
    data['pv_iso_upper']=pv.sel(isobaricInhPa=200).metpy.dequantify()*0+pv_iso_upper
    data['uchi']=uchi
    data['vchi']=vchi
    
    return data

def get_IVT_data(datafile):
    mslp=datafile.msl/100

    lats=datafile['latitude']
    lons=datafile['longitude']
    levs=datafile['isobaricInhPa']
    u=datafile['u'].sel(isobaricInhPa=slice(1000,300))
    v=datafile['v'].sel(isobaricInhPa=slice(1000,300))
    z700=datafile['gh'].sel(isobaricInhPa=700)
    q=datafile['q'].sel(isobaricInhPa=slice(1000,300))
    
    uIVT = -1/9.8*np.trapz(q * u,
                           q.isobaricInhPa*100, axis=0)
    vIVT = -1/9.8*np.trapz(q * v,
                           q.isobaricInhPa*100, axis=0)
    IVT = -1/9.8*np.trapz(q * np.sqrt(u**2 + v**2),
                          q.isobaricInhPa*100, axis=0)
    #uIVT=(1/r)*np.sum(q*u,axis=0)*(100000-30000)
    #vIVT=(1/r)*np.sum(q*v,axis=0)*(100000-30000)
    #IVT=wind_speed(uIVT * units('m/s'), vIVT  * units('m/s')).metpy.dequantify()
    
    data={}
    #data['lats']=lats
    #data['lons']=lons
    data['q']=q
    data['u']=u
    data['v']=v
    data['z700']=z700
    data['uIVT']=(u.isel(isobaricInhPa=0)*0+uIVT).where(IVT>=250)
    data['vIVT']=(v.isel(isobaricInhPa=0)*0+vIVT).where(IVT>=250)
    data['IVT']=q.isel(isobaricInhPa=0)*0+IVT
    
    return data

def get_upper_data(datafile):
    mslp=datafile['msl']/100
    tcwv=datafile['tcwv']
    #pwat=datafile['pwat']
    
    levs=datafile['isobaricInhPa']
    z500=datafile.sel(isobaricInhPa=500).gh##['gh'].data[np.where(levs==500)[0][0]]
    u300=datafile.sel(isobaricInhPa=300).u
    v300=datafile.sel(isobaricInhPa=300).v
    spd300=wind_speed(u300 * units('m/s'), v300  * units('m/s')).metpy.dequantify()
    
    ujet300=u300.where(spd300>50*0.514444)
    vjet300=v300.where(spd300>50*0.514444)
    jet300=spd300.where(spd300>50*0.514444)
    wMID=datafile.w.sel(isobaricInhPa=slice(700,400)).mean(dim='isobaricInhPa')#/400
    wMID=wMID.where(wMID<0)
    #wMID[wMID<0]
    
    lats=datafile['latitude'][:]
    lons=datafile['longitude'][:]
    
    data={}
    #data['lats']=lats
    #data['lons']=lons
    data['z500']=z500
    data['ujet300']=ujet300
    data['vjet300']=vjet300
    data['jet300']=jet300
    data['wMID']=wMID
    data['mslp']=mslp
    data['pwat']=tcwv
        
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
