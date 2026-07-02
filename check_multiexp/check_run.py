from curses import window
from math import exp
from os import path

import xarray as xr
from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd
# import xmca
# from xmca.array import MCA  # numpy
# from xmca.xarray import xMCA  # numpy

import matplotlib.cm as cm
from matplotlib.patches import Patch
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import matplotlib.colors as mcolors 
import glob
# import cmocean as cmo
from scipy import stats
import cartopy.crs as ccrs
import matplotlib.gridspec as gridspec # GRIDSPEC !
import scipy
# import statsmodels.api as sm
# from statsmodels.regression.rolling import RollingOLS
#import xesmf as xe

import yaml
import argparse
from pathlib import Path

###########################################################################################################

datadir = '../data/'
cart_out = './output/'
cart_exp = '/ec/res4/scratch/{}/ece4/'

time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)

######################################################################################

def get_colors(exps):
    colorz = ['orange', 'steelblue', 'indianred', 'forestgreen', 'violet', 'maroon', 'teal', 'black', 'purple', 'olive', 'chocolate', 'dodgerblue', 'rosybrown', 'darkgoldenrod', 'lightseagreen', 'dimgrey', 'midnightblue']

    if len(exps) <= len(colorz):
        return colorz
    else:
        return colorz + get_spectral_colors(len(exps)-len(colorz))

def get_spectral_colors(n):
    """
    Extract n evenly spaced colors from the Spectral colormap.
    
    Parameters:
    n (int): Number of colors to extract
    
    Returns:
    list: List of RGB tuples
    """
    cmap = cm.get_cmap('nipy_spectral')
    if n > 1:
        colors = [cmap(i / (n - 1)) for i in range(n)]
    else:
        colors = [cmap(0)]

    return colors

def add_diahsb_init_to_restart(rest_file, rest_file_new = None, new_vars_2d = ['surf_ini', 'ssh_ini'], new_vars_3d = ['e3t_ini', 'tmask_ini', 'hc_loc_ini', 'sc_loc_ini']):
    """
    Adds missing fields to a restart (produced before 4.1.2) and creates a new restart (compatible with 4.1.2), filling missing variables with zeros.
    New variables can be added.
    """

    rest_oce = xr.load_dataset(rest_file)

    # Get dimension sizes
    nt = len(rest_oce['time_counter'])
    ny = len(rest_oce['y'])
    nx = len(rest_oce['x'])
    nz = len(rest_oce['nav_lev'])

    # Add 0D variables (scalars)
    for var in 'v t s'.split():
        rest_oce[f'frc_{var}'] = xr.DataArray(0.0)

    # Add 2D variables (time_counter, y, x)
    for var in new_vars_2d:
        rest_oce[var] = xr.DataArray(
        np.zeros((nt, ny, nx)),
        dims=['time_counter', 'y', 'x'])

    # Add 3D variables (time_counter, nav_lev, y, x)
    for var in new_vars_3d:
        rest_oce[var] = xr.DataArray(
        np.zeros((nt, nz, ny, nx)),
        dims=['time_counter', 'nav_lev', 'y', 'x'])
    
    if rest_file_new is None:
        rest_file_new = rest_file.replace('.nc', '_mod.nc')

    rest_oce.to_netcdf(rest_file_new)
    
    return

#################################################################################

def get_areas_nemo(exp, user, cart_exp = cart_exp, grid = 'T'):
    #ocean areas
    areas = xr.load_dataset(cart_exp.format(user) + f'/{exp}/areas.nc')
    
    gname = [nam for nam in areas.data_vars if f'-{grid}' in nam]

    if len(gname) > 1:
        raise ValueError(f'Too many grid names matching: {gname}')

    ocean_area = areas[gname[0]].values

    return ocean_area

def get_mask_nemo(exp, user, cart_exp = cart_exp, grid = 'T'):
    #ocean areas
    masks = xr.load_dataset(cart_exp.format(user) + f'/{exp}/masks.nc')

    gname = [nam for nam in masks.data_vars if f'-{grid}' in nam]
    if len(gname) > 1:
        raise ValueError(f'Too many grid names matching: {gname}')
    
    ocean_mask = ~masks[gname[0]].values.astype(bool)

    return ocean_mask

def get_ghflux(exp, user, cart_exp = cart_exp):
    # 0.1 W/m2
    try:
        gout = xr.load_dataset(cart_exp.format(user) + f'/{exp}/Goutorbe_ghflux.nc') # mW/m2
        return float(global_mean(gout.squeeze().mean('lon').drop('time')).gh_flux.values)/1000./0.66 # only over ocean
    except Exception as err:
        print("ERROR in get_ghflux:")
        print(err)
        return 0.1


def global_mean(ds, compute = True):
    """
    Global mean of oifs outputs on reduced gaussian grid. Using zonal means and lat weights, a cleaner implementation should use areas as for ocean.
    """
    try:
        all_lats = ds.lat.groupby('lat').mean()
        weights = np.cos(np.deg2rad(all_lats)).compute()
    except ValueError as coso:
        print(coso)
        print('Dask array, trying to use unique instead')
        all_lats = np.unique(ds.lat.values)
        weights = np.cos(np.deg2rad(all_lats))

    if 'time' in ds.coords:
        if 'cell' in ds.dims:
            ds_mean = ds.groupby('time.year').mean().groupby('lat').mean().weighted(weights).mean('lat')
        else:
            ds_mean = (ds.groupby('time.year').mean().groupby('lat').mean().weighted(weights).mean('lat')).mean('lon')
    else:
        if 'cell' in ds.dims:
            ds_mean = ds.groupby('lat').mean().weighted(weights).mean('lat')
        else:
            ds_mean = (ds.groupby('lat').mean().weighted(weights).mean('lat')).mean('lon')

    
    # ds_mean = ds_mean['rsut rlut rsdt tas'.split()]
    if 'rlut' in ds_mean:
        ds_mean['toa_net'] = ds_mean.rsdt - ds_mean.rlut - ds_mean.rsut
    
    if compute:
        ds_mean = ds_mean.compute()

    return ds_mean

def global_mean_oce_2d(ds, exp, user, cart_exp = cart_exp, compute = True, grid = 'T'):
    """
    Global mean of nemo outputs. Using areas and mask from respective runtime dir.
    """

    area = get_areas_nemo(exp, user, cart_exp = cart_exp, grid = grid)
    mask = get_mask_nemo(exp, user, cart_exp = cart_exp, grid = grid)

    tot_area = np.nansum(area*mask)
    #ds = ds.rename({f'x_grid_{grid}': 'x', f'y_grid_{grid}': 'y'})
    ds_time_mean = (ds*area*mask).sum(['x', 'y'])
    
    for var in ds_time_mean.data_vars:
        if var in ['tos', 'sos', 'qt_oce']:
            ds_time_mean[var] = ds_time_mean[var]/tot_area

    year_sec = 24*60*60*365.25
    gh_flux = get_ghflux(exp, user, cart_exp = cart_exp)

    # heat_trend = ds_time_mean['heatc'].diff('year')/year_sec/tot_area
    # ds_time_mean['enebal'] = heat_trend - ds_time_mean.qt_oce - gh_flux # source of energy in the ocean

    if compute:
        return ds_time_mean.compute()
    else:
        return ds_time_mean

def get_vmask_nemo(exp, user, cart_exp = cart_exp, v_grid = 'deptht', year='1850'):
    #ocean areas

    if(v_grid == 'deptht'):
        fileoce = xr.load_dataset(cart_exp.format(user) + f'/{exp}/output/nemo/' + f'{exp}_oce_1m_T_{year}-{year}.nc')
        thetao = fileoce.thetao[0]
        vmask = thetao/thetao
    else:
        fileoce = xr.load_dataset('../density/density_fields/' + f'{exp}/{exp}_{year}_density.nc')
        #fileoce = fileoce.rename({f'x_grid_T': 'x', f'y_grid_T': 'y'})
        density = fileoce.Nsquared[0]
        vmask = density/density

    return vmask

def global_mean_oce_3d(ds, exp, user, vars, cart_exp = cart_exp, compute = True, depth_mean=False,grid = 'T', singlelevel = False, lev = 0, year='1850'):
    
    area = get_areas_nemo(exp, user, cart_exp = cart_exp, grid = grid)
    
    #ds = ds.rename({f'x_grid_{grid}': 'x', f'y_grid_{grid}': 'y'}) # for 4.1.0

    ds_time_mean = ds[vars].copy()

    if(singlelevel):
        print('Global mean for one vertical level only, lev = ', lev)
        for var in ds.data_vars:
            if var in vars:
                if(var == 'Nsquared'):
                    v_grid = 'depth_mid'
                else:
                    v_grid = 'deptht'
                v_mask = get_vmask_nemo(exp, user, cart_exp = cart_exp, v_grid = v_grid, year = year)
                v_area = (v_mask*area).sum(axis=(1,2))
                ds_time_mean[var] = (ds[var]*area*v_mask).sum(['x', 'y']).compute().isel(depth_mid=lev, drop=True)
                ds_time_mean[var] = ds_time_mean[var]/v_area[lev].values   
    else:
        print('Global mean for all vertical levels')
        for var in ds.data_vars:
            if var in vars:
                if(var == 'Nsquared'):
                    v_grid = 'depth_mid'
                else:
                    v_grid = 'deptht'
                v_mask = get_vmask_nemo(exp, user, cart_exp = cart_exp, v_grid = v_grid, year = year)

                if depth_mean:
                    # get layer thicknesses as weights
                    # I think I miss half of the deepest vertical levels
                    v_levels = ds['density']['deptht']

                    #mid_points = (v_levels[1:].values + v_levels[:-1].values)/2
                    dz_mid = np.diff(v_levels)
                    thick_weights = (dz_mid[:,np.newaxis, np.newaxis] * area * v_mask)   # total weighted thickness per level
                    total_depth = thick_weights.sum() # total ocean volume
                    ds_time_mean[var] = (ds[var] * thick_weights.values).sum() / total_depth.values
                
                else:
                    #test without thickness weights, just mask and area
                    v_area = (v_mask*area).sum(axis=(1,2))                
                    ds_time_mean[var] = (ds[var]*area*v_mask).sum(['x', 'y']).compute()
                    ds_time_mean[var] = (ds_time_mean[var]/v_area.values).compute()
        
    return ds_time_mean
    
def global_mean_oce_3d_region(ds, exp, user, vars, cart_exp = cart_exp, lats=None, lev_bounds=None, compute = True, grid = 'T', singlelevel = False, lev = 0, depth_mean= False, year='1850'):
    
    area = get_areas_nemo(exp, user, cart_exp = cart_exp, grid = grid)
    ds_time_mean = ds[vars].copy()

    if(singlelevel):
        for var in ds.data_vars:
            if var in vars:
                if(var == 'Nsquared'):
                    v_grid = 'depth_mid'
                else:
                    v_grid = 'deptht'
                
                mask_reg = (ds[var].nav_lat>lats[0]) & (ds[var].nav_lat<lats[1])
                area_reg = (area*mask_reg).sel(x=slice(lats[0], lats[1]))
                v_mask = get_vmask_nemo(exp, user, cart_exp = cart_exp, v_grid = v_grid, year = ds.year.values[0])
                v_mask = v_mask.sel(x=slice(lats[0], lats[1]))
                v_area = (v_mask*area_reg).sum(axis=(1,2))
                ds_time_mean[var] = (ds[var].sel(x=slice(lats[0],lats[1]))*area_reg*v_mask).sum(['x', 'y']).compute()
                ds_time_mean[var] = ds_time_mean[var]/v_area[lev].values                
    else:
        for var in ds.data_vars:
            if var in vars:
                if(var == 'Nsquared'):
                    v_grid = 'depth_mid'
                else:
                    v_grid = 'deptht'
                mask_reg = (ds[var].nav_lat>lats[0]) & (ds[var].nav_lat<lats[1])
                area_reg = (area*mask_reg) #.sel(x=slice(lats[0], lats[1]))
                v_mask = get_vmask_nemo(exp, user, cart_exp = cart_exp, v_grid = v_grid, year = year )

                v_mask = v_mask.where(mask_reg) #sel(x=slice(lats[0], lats[1]))

                if depth_mean:
                    v_levels = ds['density']['deptht']          
                    levels = (v_levels[:-1] + v_levels[1:])/ 2 # this line is not working, is summing the same values with each other!!!

                    dz_mid = np.diff(v_levels)   # shape (n_lev-1,), thickness between cell centers                                  # shape (n_lev,) same as v_levels
        
                    levels = (v_levels[1:].values + v_levels[:-1].values)/2

                    l1 = np.where(levels>lev_bounds[0])[0][0]
                    l2 = np.where(levels>lev_bounds[1])[0][0]
                    #print(levels[l1], levels[l2])
                    thick_weights = (area_reg * v_mask *dz_mid).transpose('depth_mid', 'y', 'x')[l1:l2]

                    total_depth = thick_weights.sum()
                    ds_time_mean[var] = (ds[var].where(mask_reg)[l1:l2] * thick_weights.values).sum() / total_depth.values
                
                else:

                    v_area = (v_mask*area_reg).sum(axis=(1,2))
                    ds_time_mean[var] = (ds[var].where(mask_reg) *area_reg*v_mask).sum(['x', 'y']).compute()
                    ds_time_mean[var] = (ds_time_mean[var]/v_area.values).compute()

    return ds_time_mean

def global_mean_ice(ds, exp, user, cart_exp = cart_exp, compute = True, grid = 'T'):
    area = get_areas_nemo(exp, user, cart_exp = cart_exp, grid = grid)
    mask = get_mask_nemo(exp, user, cart_exp = cart_exp, grid = grid)

    ds_norm = (ds*area*mask)
    ds_time_mean = ds_norm.copy()

    for var in ds.data_vars:
        ds_time_mean[var + '_N'] = ds_norm[var].where(ds.nav_lat > 0.).sum(['x', 'y'])
        ds_time_mean[var + '_S'] = ds_norm[var].where(ds.nav_lat < 0.).sum(['x', 'y'])
        
    if 'sithic' in ds.data_vars:
        var = 'sithic'
        ds_time_mean[var + '_N'] = ds_time_mean[var + '_N']*ds_time_mean['siconc_N']
        ds_time_mean[var + '_S'] = ds_time_mean[var + '_S']*ds_time_mean['siconc_S']

    ds_time_mean = ds_time_mean[[var for var in ds_time_mean.data_vars if '_N' in var or '_S' in var]]

    if compute:
        return ds_time_mean.compute()
    else:
        return ds_time_mean
    
def compute_atm_clim(ds, exp, cart_out = cart_out, atmvars = 'rsut rlut rsdt tas pr'.split(), year_clim = None):
    ds = ds.rename({'time_counter': 'time'})
    ds = ds[atmvars].groupby('time.year').mean().compute()

    if year_clim is None:
        print('Using last 20 years for climatology')
        atmclim = ds.isel(year = slice(-20, None)).mean('year')
        yclim_tag = ''
    else:
        print(f'Using years {year_clim[0]}-{year_clim[1]} for climatology')
        atmclim = ds.sel(year = slice(year_clim[0], year_clim[1])).mean('year')
        yclim_tag = f'_{year_clim[0]}-{year_clim[1]}'
    atmmean = global_mean(ds, compute = True)

    if cart_out is not None:
        atmclim.to_netcdf(cart_out + f'clim_tuning_{exp}{yclim_tag}.nc')
        atmmean.to_netcdf(cart_out + f'mean_tuning_{exp}.nc')

    return atmclim, atmmean

def compute_atm_map(ds, exp, cart_out = cart_out, atmvars = 'rsut rlut rsdt tas pr rsnt rsntcs rlnt rlntcs'.split(), year_clim = None):
    print('Computing atmospheric map climatology')
    ds = ds.rename({'time_counter': 'time'})
    ds = ds[atmvars].groupby('time.year').mean().compute()

    if cart_out is not None:
        ds.to_netcdf(cart_out + f'map_tuning_{exp}.nc')

    return ds

def compute_atm3d_map(ds, exp, cart_out = cart_out, atmvars = 'wap'.split(), year_clim = None):
    print('Computing 3D atmospheric map climatology')
    ds = ds.rename({'time_counter': 'time'})
    ds = ds[atmvars].groupby('time.year').mean().compute()

    if cart_out is not None:
        ds.to_netcdf(cart_out + f'map_atm3d_tuning_{exp}.nc')

    return ds

def compute_oce_map(ds, exp, user, cart_exp = cart_exp, cart_out = cart_out, ocevars = 'tos heatc qt_oce sos'.split(), year_clim = None, grid = 'T'):
    print('Computing ocean map climatology')
    ds = ds.rename({'time_counter': 'time'})
    # print(ds.data_vars)
    ds = ds[ocevars].groupby('time.year').mean()
    
    if f'x_grid_{grid}_inner' in ds.dims:
        ds = ds.rename({f'x_grid_{grid}_inner': 'x', f'y_grid_{grid}_inner': 'y'})
    if f'x_grid_{grid}' in ds.dims:
        ds = ds.rename({f'x_grid_{grid}': 'x', f'y_grid_{grid}': 'y'})

    if cart_out is not None:
        ds.to_netcdf(cart_out + f'map_oce_tuning_{exp}.nc')

    return ds

def compute_ice_map(ds, exp, user, cart_exp = cart_exp, cart_out = cart_out, icevars = 'siconc'.split(), year_clim = None, grid = 'T'):
    
    print('Computing ice map climatology')
    ds = ds.rename({'time_counter': 'time'})
    # print(ds.data_vars)
    ds = ds[icevars].groupby('time.year').mean()
    
    if f'x_grid_{grid}_inner' in ds.dims:
        ds = ds.rename({f'x_grid_{grid}_inner': 'x', f'y_grid_{grid}_inner': 'y'})
    if f'x_grid_{grid}' in ds.dims:
        ds = ds.rename({f'x_grid_{grid}': 'x', f'y_grid_{grid}': 'y'})

    if cart_out is not None:
        ds.to_netcdf(cart_out + f'map_ice_tuning_{exp}.nc')

def compute_oce_clim(ds, exp, user, cart_exp = cart_exp, cart_out = cart_out, ocevars = 'tos heatc qt_oce sos'.split(), year_clim = None, grid = 'T'):
    ds = ds.rename({'time_counter': 'time'})
    #print(ds.data_vars)
    ds = ds[ocevars].groupby('time.year').mean()
    if f'x_grid_{grid}_inner' in ds.dims:
        ds = ds.rename({f'x_grid_{grid}_inner': 'x', f'y_grid_{grid}_inner': 'y'})
    if f'x_grid_{grid}' in ds.dims:
        ds = ds.rename({f'x_grid_{grid}': 'x', f'y_grid_{grid}': 'y'})

    if year_clim is None:
        print('Using last 20 years for climatology')
        oceclim = ds.isel(year = slice(-20, None)).mean('year').compute()
        yclim_tag = ''
    else:
        print(f'Using years {year_clim[0]}-{year_clim[1]} for climatology')
        oceclim = ds.sel(year = slice(year_clim[0], year_clim[1])).mean('year').compute()
        yclim_tag = f'_{year_clim[0]}-{year_clim[1]}'

    ocemean = global_mean_oce_2d(ds, exp, user, cart_exp, compute = True)
    
    if cart_out is not None:
        oceclim.to_netcdf(cart_out + f'clim_oce_tuning_{exp}{yclim_tag}.nc')
        ocemean.to_netcdf(cart_out + f'mean_oce_tuning_{exp}.nc')

    return oceclim, ocemean

def compute_ice_clim(ds, exp, user, cart_exp = cart_exp, cart_out = cart_out, icevars = 'sithic sivolu siconc'.split(), year_clim = None):
    ds = ds.rename({'time_counter': 'time'})
    ds = ds[icevars].groupby('time.year').mean()

    if year_clim is None:
        print('Using last 20 years for climatology')
        iceclim = ds.isel(year = slice(-20, None)).mean('year').compute()
        yclim_tag = ''
    else:
        print(f'Using years {year_clim[0]}-{year_clim[1]} for climatology')
        iceclim = ds.sel(year = slice(year_clim[0], year_clim[1])).mean('year').compute()
        yclim_tag = f'_{year_clim[0]}-{year_clim[1]}'

    icemean = global_mean_ice(ds, exp, user, cart_exp, compute = True)

    if cart_out is not None:
        iceclim.to_netcdf(cart_out + f'clim_ice_tuning_{exp}{yclim_tag}.nc')
        icemean.to_netcdf(cart_out + f'mean_ice_tuning_{exp}.nc')

    return iceclim, icemean

def compute_amoc_clim(ds, exp, cart_out = cart_out, year_clim = None):
    amoc_ts = calc_amoc_ts(ds, plot = False)
    amoc_ts = amoc_ts.groupby('time_counter.year').mean() # year as time coordinate

    if isinstance(amoc_ts, xr.Dataset):
        amoc_ts = amoc_ts['msftyz']

    ds = ds.rename({'time_counter': 'time'})
    amoc = ds['msftyz'].groupby('time.year').mean()
    amoc = amoc.compute()

    if year_clim is None:
        print('Using last 20 years for climatology')
        amoc_mean = amoc.isel(year = slice(-20, None)).mean('year')
        yclim_tag = ''
    else:
        print(f'Using years {year_clim[0]}-{year_clim[1]} for climatology')
        amoc_mean = ds.sel(year = slice(year_clim[0], year_clim[1])).mean('year')
        yclim_tag = f'_{year_clim[0]}-{year_clim[1]}'
    amoc_mean = amoc_mean.squeeze()
    
    amoc_mean = amoc_mean.compute()
    amoc_ts = amoc_ts.compute()

    if cart_out is not None:
        amoc_mean.to_netcdf(cart_out + f'amoc_2d_tuning_{exp}{yclim_tag}.nc')
        amoc_ts.to_netcdf(cart_out + f'amoc_ts_tuning_{exp}.nc')

    return amoc_mean, amoc_ts

def compute_rho_clim(ds, exp, user, cart_exp = cart_exp, cart_out = cart_out, ocevars = 'density Nsquared'.split(), year_clim = None, grid = 'T'):
    #ds = ds.rename({'time_counter': 'time'})
    # print(ds.data_vars)
    #ds = ds[ocevars].groupby('time.year').mean() # controllare se estendibile a medie mensili!! 
    #ds = ds.rename({f'x_grid_{grid}_inner': 'x', f'y_grid_{grid}_inner': 'y'})
    #ds = ds.rename({f'x_grid_{grid}': 'x', f'y_grid_{grid}': 'y'})

    if year_clim is None:
        print('Using last 20 years for climatology')
        oceclim = ds.isel(year = slice(-20, None)).mean('year').compute()
    else:
        oceclim = ds.sel(year = slice(year_clim[0], year_clim[1])).mean('year').compute()

    oceclim.to_netcdf(cart_out + f'clim_rho_tuning_{exp}.nc')

    ocemean = global_mean_oce_3d(ds, exp, user, 'density Nsquared'.split(), cart_exp, compute = True)
    ocemean.to_netcdf(cart_out + f'mean_rho_tuning_{exp}.nc')

    return oceclim, ocemean 

def calc_amoc_ts(data, ax = None, exp_name = 'exp', depth_min = 500., depth_max = 2000., lat_min = 25, lat_max = 28, ylim = (5, 20), plot = False, basin = 2):

    if plot and ax is None:
        fig, ax = plt.subplots()

    amoc = data.sel(
        depthw=slice(depth_min, depth_max), 
        basin=2
    )['msftyz']
    
    # Apply latitude constraint and compute
    amoc = amoc.where(
        (data['nav_lat'] > lat_min) & (data['nav_lat'] < lat_max)
    ).compute()
    
    # Resample to yearly means and find maximum
    amoc_yearly = amoc.resample(time_counter='YS').mean()
    #amoc_yearly = amoc.groupby('time.year').mean()
    amoc_max = amoc_yearly.max(dim=['depthw', 'y'])
    
    # Plot timeseries
    if plot:
        amoc_max.plot(ylim=ylim, label = exp_name, ax = ax)

    return amoc_max

##################################### READ OUTPUTS ################################

def file_list(exp, user, cart_exp = '/ec/res4/scratch/{}/ece4/', remove_last_year = False, coupled = True, density= False, atm_remapped = False):
    if atm_remapped:
        cart = f'{cart_exp.format(user)}/{exp}/output/oifs_remap/'
    else:
        cart = f'{cart_exp.format(user)}/{exp}/output/oifs/'
    filz_atm = cart + f'{exp}_atm_cmip6_1m_*.nc'
    filz_atm3d = cart + f'{exp}_atm_cmip6_pl_*.nc'

    cart = f'{cart_exp.format(user)}/{exp}/output/nemo/'
    filz_amoc = cart + f'{exp}_oce_1m_diaptr3d_*.nc'
    filz_nemo = cart + f'{exp}_oce_1m_T_*.nc'
    filz_ice = cart + f'{exp}_ice_1m_*.nc'

    # ftv3_oce_1m_diaptr2d_1991-1991.nc -> hf_basin

    if remove_last_year:
        # Still running, remove last year
        fils = glob.glob(filz_atm)
        fils.sort()
        filz_atm = fils[:-1]

        fils = glob.glob(filz_atm3d)
        fils.sort()
        filz_atm3d = fils[:-1]

        if coupled:
            fils = glob.glob(filz_nemo)
            fils.sort()
            filz_nemo = fils[:-1]

            fils = glob.glob(filz_ice)
            fils.sort()
            filz_ice = fils[:-1]

            fils = glob.glob(filz_amoc)
            fils.sort()
            filz_amoc = fils[:-1]
        else:
            filz_amoc = []
            filz_nemo = []
            filz_ice = []
        
    if density:
        filz_rho = '../density/density_fields/' + f'{exp}/{exp}_*_density.nc'
        return filz_atm, filz_atm3d, filz_nemo, filz_amoc, filz_ice, filz_rho

    else:
        return filz_atm, filz_atm3d, filz_nemo, filz_amoc, filz_ice

def read_output_map(exps, user = None, read_again = [], cart_exp = cart_exp, cart_out = cart_out, atmvars = 'rsut rlut rsdt tas rlnt rlntcs rsnt rsntcs'.split(), atmvars3d='ta ua va wap cl'.split(), ocevars = 'tos qt_oce'.split(), icevars='siconc'.split(), atm_only = False, year_clim = None, density=False, density_only=False, flag_omega = False):
    """
    Reads outputs and computes global means.

    exps: list of experiment names to read
    user: list of users
    read_again: list of exps to read again (if run has proceeded)
    atm_only: compute only atm diags
    year_clim: set years for computing climatologies (if None, considers last 20 years)
    """

    if isinstance(user, str):
        user = len(exps)*[user]
    else:
        if len(user) != len(exps):
            raise ValueError(f"Length not corresponding: exps {len(exps)}, user {len(user)}")

    filz_atm = dict()
    filz_atm3d = dict()
    filz_nemo = dict()
    filz_ice = dict()

    for exp, us in zip(exps, user):
        filz_atm[exp], filz_atm3d[exp], filz_nemo[exp], _, filz_ice[exp] = file_list(exp, us, cart_exp = cart_exp)
        
    atmmap_exp = dict()
    atmmap3d_exp = dict()
    ocemap_exp = dict()
    icemap_exp = dict()

    if(atm_only):
        coupled = False
    else:
        coupled = True

    for exp, us in zip(exps, user):
        print(exp)
        if os.path.exists(cart_out + f'map_tuning_{exp}.nc') and exp not in read_again:
            print('Already computed, reading clim..')
            existing_atm = xr.load_dataset(cart_out + f'map_tuning_{exp}.nc')
            atmmap_exp[exp] = existing_atm            

            existing_oce = xr.load_dataset(cart_out + f'map_oce_tuning_{exp}.nc')
            existing_vars = list(existing_oce.data_vars)
            missing_vars = [v for v in ocevars if v not in existing_vars]
            
            if len(missing_vars) == 0:
                ocemap_exp[exp] = existing_oce           
            else:
                print(f'Computing missing OCE vars: {missing_vars}')
                
                ds = xr.open_mfdataset(filz_nemo[exp],chunks={'time_counter': 240})

                new_atm = compute_oce_map(ds,exp,user=user, cart_out=cart_out,ocevars=missing_vars,year_clim=year_clim)

                # merge old + new
                updated_atm = xr.merge([existing_oce, new_atm])
                updated_atm.to_netcdf(cart_out + f'map_oce_tuning_{exp}.nc', mode='w')

                ocemap_exp[exp] = updated_atm

            if (flag_omega):
                atmmap3d_exp[exp] = xr.load_dataset(cart_out + f'map_atm3d_tuning_{exp}_500hPa.nc')
            else:
                atmmap3d_exp[exp] = xr.load_dataset(cart_out + f'map_atm3d_tuning_{exp}.nc')

            if coupled:
                if os.path.exists(cart_out + f'map_oce_tuning_{exp}.nc'):
                    ocemap_exp[exp] = xr.load_dataset(cart_out + f'map_oce_tuning_{exp}.nc')
                    icemap_exp[exp] = xr.load_dataset(cart_out+f'map_ice_tuning_{exp}.nc')
                
        else:
            print('Computing clim...')

            ds = xr.open_mfdataset(filz_atm[exp], decode_times=time_coder, chunks = {'time_counter': 240})
            # ATM CLIM
            atmmap_exp[exp] = compute_atm_map(ds, exp, cart_out = cart_out, atmvars = atmvars, year_clim = year_clim)

            ds.close()
            ds = xr.open_mfdataset(filz_atm3d[exp], decode_times=time_coder, chunks = {'time_counter': 240})
            atmmap3d_exp[exp] = compute_atm3d_map(ds, exp, cart_out = cart_out, atmvars = atmvars3d, year_clim = year_clim) 
            ds.close()
            

            if coupled:
                # OCE CLIM
                ds = xr.open_mfdataset(filz_nemo[exp], decode_times=time_coder, chunks = {'time_counter': 240})
                ocemap_exp[exp] = compute_oce_map(ds, exp, us, cart_exp = cart_exp, cart_out = cart_out, ocevars = ocevars, year_clim = year_clim)
                ds.close()

                ds = xr.open_mfdataset(filz_ice[exp], decode_times=time_coder, chunks = {'time_counter': 240})
                icemap_exp[exp] = compute_ice_map(ds, exp, us, cart_exp = cart_exp, cart_out = cart_out, icevars = icevars, year_clim = year_clim)
                ds.close()

                
    clim_all = dict()
    clim_all['atm_map'] = atmmap_exp
    clim_all['atm3d_map'] = atmmap3d_exp
    if coupled:
        clim_all['oce_map'] = ocemap_exp
        clim_all['ice_map'] = icemap_exp
    
    return clim_all

def read_output(exps, user=None, read_again=[], cart_exp=cart_exp, cart_out=cart_out,
                atmvars='rsut rlut rsdt tas pr'.split(),
                ocevars='tos heatc qt_oce sos'.split(),
                icevars='siconc sivolu sithic'.split(),
                atm_only=False, year_clim=None, file_lists=None, ongoing = [], density=False):
    """
    Reads outputs and computes global means.

    exps: list of experiment names to read
    user: list of users
    read_again: list of exps to read again (if run has proceeded)
    atm_only: compute only atm diags
    year_clim: set years for computing climatologies (if None, considers last 20 years)
    """

    time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)

    if isinstance(user, str):
        user = len(exps) * [user]
    else:
        if len(user) != len(exps):
            raise ValueError(f"Length not corresponding: exps {len(exps)}, user {len(user)}")

    if file_lists is not None:
        if density:
            filz_atm, filz_atm3d, filz_nemo, filz_amoc, filz_ice, filz_rho = file_lists
        else:
            filz_atm, filz_atm3d, filz_nemo, filz_amoc, filz_ice = file_lists
    else:
        filz_atm  = dict()
        filz_atm3d = dict()
        filz_amoc = dict()
        filz_rho = dict()
        filz_nemo = dict()
        filz_ice  = dict()
        for exp, us in zip(exps, user):
            remove_last_year = False
            if exp in ongoing:
                remove_last_year = True

            if density:
                filz_atm[exp], filz_atm3d[exp], filz_nemo[exp], filz_amoc[exp], filz_ice[exp], filz_rho[exp] = file_list(exp, us, cart_exp = cart_exp, density=True)
            else:
                filz_atm[exp], filz_atm3d[exp], filz_nemo[exp], filz_amoc[exp], filz_ice[exp] = file_list(exp, us, cart_exp = cart_exp, remove_last_year = remove_last_year)
                
    # ── helpers ───────────────────────────────────────────────────────────────

    def _files_exist(pattern_or_list):
        if isinstance(pattern_or_list, str):
            return len(glob.glob(pattern_or_list)) > 0
        return len(pattern_or_list) > 0

    def _needs_compute(clim_path, mean_path, required_vars=None):
        """
        Returns (action, missing_vars) where action is one of:
          False          – files exist, vars complete, exp not in read_again → just read
          'update'       – files exist, vars complete, exp in read_again     → append new timesteps
          'missing_vars' – files exist, vars incomplete (read_again only)    → compute & merge missing vars only
          'from_scratch' – files missing                                     → compute from scratch
        """
        if not (os.path.exists(clim_path) and os.path.exists(mean_path)):
            return 'from_scratch', required_vars
        if exp in read_again:
            if required_vars is not None:
                try:
                    ds_saved = xr.open_dataset(clim_path)
                    missing = [v for v in required_vars if v not in ds_saved]
                except Exception:
                    missing = required_vars
                if missing:
                    print(f'  → missing vars in {clim_path}: {missing}')
                    return 'missing_vars', missing
            return 'update', None
        return False, None

    def _open_mfdataset_safe(filz, exp, us, coupled, suffix='atm'):
        try:
            return xr.open_mfdataset(filz, decode_times=time_coder, chunks={'time_counter': 240})
        except OSError as err:
            print(err)
            if err.errno == -101:
                print('Run still ongoing, removing last year')
                new_filz = file_list(exp, us, cart_exp=cart_exp,
                                    remove_last_year=True, coupled=coupled)
                filz_atm_new, filz_atm3d_new, filz_nemo_new, filz_amoc_new, filz_ice_new = new_filz
                filz_new = {'atm': filz_atm_new, 'atm3d': filz_atm3d_new, 'oce': filz_nemo_new,
                    'amoc': filz_amoc_new, 'ice': filz_ice_new}[suffix]
                return xr.open_mfdataset(filz_new, decode_times=time_coder,
                                        chunks={'time_counter': 240})
            else:
                raise err

    def _normalise_amoc_ts(ds):
        if 'time_counter' in ds.dims:
            ds = ds.groupby('time_counter.year').mean()
        if isinstance(ds, xr.Dataset):
            ds = ds['msftyz']
        return ds

    # ── per-domain compute (full dataset) ─────────────────────────────────────

    def _compute_atm(exp, us, coupled, vars=atmvars, save=True):
        ds = _open_mfdataset_safe(filz_atm[exp], exp, us, coupled, suffix='atm')
        return compute_atm_clim(ds, exp,
                                cart_out=cart_out if save else None,
                                atmvars=vars, year_clim=year_clim)

    def _compute_oce(exp, us, vars=ocevars, save=True):
        #print(filz_nemo[exp])
        ds = _open_mfdataset_safe(filz_nemo[exp], exp, us, coupled=True, suffix='oce')
        # ds = xr.open_mfdataset(filz_nemo[exp], decode_times=time_coder,
        #                        chunks={'time_counter': 240})
        return compute_oce_clim(ds, exp, us, cart_exp=cart_exp,
                                cart_out=cart_out if save else None,
                                ocevars=vars, year_clim=year_clim)

    def _compute_ice(exp, us, vars=icevars, save=True):
        ds = _open_mfdataset_safe(filz_ice[exp], exp, us, coupled=True, suffix='ice')
        # ds = xr.open_mfdataset(filz_ice[exp], decode_times=time_coder,
        #                        chunks={'time_counter': 240})
        return compute_ice_clim(ds, exp, us, cart_exp=cart_exp,
                                cart_out=cart_out if save else None,
                                icevars=vars, year_clim=year_clim)

    def _compute_amoc(exp, save=True):
        ds = _open_mfdataset_safe(filz_amoc[exp], exp, us, coupled=True, suffix='amoc')
            # ds = xr.open_mfdataset(filz_amoc[exp], decode_times=time_coder,
            #                        chunks={'time_counter': 240})
        return compute_amoc_clim(ds, exp,
                                cart_out=cart_out if save else None,
                                year_clim=year_clim)

    # ── per-domain update (append new timesteps only) ─────────────────────────

    def _update_domain(exp, us, domain, clim_old, mean_old, filz,
                       compute_fn, clim_path, mean_path):
        last_year = int(mean_old.year[-1].values)
        print(f'[{domain}] Last year in saved data: {last_year}')

        # try:
        #     ds = xr.open_mfdataset(filz, decode_times=time_coder, chunks={'time_counter': 240})
        # except OSError as err:
        ds = _open_mfdataset_safe(filz, exp, us, True if domain in ['oce','ice'] else False, suffix=domain) # THIS changes the file list!

        ds_new = ds.sel(time_counter=slice(f'{last_year + 1}0101', None))
        if len(ds_new.time_counter) == 0:
            print(f'[{domain}] No new data, using existing diagnostics')
            return clim_old, mean_old
        print(f'[{domain}] Found {len(ds_new.time_counter)} new time steps')
        clim_new, mean_new = compute_fn(ds_new)
        mean_updated = xr.concat([mean_old, mean_new], dim='year')
        clim_new.to_netcdf(clim_path)
        mean_updated.to_netcdf(mean_path)
        return clim_new, mean_updated

    def _update_amoc(exp, amoc_ts_old, amoc_mean_old):
        last_year = int(amoc_ts_old.year[-1].values)
        # ds = xr.open_mfdataset(filz_amoc[exp], decode_times=time_coder,
        #                        chunks={'time_counter': 240})
        ds = _open_mfdataset_safe(filz_amoc[exp], exp, us, True, suffix='amoc')

        
        ds_new = ds.sel(time_counter=slice(f'{last_year + 1}0101', None))
        if len(ds_new.time_counter) == 0:
            print('[amoc] No new data, using existing diagnostics')
            return amoc_mean_old, amoc_ts_old
        print(f'[amoc] Found {len(ds_new.time_counter)} new time steps')
        amoc_mean_new, amoc_ts_new = compute_amoc_clim(ds_new, exp,
                                                        cart_out=None,
                                                        year_clim=year_clim)
        amoc_ts_updated = xr.concat([amoc_ts_old, amoc_ts_new], dim='year')
        amoc_mean_new.to_netcdf(cart_out + f'amoc_2d_tuning_{exp}.nc')
        amoc_ts_updated.to_netcdf(cart_out + f'amoc_ts_tuning_{exp}.nc')
        return amoc_mean_new, amoc_ts_updated

    # ── merge missing vars into existing saved files ───────────────────────────

    def _merge_missing_vars(domain, clim_path, mean_path, missing, compute_fn_missing):
        """
        Compute only the missing variables on the full dataset and merge
        into the existing clim/mean files.
        compute_fn_missing: callable(missing_vars) -> (clim_new, mean_new)
        """
        print(f'[{domain}] Computing missing vars on full dataset: {missing}')
        clim_old = xr.load_dataset(clim_path)
        mean_old = xr.load_dataset(mean_path)

        clim_new, mean_new = compute_fn_missing(missing)

        clim_merged = xr.merge([clim_old, clim_new[missing]])
        mean_merged = xr.merge([mean_old, mean_new[missing]])

        clim_merged.to_netcdf(clim_path)
        mean_merged.to_netcdf(mean_path)
        return clim_merged, mean_merged

    # ── output containers ─────────────────────────────────────────────────────

    atmmean_exp   = dict()
    atmclim_exp   = dict()
    oceclim_exp   = dict()
    ocemean_exp   = dict()
    amoc_mean_exp = dict()
    amoc_ts_exp   = dict()
    iceclim_exp   = dict()
    icemean_exp   = dict()
    if density:
        rhoclim_exp = dict()
        rhomean_exp = dict()

    coupled_exps = []

    # ═════════════════════════════════════════════════════════════════════════
    for exp, us in zip(exps, user):
        print(f'\n{"="*60}\n{exp}')

        # ── detect coupled / ocean-only ───────────────────────────────────────
        coupled = False
        if not atm_only:
            coupled = (os.path.exists(cart_out + f'clim_oce_tuning_{exp}.nc') or
                       os.path.exists(cart_out + f'oce_tuning_{exp}.nc') or
                       _files_exist(filz_nemo[exp]))
            print('coupled' if coupled else f'No ocean files found for {exp}. Assuming atm-only')

        ocean_only = (not _files_exist(filz_atm[exp]) and
                      not os.path.exists(cart_out + f'clim_tuning_{exp}.nc'))
        if ocean_only:
            print(f'No atm files found for {exp}. Assuming ocean-only')

        coupled_exps.append(True if coupled else False)

        # ── file paths ────────────────────────────────────────────────────────

        if year_clim is None:
            yclim_tag = ''
        else:
            yclim_tag = f'_{year_clim[0]}-{year_clim[1]}'

        atm_clim_path = cart_out + f'clim_tuning_{exp}{yclim_tag}.nc'
        atm_mean_path = cart_out + f'mean_tuning_{exp}.nc'
        oce_clim_path = cart_out + f'clim_oce_tuning_{exp}{yclim_tag}.nc'
        oce_mean_path = cart_out + f'mean_oce_tuning_{exp}.nc'
        ice_clim_path = cart_out + f'clim_ice_tuning_{exp}{yclim_tag}.nc'
        ice_mean_path = cart_out + f'mean_ice_tuning_{exp}.nc'
        amoc_2d_path  = cart_out + f'amoc_2d_tuning_{exp}{yclim_tag}.nc'
        amoc_ts_path  = cart_out + f'amoc_ts_tuning_{exp}.nc'
        rho_clim_path  = cart_out + f'clim_rho_tuning_{exp}{yclim_tag}.nc'
        rho_mean_path  = cart_out + f'mean_rho_tuning_{exp}.nc'

        # ── per-domain actions ────────────────────────────────────────────────
        atm_action,  atm_missing  = _needs_compute(atm_clim_path, atm_mean_path, atmvars) if not ocean_only else (False, None)
        oce_action,  oce_missing  = _needs_compute(oce_clim_path, oce_mean_path, ocevars) if coupled        else (False, None)
        ice_action,  ice_missing  = _needs_compute(ice_clim_path, ice_mean_path, icevars) if coupled        else (False, None)
        amoc_action, _            = _needs_compute(amoc_2d_path,  amoc_ts_path)           if coupled        else (False, None)
        if density:
            rho_action, _            = _needs_compute(rho_clim_path, rho_mean_path)       if coupled        else (False, None)
            
        # ── ATM ───────────────────────────────────────────────────────────────
        if not ocean_only:
            if not atm_action:
                print('[atm] Already computed, reading...')
                atmclim_exp[exp] = xr.load_dataset(atm_clim_path)
                atmmean_exp[exp] = xr.load_dataset(atm_mean_path)

            elif atm_action == 'missing_vars':
                atmclim_exp[exp], atmmean_exp[exp] = _merge_missing_vars(
                    'atm', atm_clim_path, atm_mean_path,
                    missing=atm_missing,
                    compute_fn_missing=lambda vars: compute_atm_clim(
                        _open_mfdataset_safe(filz_atm[exp], exp, us, coupled, suffix='atm'),
                        exp, cart_out=None, atmvars=vars, year_clim=year_clim))

            elif atm_action == 'update':
                print('[atm] Updating with new data...')
                clim_old = xr.load_dataset(atm_clim_path)
                mean_old = xr.load_dataset(atm_mean_path)
                atmclim_exp[exp], atmmean_exp[exp] = _update_domain(
                    exp, us, 'atm', clim_old, mean_old,
                    filz=filz_atm[exp],
                    compute_fn=lambda ds: compute_atm_clim(
                        ds, exp, cart_out=None, atmvars=atmvars, year_clim=year_clim),
                    clim_path=atm_clim_path, mean_path=atm_mean_path)

            else:  # 'from_scratch'
                print('[atm] Computing from scratch...')
                atmclim_exp[exp], atmmean_exp[exp] = _compute_atm(exp, us, coupled)

        # ── OCE ───────────────────────────────────────────────────────────────
        if coupled:
            if not oce_action:
                print('[oce] Already computed, reading...')
                if os.path.exists(oce_clim_path):
                    oceclim_exp[exp] = xr.load_dataset(oce_clim_path)
                    ocemean_exp[exp] = xr.load_dataset(oce_mean_path)
                else:  # legacy filename
                    oceclim_exp[exp] = xr.load_dataset(cart_out + f'oce_tuning_{exp}.nc')
                    ocemean_exp[exp] = None

            elif oce_action == 'missing_vars':
                oceclim_exp[exp], ocemean_exp[exp] = _merge_missing_vars(
                    'oce', oce_clim_path, oce_mean_path,
                    missing=oce_missing,
                    compute_fn_missing=lambda vars: compute_oce_clim(
                        xr.open_mfdataset(filz_nemo[exp], decode_times=time_coder, chunks={'time_counter': 240}),
                        exp, us, cart_exp=cart_exp, cart_out=None,
                        ocevars=vars, year_clim=year_clim))

            elif oce_action == 'update':
                print('[oce] Updating with new data...')
                clim_old = xr.load_dataset(oce_clim_path)
                mean_old = xr.load_dataset(oce_mean_path)
                oceclim_exp[exp], ocemean_exp[exp] = _update_domain(
                    exp, us, 'oce', clim_old, mean_old,
                    filz=filz_nemo[exp],
                    compute_fn=lambda ds: compute_oce_clim(
                        ds, exp, us, cart_exp=cart_exp, cart_out=None,
                        ocevars=ocevars, year_clim=year_clim),
                    clim_path=oce_clim_path, mean_path=oce_mean_path)

            else:  # 'from_scratch'
                print('[oce] Computing from scratch...')
                oceclim_exp[exp], ocemean_exp[exp] = _compute_oce(exp, us)

            # ── ICE ───────────────────────────────────────────────────────────
            if not ice_action:
                print('[ice] Already computed, reading...')
                iceclim_exp[exp] = xr.load_dataset(ice_clim_path)
                icemean_exp[exp] = xr.load_dataset(ice_mean_path)

            elif ice_action == 'missing_vars':
                iceclim_exp[exp], icemean_exp[exp] = _merge_missing_vars(
                    'ice', ice_clim_path, ice_mean_path,
                    missing=ice_missing,
                    compute_fn_missing=lambda vars: compute_ice_clim(
                        xr.open_mfdataset(filz_ice[exp], decode_times=time_coder, chunks={'time_counter': 240}),
                        exp, us, cart_exp=cart_exp, cart_out=None,
                        icevars=vars, year_clim=year_clim))

            elif ice_action == 'update':
                print('[ice] Updating with new data...')
                clim_old = xr.load_dataset(ice_clim_path)
                mean_old = xr.load_dataset(ice_mean_path)
                iceclim_exp[exp], icemean_exp[exp] = _update_domain(
                    exp, us, 'ice', clim_old, mean_old,
                    filz=filz_ice[exp],
                    compute_fn=lambda ds: compute_ice_clim(
                        ds, exp, us, cart_exp=cart_exp, cart_out=None,
                        icevars=icevars, year_clim=year_clim),
                    clim_path=ice_clim_path, mean_path=ice_mean_path)

            else:  # 'from_scratch'
                print('[ice] Computing from scratch...')
                iceclim_exp[exp], icemean_exp[exp] = _compute_ice(exp, us)
            
            ## density
            if density:
                if not rho_action:
                    print('[rho] Already computed, reading...')
                    rhoclim_exp[exp] = xr.load_dataset(rho_clim_path)
                    rhomean_exp[exp] = xr.load_dataset(rho_mean_path)
                elif rho_action == 'from_scratch':
                    ds = xr.open_mfdataset(filz_rho[exp], decode_times=time_coder, chunks = {'time': 20})
                    rhomean_exp[exp], rhoclim_exp[exp] = compute_rho_clim(ds, exp, us, cart_exp = cart_exp, cart_out = cart_out, year_clim = year_clim)
                    ds.close()
                else:
                    raise ValueError('rho update not implemented')

            # ── AMOC ──────────────────────────────────────────────────────────
            if not amoc_action:
                if os.path.exists(amoc_ts_path):
                    print('[amoc] Already computed, reading...')
                    amoc_ts_exp[exp]   = _normalise_amoc_ts(xr.load_dataset(amoc_ts_path))
                    amoc_mean_exp[exp] = xr.load_dataset(amoc_2d_path)

            elif amoc_action == 'update':
                print('[amoc] Updating with new data...')
                amoc_ts_old   = _normalise_amoc_ts(xr.load_dataset(amoc_ts_path))
                amoc_mean_old = xr.load_dataset(amoc_2d_path)
                amoc_mean_exp[exp], amoc_ts_exp[exp] = _update_amoc(
                    exp, amoc_ts_old, amoc_mean_old)

            else:  # 'from_scratch'
                print('[amoc] Computing from scratch...')
                amoc_mean_exp[exp], amoc_ts_exp[exp] = _compute_amoc(exp, save=True)

    # ── assemble output ───────────────────────────────────────────────────────
    clim_all = dict()
    clim_all['atm_clim']  = atmclim_exp
    clim_all['atm_mean']  = atmmean_exp
    if np.any(coupled_exps):
        clim_all['oce_clim']  = oceclim_exp
        clim_all['oce_mean']  = ocemean_exp
        clim_all['ice_clim']  = iceclim_exp
        clim_all['ice_mean']  = icemean_exp
        clim_all['amoc_mean'] = amoc_mean_exp
        clim_all['amoc_ts'] = amoc_ts_exp
        if density:
            clim_all['rho_mean'] = rhomean_exp
            clim_all['rho_clim'] = rhoclim_exp

    return clim_all

def create_ds_exp(exp_dict):
    """
    Creates a multiexp dataset with a new dimension "exp".
    """
    if 'lat' in list(exp_dict.values())[0].coords:
        # Round latitudes to avoid problems when doing groupby
        okdict = {}
        for exp in exp_dict:
            okdict[exp] = exp_dict[exp].assign_coords(lat = exp_dict[exp].lat.round(2))
    else:
        okdict = exp_dict

    x_ds = xr.concat(okdict.values(), dim=pd.Index(okdict.keys(), name='exp'))
    return x_ds


####################################### PLOTS #######################################

def plot_amoc_2d(amoc_mean, exp = None, ax = None):
    if ax is None:
        fig, ax = plt.subplots(figsize = (12,8))

    if isinstance(amoc_mean, xr.Dataset):
        amoc_mean = amoc_mean['msftyz']

    amoc_mean.sel(basin = 2).plot.contourf(x = 'nav_lat', y = 'depthw', ylim = (3000, 0), xlim = (-30, 70), levels = np.arange(-16, 16.1, 2), ax = ax)
    ax.set_title(exp)

    return ax


def plot_amoc_ts(amoc_max, exp, ylim = (5, 20), ax = None, color = None, text_xshift = 10):
    if ax is None:
        fig, ax = plt.subplots()
    
    if isinstance(amoc_max, xr.Dataset):
        amoc_max = amoc_max['msftyz']

    amoc_max = amoc_max.groupby('time_counter.year').mean()

    amoc_max.plot(ylim=ylim, label = exp, ax = ax, color = color)
    ax.text(amoc_max.year[-1]+text_xshift, amoc_max[-1], exp, fontsize=12, ha='right', color = color)

    return ax


def plot_greg(atmmean_exp, exps, cart_out = cart_out, exp_type = 'PI', n_end = 20, imbalance = -0.9, ylim = None, colors = None, year_clim = None, rolling = None, s_dot = 1000, labels = True):
    """
    gregory plot
    """

    fig, ax = plt.subplots(figsize=(12, 8))

    ax.axhline(0., color = 'grey', lw = 0.5)

    if colors is None:
        colors = get_colors(exps)

    for exp, col in zip(exps, colors):
        if not rolling:
            ax.plot(atmmean_exp[exp].tas, atmmean_exp[exp].toa_net, label = exp, lw = 0.2, color = col)
        else:
            ax.plot(atmmean_exp[exp].tas.rolling(year = rolling, min_periods = 1).mean(), atmmean_exp[exp].toa_net.rolling(year = rolling, min_periods = 1).mean(), label = exp, lw = 0.2, color = col)
        #ax.scatter(atmmean_exp[exp].tas.sel(year = slice(1990, 2000)).mean(), atmmean_exp[exp].toa_net.sel(year = slice(1990, 2000)).mean(), s = 1000, color = 'red', marker = 'o')
        if year_clim is None:
            x, y = atmmean_exp[exp].tas.isel(year = slice(-n_end, None)).mean(), atmmean_exp[exp].toa_net.isel(year = slice(-n_end, None)).mean()
        else:
            x, y = atmmean_exp[exp].tas.sel(year = slice(year_clim[0], year_clim[1])).mean(), atmmean_exp[exp].toa_net.sel(year = slice(year_clim[0], year_clim[1])).mean()
        ax.scatter(x, y, s = s_dot, color = col, marker = 'o', alpha = 0.5, zorder = 3)
        if labels:
            ax.text(x+0.1, y+0.1, exp, fontsize=12, ha='right', color = col)

    ### plot target shades
    xlim_tot = ax.get_xlim()
    ylim_tot = ax.get_ylim()
    
    #PD
    tas_clim_PD = 273.15+14.5 # ERA5 
    #tas_clim_PD = 273.15+15 #Berkeley
    net_toa_clim_PD = 0.9
    
    #PI
    tas_clim_PI = 273.15+13.5
    net_toa_clim_PI = 0.

    if exp_type == 'PI' or exp_type == 'all':
        tas_clim = tas_clim_PI
        net_toa_clim = net_toa_clim_PI
        ylim_tot = (min(ylim_tot[0], 0.), max(ylim_tot[1], net_toa_clim - imbalance + 0.2))
        xlim_tot = (min(xlim_tot[0], tas_clim - 0.2), max(xlim_tot[1], tas_clim + 0.2))
        ax.fill_betweenx(np.arange(ylim_tot[0], ylim_tot[1], 0.1), tas_clim - 0.15, tas_clim + 0.15, color = 'grey', alpha = 0.2, edgecolor = None, zorder = 0)
        ax.fill_between(np.arange(xlim_tot[0], xlim_tot[1], 0.1), net_toa_clim - imbalance - 0.15, net_toa_clim - imbalance + 0.15, color = 'grey', alpha = 0.2, edgecolor = None, zorder = 0)
    
    if exp_type == 'PD' or exp_type == 'all':
        tas_clim = tas_clim_PD
        net_toa_clim = net_toa_clim_PD
        ylim_tot = (min(ylim_tot[0], 0.), max(ylim_tot[1], net_toa_clim - imbalance + 0.2))
        xlim_tot = (min(xlim_tot[0], tas_clim - 0.2), max(xlim_tot[1], tas_clim + 0.2))
        ax.fill_betweenx(np.arange(ylim_tot[0], ylim_tot[1], 0.1), tas_clim - 0.15, tas_clim + 0.15, color = 'burlywood', alpha = 0.2, edgecolor = None, zorder = 0)
        ax.fill_between(np.arange(xlim_tot[0], xlim_tot[1], 0.1), net_toa_clim - imbalance - 0.15, net_toa_clim - imbalance + 0.15, color = 'burlywood', alpha = 0.2, edgecolor = None, zorder = 0)

    
    ax.set_xlabel('GTAS (K)')
    ax.set_ylabel('net TOA (W/m$^2$)')
    #plt.legend()

    if ylim is not None:
        ax.set_ylim(ylim)

    name = '-'.join(exps)
    fig.savefig(cart_out + f'check_tuning_{name}.pdf')
    #fig.savefig(cart_out + f'check_tuning_{'-'.join(exps)}.pdf')
    plt.show()

    return fig



def plot_amoc_vs_gtas(clim_all, exps = None, cart_out = cart_out, exp_type = 'PI', n_end = 20, colors = None, labels = None, colors_legend = None, lw = 0.3, alpha = 0.5, background_color = None, year_clim = None, rolling = None, s_dot = 1000):
    fig, ax = plt.subplots(figsize=(12, 8))

    if exps is None:
        exps = clim_all['atm_mean'].keys()

    if colors is None:
        colors = get_colors(exps)

    # print('AAAAAA')
    # print(clim_all['amoc_ts'].keys())

    for exp, col in zip(exps, colors):
        if exp not in clim_all['amoc_ts']: 
            print(f'AMOC not computed for {exp}')
            continue
        if isinstance(clim_all['amoc_ts'][exp], xr.DataArray):
            y = clim_all['amoc_ts'][exp]
        else:
            y = clim_all['amoc_ts'][exp]['msftyz']

        x = clim_all['atm_mean'][exp]['tas']
        if 'time_counter' in y.dims:
            y = y.groupby('time_counter.year').mean()
            
        y = y.squeeze()

        if not rolling:
            ax.plot(x, y, label = exp, lw = lw, color = col)
        else:
            ax.plot(x.rolling(year = rolling, min_periods = 1).mean(), y.rolling(year = rolling, min_periods = 1).mean(), label = exp, lw = lw, color = col)
        
        if year_clim is None:
            x, y = x.isel(year = slice(-n_end, None)).mean(), y.isel(year = slice(-n_end, None)).mean()
        else:
            x, y = x.sel(year = slice(year_clim[0], year_clim[1])).mean(), y.sel(year = slice(year_clim[0], year_clim[1])).mean()

        ax.scatter(x, y, s = s_dot, color = col, marker = 'o', edgecolors = col, alpha = 0.5, zorder = 3)
        ax.text(x+0.1, y+0.1, exp, fontsize=12, ha='right', color = col)

    ax.set_xlabel('GTAS (K)')
    ax.set_ylabel('AMOC max (Sv)')

    xlim_tot = ax.get_xlim()
    ylim_tot = ax.get_ylim()
    
    #PD
    tas_clim_PD = 273.15+14.5 # ERA5 
    #tas_clim_PD = 273.15+15 #Berkeley
    
    #PI
    tas_clim_PI = 273.15+13.5

    if exp_type == 'PI' or exp_type == 'all':
        tas_clim = tas_clim_PI
        col = 'grey'
        ax.fill_betweenx(np.arange(ylim_tot[0], ylim_tot[1], 0.1), tas_clim - 0.15, tas_clim + 0.15, color = col, alpha = 0.2, edgecolor = None)
    if exp_type == 'PD' or exp_type == 'all':
        tas_clim = tas_clim_PD
        col = 'burlywood'
        ax.fill_betweenx(np.arange(ylim_tot[0], ylim_tot[1], 0.1), tas_clim - 0.15, tas_clim + 0.15, color = col, alpha = 0.2, edgecolor = None)
    
    ax.fill_between(np.arange(xlim_tot[0], xlim_tot[1], 0.1), 15, 20, color = col, alpha = 0.2, edgecolor = None)

    if background_color is not None:
        ax.set_facecolor(background_color)

    if labels is None:
        plt.legend()
    else:
        if colors_legend is None:
            if len(colors) == len(labels):
                colors_legend = colors
            else:
                raise ValueError("specify colors for labels in legend")
        legend_elements = [Patch(facecolor=col, label=lab) for col, lab in zip(colors_legend, labels)]

        ax.legend(handles=legend_elements)

    name = '-'.join(exps)
    fig.savefig(cart_out + f'check_amoc_vs_gtas_{name}.pdf')
    plt.show()

    return fig

def plot_custom_greg(x_ds, y_ds, x_target, y_target, color_var = None, exps = None, cart_out = cart_out, n_end = 20, colors = None, labels = None, colors_legend = None, lw = 0.3, alpha = 0.5, background_color = None, cmap_name = 'viridis', xlabel = '', ylabel = '', cbar_label = ''):

    if isinstance(x_ds, dict):
        x_ds = xr.concat(x_ds.values(), dim=pd.Index(x_ds.keys(), name='exp'))
    if isinstance(y_ds, dict):
        y_ds = xr.concat(y_ds.values(), dim=pd.Index(y_ds.keys(), name='exp'))

    fig, ax = plt.subplots(figsize=(12, 8))

    y_ext = (np.min(y_ds), np.max(y_ds))
    x_ext = (np.min(x_ds), np.max(x_ds))

    ax.fill_betweenx(np.arange(y_ext[0], y_ext[1], 0.1), x_target[0], x_target[1], color = 'grey', alpha = 0.2, edgecolor = None)
    ax.fill_betweenx(np.arange(y_target[0], y_target[1], 0.1), x_ext[0], x_ext[1], color = 'grey', alpha = 0.2, edgecolor = None)

    if exps is None:
        exps = x_ds.exp.values

    if color_var is not None:
        cmap = plt.cm.get_cmap(cmap_name)
        norm = Normalize(vmin=color_var.min(), vmax=color_var.max())

        colors = [cmap(norm(val)*0.8+0.1) for val in color_var]
    elif colors is None:
        colors = get_colors(exps)

    for exp, col in zip(exps, colors):
        x = x_ds.sel(exp = exp)
        y = y_ds.sel(exp = exp)

        ax.plot(x, y, label = exp, lw = lw, color = col)
        
        x, y = x.isel(year = slice(-n_end, None)).mean(), y.isel(year = slice(-n_end, None)).mean()
        ax.scatter(x, y, s = 1000, color = col, marker = 'o', edgecolors = col, alpha = 0.5, zorder = 3)
        ax.text(x+0.1, y+0.1, exp, fontsize=12, ha='right', color = col)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if background_color is not None:
        ax.set_facecolor(background_color)

    if labels is None:
        plt.legend()
    else:
        if colors_legend is None:
            if len(colors) == len(labels):
                colors_legend = colors
            else:
                raise ValueError("specify colors for labels in legend")
        legend_elements = [Patch(facecolor=col, label=lab) for col, lab in zip(colors_legend, labels)]

        ax.legend(handles=legend_elements)

    # Add colorbar below the graph
    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), 
                        ax=ax, orientation='vertical', pad=0.15)
    cbar.set_label(cbar_label)
    name = '-'.join(exps)
    fig.savefig(cart_out + f'check_{x_ds.name}_vs_{y_ds.name}_{name}.pdf')
    plt.show()

    return

def plot_zonal_fluxes_vs_ceres(atm_clim, exps, plot_anomalies = True, weighted = False, datadir = datadir, cart_out = cart_out, colors = None, ylim = None):
    """
    plot_anomalies: plots anomalies wrt CERES
    weighted: weights for cosine of latitude
    """
    ceresmean = xr.open_dataset(datadir + 'ceres_clim_2001-2011.nc')

    atmclim = create_ds_exp(atm_clim)
    atmclim = atmclim.groupby('lat').mean()
    atmclim['toa_net'] = atmclim.rsdt - (atmclim.rsut + atmclim.rlut)

    if weighted:
        weights = np.cos(np.deg2rad(atmclim.lat)).compute()

    #####

    ceres_vars = ['toa_lw_all_mon', 'toa_sw_all_mon', 'toa_net_all_mon', 'solar_mon']
    okvars = ['rlut', 'rsut', 'toa_net', 'rsdt']

    figs = []
    for var, cvar in zip(okvars, ceres_vars):
        fig, ax = plt.subplots(figsize=(12, 8))
        y_ref = ceresmean.interp(lat = atmclim.lat)[cvar]
        
        if plot_anomalies: ax.axhline(0., color = 'lightgrey')
        if colors is None: colors = get_colors(exps)

        for exp, col in zip(exps, colors):
            y = atmclim.sel(exp = exp)[var]
            if plot_anomalies: y -= y_ref
            if weighted: y *= weights

            ax.plot(atmclim.lat, y, label = exp, color = col)
            ax.text(100, y.values[-1], exp, fontsize=12, ha='right', color = col)
            
        if not plot_anomalies: 
            if weighted: y_ref *= weights
            ax.plot(atmclim.lat, y_ref, label = 'CERES', color = 'black')
            ax.text(100, y_ref.values[-1], 'CERES', fontsize=12, ha='right', color = 'black')

        ax.set_xlabel('lat')
        add = ''
        if weighted: add = ' (weighted with cosine)'
        if plot_anomalies:
            ax.set_ylabel(f'{var} bias wrt CERES 2001-2011 (W/m2)'+add)
        else:
            ax.set_ylabel(f'{var} vs CERES 2001-2011 (W/m2)'+add)

        plt.xlim(-90, 105)
        if ylim is not None: plt.ylim(ylim)
        #plt.legend()

        add = ''
        if not plot_anomalies: add = '_full'
        if weighted: add += '_weighted'

        name = '-'.join(exps)
        fig.savefig(cart_out + f'check_radiation_vs_ceres_{name}{add}.pdf')
        figs.append(fig)

    return figs

def plot_zonal_fluxes_vs_ref(atm_clim, exps, ref_exp, plot_anomalies=True, weighted=False, 
                             datadir=None, cart_out=None, colors=None, ylim=None):
    """
    plot_anomalies: plots anomalies wrt reference experiment
    weighted: weights for cosine of latitude
    """

    if cart_out is None:
        cart_out = './'

    atmclim = create_ds_exp(atm_clim)
    atmclim = atmclim.groupby('lat').mean()
    atmclim['toa_net'] = atmclim.rsdt - (atmclim.rsut + atmclim.rlut)

    if weighted:
        weights = np.cos(np.deg2rad(atmclim.lat)).compute()

    okvars = ['rlut', 'rsut', 'toa_net', 'rsdt']

    figs = []
    if colors is None:
        colors = get_colors(exps)

    for var in okvars:
        fig, ax = plt.subplots(figsize=(12, 8))
        y_ref = atmclim.sel(exp=ref_exp)[var]

        if plot_anomalies:
            ax.axhline(0., color='lightgrey')

        for exp, col in zip(exps, colors):
            if exp == ref_exp:
                continue  # non serve confrontare il ref con se stesso
            y = atmclim.sel(exp=exp)[var]
            if plot_anomalies:
                y = y - y_ref
            if weighted:
                y = y * weights

            ax.plot(atmclim.lat, y, label=exp, color=col)
            ax.text(float(atmclim.lat.max()), y.values[-1], exp, fontsize=12, ha='right', color=col)

        if not plot_anomalies:
            if weighted:
                y_ref = y_ref * weights
            ax.plot(atmclim.lat, y_ref, label=f'{ref_exp} (ref)', color='black', lw=2)
            ax.text(float(atmclim.lat.max()), y_ref.values[-1], ref_exp, fontsize=12, ha='right', color='black')

        ax.set_xlabel('Latitude')
        add = ''
        if weighted:
            add = ' (weighted with cosine)'
        if plot_anomalies:
            ax.set_ylabel(f'{var} bias wrt {ref_exp} (W/m2)' + add)
        else:
            ax.set_ylabel(f'{var} vs {ref_exp} (W/m2)' + add)

        plt.xlim(-90, 105)
        if ylim is not None:
            plt.ylim(ylim)

        add = ''
        if not plot_anomalies:
            add = '_full'
        if weighted:
            add += '_weighted'

        name = '-'.join(exps)
        figname = f'check_radiation_vs_ref_{ref_exp}_{var}_{name}{add}.pdf'
        fig.savefig(os.path.join(cart_out, figname))
        figs.append(fig)

    return figs

def plot_zonal_fluxes_by_param(atm_clim, ref_exp, param_map, cart_out, 
                               plot_anomalies=True, weighted=False, colors=None, ylim=None):
    """
    Genera un plot per ciascun parametro modificato (± variazione) confrontando vs ref_exp.

    param_map: dict con chiavi = parametri, valori = tuple (exp_minus, exp_plus)
    """

    atmclim = create_ds_exp(atm_clim)
    atmclim = atmclim.groupby('lat').mean()
    atmclim['toa_net'] = atmclim.rsdt - (atmclim.rsut + atmclim.rlut)

    if weighted:
        weights = np.cos(np.deg2rad(atmclim.lat)).compute()

    okvars = ['rlut', 'rsut', 'toa_net', 'rsdt']
    figs = []

    if colors is None:
        colors = ['#1f77b4', '#ff7f0e']  # blu = -%, arancio = +%

    for param, (exp_minus, exp_plus) in param_map.items():
        fig, axes = plt.subplots(len(okvars), 1, figsize=(12, 4*len(okvars)), sharex=True)

        for i, var in enumerate(okvars):
            ax = axes[i] if len(okvars) > 1 else axes
            y_ref = atmclim.sel(exp=ref_exp)[var]

            if plot_anomalies:
                ax.axhline(0., color='lightgrey')

            for exp, col, label in zip(
                [exp_minus, exp_plus],
                colors,
                [f"-50%", f"+50%"]
            ):
                y = atmclim.sel(exp=exp)[var]
                if plot_anomalies:
                    y = y - y_ref
                if weighted:
                    y = y * weights

                ax.plot(atmclim.lat, y, label=label, color=col, lw=2)
                ax.text(float(atmclim.lat.max()), y.values[-1], label, fontsize=11, ha='right', color=col)

            ax.set_ylabel(f"{var} (W/m2)")
            ax.set_title(f"{param} — {var}", fontsize=13)
            ax.grid(True, ls='--', alpha=0.3)
            if ylim is not None:
                ax.set_ylim(ylim)

        axes[-1].set_xlabel('Latitude')

        plt.suptitle(f"{param}: effect of ±50% variation vs {ref_exp}", fontsize=15)
        plt.xlim(-90, 90)
        plt.legend(loc='upper right')

        add = ''
        if weighted:
            add = '_weighted'

        figname = f'zonal_fluxes_{param}_vs_{ref_exp}{add}.pdf'
        fig.savefig(os.path.join(cart_out, figname), bbox_inches='tight')
        figs.append(fig)

    return figs

def plot_map_ocean(oce_clim, exps, var, ref_exp = None, vmin = None, vmax = None, xlabel = None, ylabel = None):
    """
    oce_clim is clim_all['oce_clim'] produced by read_output
    exps: list of experiments
    var: var name to plot
    ref_exp: if specified, plot differences to ref_exp

    TO BE IMPROVED: regrid and plot with cartopy

    """
    if ref_exp is not None and ref_exp not in exps:
        print(f'WARNING: {ref_exp} not in exps! plotting absolute values')
        ref_exp = None

    nx = int(np.ceil(np.sqrt(len(exps))))
    ny = int(np.ceil(len(exps)/nx))

    fig, axs = plt.subplots(nx, ny, figsize = (12, 12))
    
    if nx*ny > 1:
        axs = axs.flatten()
    else:
        axs = [axs]

    for exp, ax in zip(exps, axs):
        if ref_exp is not None:
            if exp == ref_exp:
                oce_clim[exp].tos.plot.pcolormesh(ax = ax)
                ax.set_title(exp)
            else:
                (oce_clim[exp]-oce_clim[ref_exp]).tos.plot.pcolormesh(vmin = vmin, vmax = vmax, ax = ax, cmap = 'RdBu_r')
                ax.set_title(f'{exp} - {ref_exp}')
        else:
            oce_clim[exp].tos.plot.pcolormesh(ax = ax, vmin = vmin, vmax = vmax)
            ax.set_title(exp)
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    
    plt.tight_layout()
    
    return fig

def plot_amoc_2d_all(amoc_mean, exps, cart_out = cart_out):
    nx = int(np.ceil(np.sqrt(len(exps))))
    ny = int(np.ceil(len(exps)/nx))
    fig, axs = plt.subplots(nx, ny, figsize = (12, 12))

    if nx*ny > 1:
        axs = axs.flatten()
    else:
        axs = [axs]

    for exp, ax in zip(exps, axs):
        if exp not in amoc_mean:
            print(f'AMOC not computed for {exp}')
            continue
        plot_amoc_2d(amoc_mean[exp], exp=exp, ax = ax)

    plt.tight_layout()
    name = '-'.join(exps)
    fig.savefig(cart_out + f'check_amoc_2d_{name}.pdf')
    return fig

def plot_zonal_tas_vs_ref(atmclim, exps, ref_exp = None, cart_out = cart_out, colors=None):
    # Missing tas reference
    atmclim = create_ds_exp(atmclim)
    atmclim = atmclim.groupby('lat').mean()

    if ref_exp is not None and ref_exp not in exps:
        print(f'WARNING: {ref_exp} not in exps! plotting absolute values')
        ref_exp = None

    fig, ax = plt.subplots(figsize=(12, 8))

    y_ref = None
    if ref_exp is not None:
        y_ref = atmclim.sel(exp = ref_exp)['tas']

    if colors is None:
        colors = get_colors(exps)

    for exp, col in zip(exps, colors):
        y = atmclim.sel(exp = exp)['tas']
        
        if y_ref is not None: y = y - y_ref

        plt.plot(atmclim.lat, y, label = exp, color = col)

        plt.text(100, y.values[-1], exp, fontsize=12, ha='right', color = col)
        
    ax.axhline(0., color = 'grey')
    plt.xlim(-90, 105)
    #plt.legend()
    
    ax.set_xlabel('lat')
    if ref_exp is not None:
        name = '-'.join([exp for exp in exps if exp != ref_exp])
        ax.set_ylabel(f'zonal temp diff wrt {ref_exp} (K)')
        fig.savefig(cart_out + f'check_zonal_tas_{name}_vs_{ref_exp}.pdf')
    else:    
        name = '-'.join(exps)
        ax.set_ylabel('zonal temp (K)')
        fig.savefig(cart_out + f'check_zonal_tas_{name}.pdf')


    return fig


def plot_zonal_var(atmclim, exps, var, ref_exp = None, cart_out = cart_out):
    # Missing tas reference
    atmclim = create_ds_exp(atmclim)
    atmclim = atmclim.groupby('lat').mean()

    if ref_exp is not None and ref_exp not in exps:
        print(f'WARNING: {ref_exp} not in exps! plotting absolute values')
        ref_exp = None

    fig, ax = plt.subplots(figsize=(12, 8))

    y_ref = None
    if ref_exp is not None:
        y_ref = atmclim.sel(exp = ref_exp)[var]

    colors = get_colors(exps)

    for exp, col in zip(exps, colors):
        y = atmclim.sel(exp = exp)[var]
        
        if y_ref is not None: y = y - y_ref

        plt.plot(atmclim.lat, y, label = exp, color = col)

        plt.text(100, y.values[-1], exp, fontsize=12, ha='right', color = col)
        
    ax.axhline(0., color = 'grey')
    plt.xlim(-90, 105)
    #plt.legend()
    
    ax.set_ylabel(var if ref_exp is None else var + f" (wrt {ref_exp})")
    ax.set_xlabel('lat')

    return fig


def plot_var_ts(clim_all, domain, vname, exps = None, ref_exp = None, rolling = None, norm_factor = 1., cart_out = cart_out, colors = None):
    """
    Plots timeseries of var "vname" in domain "domain" for all exps.

    Domain is one among: ['atm', 'oce', 'ice']
    """

    if domain not in ['atm', 'oce', 'ice', 'amoc', 'rho']:
        raise ValueError('domain should be one among: atm, oce, ice, amoc, rho')
    
    if domain == 'amoc':
        ts_dataset = clim_all[f'{domain}_ts']
    else:
        ts_dataset = clim_all[f'{domain}_mean']

    ts_dataset = {co: ts_dataset[co] for co in ts_dataset if ts_dataset[co] is not None}

    fig, ax = plt.subplots(figsize=(12, 8))

    if exps is None: exps = ts_dataset.keys()
    if len(ts_dataset.keys()) == 0: 
        print('NO data to plot!')
        return fig
    
    if len(ts_dataset.keys()) == 0: 
        print('NO data to plot!')
        return fig
    
    ts_dataset = create_ds_exp(ts_dataset)

    if ref_exp is not None and ref_exp not in exps:
        print(f'WARNING: {ref_exp} not in exps! plotting absolute values')
        ref_exp = None

    if isinstance(ts_dataset, xr.Dataset):
        ts_dataset = ts_dataset[vname]

    y_ref = None
    if ref_exp is not None:
        y_ref = norm_factor*ts_dataset.sel(exp = ref_exp)

    if colors is None:
        colors = get_colors(exps)

    for exp, col in zip(exps, colors):
        y = norm_factor*ts_dataset.sel(exp = exp)
        
        if y_ref is not None: y = y - y_ref

        if rolling is not None:
            y.rolling(year = rolling).mean().plot(label = exp, color = col, ax = ax)
        else:
            y.plot(label = exp, color = col, ax = ax)

        # ax.text(y.year[-1] + 5, np.nanmean(y.values[-30:]), exp, fontsize=12, ha='right', color = col) # not working for some evil reason
    
    ax.set_title('')
    ax.legend()
    name = '-'.join([exp for exp in exps])
    fig.savefig(cart_out + f'check_ts_{domain}_{vname}_{name}.pdf')
    
    return fig


def check_energy_balance_ocean(clim_all, remove_ice_formation = False):
    fact = 334*1000*1000/(3.1e7*4*3.14*6e6**2) # to convert sea ice formation in W/m2

    # (clim_all['oce_mean'][exp]['enebal']+clim_all['ice_mean'][exp]['sivolu_N'].diff('year')*fact).rolling(year = 20).mean().plot(label = exp, color = col, ls = ':')
    return

# ============================================================
# FUNCTIONS FOR PARAMETERS PLOTS
def load_param_values(folder):
    """
    Reads all tuning_XX.yml files in the specified folder and returns
    a dictionary with parameter values for each experiment.
    Also handles YAML files starting with '- base.context:'.
    """
    param_dict = {}
    for f in glob.glob(os.path.join(folder, "tuning_*.yml")):
        exp_name = os.path.basename(f).replace("tuning_", "").replace(".yml", "")
        with open(f) as fin:
            data = yaml.safe_load(fin)

        # If it's a list extract the first element
        if isinstance(data, list) and len(data) > 0:
            data = data[0]

        try:
            tuning = data['base.context']['model_config']['oifs']['tuning']
        except Exception as e:
            print(f"⚠️ Skipping {f}: unexpected YAML structure ({type(data)}). Error: {e}")
            continue

        params = {}
        for block in tuning.values():
            for k, v in block.items():
                if v is not None:
                    try:
                        params[k] = float(v)
                    except ValueError:
                        print(f"⚠️ Non-numeric value for {k} in {f}: {v}")
        param_dict[exp_name] = params

    print(f"Loaded {len(param_dict)} tuning files from {folder}")
    return param_dict

def compute_slope_and_linearity(ds_minus, ds_ref, ds_plus, param_name, param_values, var='toa_net'):
    """
    Calculate the slope (normalized change with respect to the parameter change)
    and the coefficient of determination R² for each spatial point.
    """

    # Temporal mean → get 2D maps
    if 'year' in ds_minus.dims:
        y_minus = ds_minus[var].mean('year')
        y_ref   = ds_ref[var].mean('year')
        y_plus  = ds_plus[var].mean('year')
    elif 'time_counter' in ds_minus.dims:
        y_minus = ds_minus[var].mean('time_counter')
        y_ref   = ds_ref[var].mean('time_counter')
        y_plus  = ds_plus[var].mean('time_counter')
    else:
        raise ValueError("No time dimension found ('year' or 'time_counter')")

    # Parameter values (x)
    x_vals = np.array([
        param_values['minus'][param_name],
        param_values['ref'][param_name],
        param_values['plus'][param_name]
    ])

    # Stack the 3 simulations into a single DataArray
    y_stack = xr.concat([y_minus, y_ref, y_plus], dim='param_change')
    y_stack = y_stack.assign_coords(param_change=x_vals)

    y_stack = y_stack.chunk({'param_change': -1})

    # Linear regression function for each cell
    def linfit(x, y):
        p = np.polyfit(x, y, 1)
        slope = p[0]
        corr = np.corrcoef(x, y)[0, 1]
        return slope, corr**2  # returns slope and R²

    # Apply vectorized over all cells
    slope, r2 = xr.apply_ufunc(
        linfit,
        y_stack.param_change,
        y_stack,
        input_core_dims=[["param_change"], ["param_change"]],
        output_core_dims=[[], []],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float, float]
    )

    slope.attrs["r2_mean"] = float(r2.mean().values)
    slope.attrs["r2_min"] = float(r2.min().values)

    return slope, r2

def mask_insignificant(slope, ds_minus, ds_ref, ds_plus, var='toa_net', threshold=0.1):
    """
    Maschera i punti dove la risposta è inferiore a una frazione del range massimo.
    """
    y_minus = ds_minus[var]
    y_plus = ds_plus[var]
    response_range = np.abs(y_plus - y_minus)
    max_change = response_range.max()
    mask = response_range < (threshold * max_change)
    slope_masked = slope.where(~mask)
    slope_masked.attrs['mask_info'] = f"Masked where Δresponse < {threshold*100:.1f}% of max"
    return slope_masked

def regrid_to_regular_smm_safe(ds, target_grid="r180x90", method="ycon", grid_in=None):
    import shutil
    from smmregrid import cdo_generate_weights, Regridder

    os.environ["PATH"] += ":/usr/local/apps/cdo/2.5.1/bin"
    os.environ["CDO_PTHREADS"] = "1"

    if shutil.which("cdo") is None:
        print("CDO not found in PATH. Skip regridding.")
        return ds

    if 'cell' not in ds.dims:
        print("Dataset already on regular grid. Skip regrid.")
        return ds

    # If not provided, take the first timestep of the dataset
    if grid_in is None:
        grid_in = ds.isel(time_counter=0)

    try:
        weights = cdo_generate_weights(grid_in, target_grid=target_grid, method=method)
        regridder = Regridder(weights=weights)
        ds_reg = regridder.regrid(ds)
        print(f"Regridding completed on {target_grid}")
        return ds_reg
    except Exception as e:
        print(f"Regridding failed: {e}")
        return ds
    

def plot_all_slopes(slope_dict, r2_dict=None, vmin=-3, vmax=3, cmap='RdBu_r',
                    r2_thresh=0.5, filename=None, label='Net TOA (W/m²)'):
    """
    Creates a single figure with all slope maps.
    If r2_dict is provided, highlights statistically significant areas (R² > r2_thresh).
    """
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import numpy as np
    import math

    n = len(slope_dict)
    ncols = math.ceil(math.sqrt(n))
    nrows = math.ceil(n / ncols)

    fig, axs = plt.subplots(
        nrows, ncols,
        figsize=(4*ncols, 2.5*nrows),
        subplot_kw={'projection': ccrs.PlateCarree()}
    )

    if nrows*ncols > 1:
        axs = axs.flatten()
    else:
        axs = [axs]

    for i, (param, field) in enumerate(slope_dict.items()):
        ax = axs[i]

        # Reduce slope to 2D if necessary
        extra_dims = [d for d in field.dims if d not in ['lat', 'lon']]
        if extra_dims:
            print(f"Slope {param} has extra dimensions {extra_dims}, averaging.")
            field = field.mean(extra_dims)

        data = field.values
        lon2d, lat2d = np.meshgrid(field['lon'], field['lat'])

        # Alpha based on R²
        alpha_mask = 1.0
        if r2_dict is not None and param in r2_dict:
            r2_field = r2_dict[param]
            extra_dims_r2 = [d for d in r2_field.dims if d not in ['lat', 'lon']]
            if extra_dims_r2:
                r2_field = r2_field.mean(extra_dims_r2)
            r2_data = r2_field.interp_like(field, method="nearest").values
            alpha_mask = np.where(r2_data >= r2_thresh, 1.0, 0.3)

        # Plot
        im = ax.pcolormesh(
            lon2d, lat2d, data,
            vmin=vmin, vmax=vmax, cmap=cmap,
            alpha=alpha_mask,
            transform=ccrs.PlateCarree(),
            shading="auto"
        )

        ax.coastlines(linewidth=0.5)
        ax.set_title(param, fontsize=12)

    for ax in axs[len(slope_dict):]:
        ax.remove()

    # Common colorbar
    cbar_ax = fig.add_axes([0.25, 0.08, 0.5, 0.03])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
    cbar.set_label(label)

    if filename:
        plt.savefig(filename, bbox_inches='tight')
    plt.show()


# Wrapper for slope and plots
def calc_and_plot_slopes_from_raw(param_map, ref_exp='k000', user=None,
                                  cart_exp='/ec/res4/scratch/{}/ece4/', var='toa_net',
                                  threshold=0.1, target_grid='r180x90', r2_thresh=0.5):
    """
    Calculates slope and R² for each parameter, then shows two sets of maps:
      (1) slope normalized per 1%
      (2) total anomaly minus→plus
    Masks non-significant areas based on R².
    """
    import os, dask
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    dask.config.set(scheduler='single-threaded')

    slope_dict, r2_dict, anom_full_dict, slope_30pct_dict = {}, {}, {}, {}

    # --- load tuning values
    param_folder = '/ec/res4/hpcperm/ecme3038/ecearth/ecearth4/ECtuner/exps_415/'
    param_yaml = load_param_values(param_folder)

    def normalize_exp_key(exp, available_keys):
        exp_num = exp.replace('k', '').lstrip('0') or '0'
        key = exp_num.zfill(2)
        if key not in available_keys:
            raise KeyError(f"No matching experiment '{exp}' in YAML ({list(available_keys)})")
        return key

    for param, exps in param_map.items():

        print(f"\n=== PARAM {param} | exps = {exps}")

        if len(exps) != 2:
            print(f"Parameter {param} does not have two experiments. Skip.")
            continue

        exp_minus, exp_plus = exps
        exp_list = [exp_minus, ref_exp, exp_plus]
        ds_dict = {}

        # --- load dataset
        for exp in exp_list:
            filz = glob.glob(f'{cart_exp.format(user)}/{exp}/output/oifs/{exp}_atm_cmip6_1m_*.nc')
            if not filz:
                raise FileNotFoundError(f"NetCDF files not found for {exp}")

            ds = xr.open_mfdataset(filz, decode_times=time_coder, chunks={})
            ds = ds[['rsut', 'rlut', 'rsdt', 'tas']]
            if 'cell' in ds.dims:
                print(f"Regridding {exp} with CDO on {target_grid}...")
                grid_file = filz[0]
                grid_in = xr.open_dataset(grid_file).isel(time_counter=0)
                ds = regrid_to_regular_smm_safe(ds, target_grid=target_grid, method="ycon", grid_in=grid_in)
                print(f"Regrid completed: dims = {list(ds.dims.keys())}")
            ds['toa_net'] = ds['rsdt'] - ds['rlut'] - ds['rsut']
            ds = ds.rename({'time_counter': 'time'}).chunk({'time': 240})
            ds = ds.groupby('time.year').mean()
            ds_dict[exp] = ds

            print(f"  Loaded dataset for {exp}: dims={list(ds.dims.keys())}")

        # --- retrieve parameter values
        print("  YAML keys:", sorted(param_yaml.keys()))
        print(f"  Trying to match exps: minus={exp_minus}, ref={ref_exp}, plus={exp_plus}")
        try:
            key_minus = normalize_exp_key(exp_minus, param_yaml.keys())
            key_ref   = normalize_exp_key(ref_exp,   param_yaml.keys())
            key_plus  = normalize_exp_key(exp_plus,  param_yaml.keys())
        except KeyError as e:
            print(f"❌ Skip {param}: {e}")
            print("   Available YAML keys:", sorted(param_yaml.keys()))
            continue

        p_minus = float(param_yaml[key_minus][param])
        p_ref   = float(param_yaml[key_ref][param])
        p_plus  = float(param_yaml[key_plus][param])
        param_values = {'minus': {param: p_minus}, 'ref': {param: p_ref}, 'plus': {param: p_plus}}

        # --- calculate slope and linearity
        slope, r2 = compute_slope_and_linearity(ds_dict[exp_minus], ds_dict[ref_exp], ds_dict[exp_plus],
                                                param, param_values, var=var)

        # --- total anomaly (minus→plus)
        delta_full = p_plus - p_minus
        anom_full = slope * delta_full
        anom_full.name = f"{param}_anom_full"
        anom_full.attrs['units'] = 'W/m²'
        anom_full.attrs['descr'] = f"TOA change for Δparam={delta_full:.3g}"

        # --- slope normalized per 1%
        slope_per30pct = slope * (abs(p_ref) * 0.3 if p_ref not in [0, None, np.nan] else np.nan)
        slope_per30pct.name = f"{param}_slope_per30pct"
        slope_per30pct.attrs['units'] = 'W/m² per 30%'

        slope_dict[param] = slope
        r2_dict[param] = r2
        anom_full_dict[param] = anom_full
        slope_30pct_dict[param] = slope_per30pct

        r2_mean = slope.attrs.get('r2_mean', np.nan)
        r2_min  = slope.attrs.get('r2_min', np.nan)
        print(f" {param}: mean R²={r2_mean:.3f}, min R²={r2_min:.3f}")

    # --- Plot 1: slope per 30%
    print("\nPlot 1: Sensitivity normalized (W/m² per 30%)")
    plot_all_slopes(slope_30pct_dict, r2_dict=r2_dict, vmin=-3, vmax=3, cmap='RdBu_r', r2_thresh=r2_thresh,
                    filename='plot_slope_per30pct.png', label='TOA Net (W/m² per 30% param change)')

    # --- Plot 2: physical effect (total anomaly)
    print("\nPlot 2: Total effect minus→plus (W/m²)")
    plot_all_slopes(anom_full_dict, r2_dict=r2_dict, vmin=-10, vmax=10, cmap='RdBu_r', r2_thresh=r2_thresh,
                    filename='plot_anom_full.png', label='TOA Net anomaly (W/m²)')

    return slope_dict, r2_dict, slope_30pct_dict, anom_full_dict

#####################################################################################################################################################################

def global_mean_oce_2d_region(ds, exp, user, var, cart_exp=cart_exp,
                              lats=None, lons=None,
                              compute=True, grid='T'):

    area = get_areas_nemo(exp, user, cart_exp=cart_exp, grid=grid)
    field = ds[var]
    mask = np.ones_like(area, dtype=bool)

    try:
        if lats is not None:
            mask &= (field.nav_lat > lats[0]) & (field.nav_lat < lats[1])
        if lons is not None:
            mask &= (field.nav_lon > lons[0]) & (field.nav_lon < lons[1])
    except:
        if lats is not None:
            mask &= (field.nav_lat_grid_T > lats[0]) & (field.nav_lat_grid_T < lats[1])
        if lons is not None:
            mask &= (field.nav_lon_grid_T > lons[0]) & (field.nav_lon_grid_T < lons[1])

    area_reg = area * mask

    if compute:  
        mean = ((field * area_reg).sum(['x', 'y'])/ area_reg.sum())
        return mean.compute() 
    
    else:
        return field *mask

def compute_base_anom_state(exps, user=None, read_again=[], cart_exp='/ec/res4/scratch/{}/ece4/',
                             cart_out='./output/', ref_exp=None, atm_only=False,
                             year_clim=None, year='1850', density=False, color=None,
                             var_list=None, otexps=False):
    """
    Compute anomaly state for a list of variables and associated domains.

    Parameters
    ----------
    var_list : list of dicts, each with keys:
        'name'   : str   - variable name, e.g. 'tas'
        'domain' : str   - one of 'atm', 'oce', 'ice', 'amoc', 'rho', 'oce_map'
        'region' : str, optional - 'SO', 'NA', or '' for global
        'yr70'   : bool, optional - whether to return years 55-85 array (True) or scalar (False)
                   (ignored when otexps=True)
    otexps : bool
        If False (default): coupled-experiment mode. Computes anomaly relative to a
        linear-detrended reference experiment (ref_exp), with pi_mean/pi_std from years
        120-150 of ref_exp, and yr70/yr150 anomalies relative to it.
        If True: single-experiment / "ot" mode. No reference experiment or detrending;
        just reads the last 20 years of exps directly as pi_mean/pi_std.

    Example
    -------
    var_list = [{'name': 'tas', 'domain': 'atm', 'yr70': False}]
    """
    if cart_out is None:
        raise ValueError('cart_out not specified!')
    if not os.path.exists(cart_out):
        os.mkdir(cart_out)
    if var_list is None:
        raise ValueError('var_list must be specified!')

    # ── collect which variables are needed per domain ─────────────────────────
    domain_vars = {}
    for v in var_list:
        domain_vars.setdefault(v['domain'], []).append(v['name'])

    clim_all = read_output(exps, user=user, read_again=read_again, cart_exp=cart_exp,
                           cart_out=cart_out, atm_only=atm_only, year_clim=year_clim, density=density)

    map_all = read_output_map(exps, user=user, read_again=read_again, cart_exp=cart_exp,
                           cart_out=cart_out, atm_only=atm_only, year_clim=year_clim, density=density)

    # ── load datasets per domain ──────────────────────────────────────────────
    def load_domain(clim_dict, key):
        ds = clim_dict[key]
        ds = {co: ds[co] for co in ds if ds[co] is not None}
        return create_ds_exp(ds)

    domain_map = {
        'atm'    : lambda: load_domain(clim_all, 'atm_mean'),
        'oce'    : lambda: load_domain(clim_all, 'oce_mean'),
        'ice'    : lambda: load_domain(clim_all, 'ice_mean'),
        'amoc'   : lambda: load_domain(clim_all, 'amoc_ts'),
        'rho'    : lambda: load_domain(clim_all, 'rho_clim'),
        'oce_map': lambda: load_domain(map_all,  'oce_map'),
    }

    loaded_domains = {}
    for domain in domain_vars:
        if domain in domain_map:
            loaded_domains[domain] = domain_map[domain]()

    # ── compute anomalies / values ───────────────────────────────────────────
    results = {}

    # experiment selector differs: list-based (coupled) vs direct (otexps)
    exp0 = exps if otexps else exps[0]

    for v in var_list:
        vname  = v['name']
        domain = v['domain']
        region = v.get('region', '')
        yr70   = v.get('yr70', False)

        if domain == 'rho':
            continue   # handled separately below

        ds = loaded_domains[domain]

        if domain == 'amoc':
            var = ds.sel(x=0)
            var.name = vname
        else:
            var = ds[vname]

        # region-specific extraction
        if vname == 'tos' and region == 'SO':
            if otexps:
                var_exp = global_mean_oce_2d_region(ds.sel(exp=exp0), exp0[0], 'itcv', vname,
                                                    cart_exp, lats=[-60, -30])
            else:
                var_ref = global_mean_oce_2d_region(ds.sel(exp=ref_exp), ref_exp, 'itcv', vname,
                                                    cart_exp, lats=[-60, -30])
                var_exp = global_mean_oce_2d_region(ds.sel(exp=exp0), exp0, 'itcv', vname,
                                                    cart_exp, lats=[-60, -30])

        elif vname == 'tos' and region == 'NA':
            na_lats = [45, 60]
            na_lons = [-60, 0]
            if otexps:
                var_exp = global_mean_oce_2d_region(ds.sel(exp=exp0), exp0[0], 'itcv', vname,
                                                    cart_exp, lats=na_lats, lons=na_lons)
            else:
                var_ref = global_mean_oce_2d_region(ds.sel(exp=ref_exp), ref_exp, 'itcv', vname,
                                                    cart_exp, lats=na_lats, lons=na_lons)
                var_exp = global_mean_oce_2d_region(ds.sel(exp=exp0), exp0, 'itcv', vname,
                                                    cart_exp, lats=na_lats, lons=na_lons)
        elif vname == 'tos' and region == 'global':
            na_lats = [-60, 60]
            if otexps:
                var_exp = global_mean_oce_2d_region(ds.sel(exp=exp0), exp0[0], 'itcv', vname,
                                                    cart_exp, lats=na_lats)
            else:
                var_ref = global_mean_oce_2d_region(ds.sel(exp=ref_exp), ref_exp, 'itcv', vname,
                                                    cart_exp, lats=na_lats)
                var_exp = global_mean_oce_2d_region(ds.sel(exp=exp0), exp0, 'itcv', vname,
                                                    cart_exp, lats=na_lats)

        elif vname == 'mldr10_1':
            if otexps:
                field_exp = global_mean_oce_2d_region(ds.sel(exp=exp0), exp0[0], 'itcv', vname,
                                                   cart_exp, lats=[50, 70], lons=[-60, 0], compute=False)
                var_exp = field_exp.max(axis=(1, 2))
            else:
                field_ref = global_mean_oce_2d_region(ds.sel(exp=ref_exp), ref_exp, 'itcv', vname,
                                                    cart_exp, lats=[50, 70], lons=[-60, 0], compute=False)
                var_ref = field_ref.max(axis=(1, 2))
                field_exp = global_mean_oce_2d_region(ds.sel(exp=exp0), exp0, 'itcv', vname,
                                                    cart_exp, lats=[50, 70], lons=[-60, 0], compute=False)
                var_exp = field_exp.max(axis=(1, 2))

        else:
            var_exp = var.sel(exp=exp0)
            if not otexps:
                var_ref = var.sel(exp=ref_exp)

        # ── build results: differ by mode ─────────────────────────────────────
        if otexps:
            results[vname + region] = {
                'pi_mean': float(var_exp[-20:].mean().values),
                'pi_std' : float(var_exp[-20:].std().values),
            }
        else:
            trend, intercept, _, _, _ = stats.linregress(np.arange(0, len(var_ref)), var_ref.values)
            x = var_exp - (trend * np.arange(0, len(var_ref)) + intercept)

            results[vname + region] = {
                'pi_mean': float(var_ref[120:150].mean().values),
                'pi_std' : float(var_ref[120:150].std().values),
                'yr70'   : x[55:85].values if yr70 else float(x[55:85].mean().values),
                'yr150'  : float(x[-1].values),
            }

    # ── 3D rho variables ──────────────────────────────────────────────────────
    rho_vars = [v for v in var_list if v['domain'] == 'rho']
    if rho_vars:
        rho_dataset = loaded_domains['rho']
        rho_exp = exps if otexps else ref_exp   # otexps uses exps directly, coupled uses ref_exp

        for v in rho_vars:
            vname      = v['name']
            lev_bounds = v.get('lev_bounds', [0, 500])
            lats       = v.get('lats',       [-90, 90])
            regions    = v.get('regions',    {})

            var_mean = global_mean_oce_3d(rho_dataset.sel(exp=rho_exp), rho_exp, 'itcv',
                                          vname, depth_mean=True, year=year)[vname]
            var_deep = global_mean_oce_3d_region(rho_dataset.sel(exp=rho_exp), rho_exp, 'itcv',
                                                  vname, depth_mean=True, lats=lats,
                                                  lev_bounds=lev_bounds, year=year)[vname]

            results[vname] = {
                'pi_full': float(var_mean.values),
                'pi_mean': float(var_deep.values),
            }

            for rname, rparams in regions.items():
                val = global_mean_oce_3d_region(rho_dataset.sel(exp=rho_exp), rho_exp, 'itcv',
                                                 vname, depth_mean=True,
                                                 lats=rparams['lats'],
                                                 lev_bounds=rparams['lev_bounds'],
                                                 year=year)[vname]
                results[vname][rname] = float(val.values)

    return results
   
def return_vars(exps, user = None, vnames = None, domain=None, lat_bounds = [-90,90], lon_bounds = [0,360], read_again = [], ax=None, cart_exp = '/ec/res4/scratch/{}/ece4/', cart_out = './output/', atm_only = False, atmvars = 'rsut rlut rsdt tas'.split(), ocevars = 'tos heatc qt_oce sos'.split(), year_clim = None, density=False, flag_omega = False):
    """
    Returns vnames variables at year 70
    """

    cart_out_nc = cart_out + '/exps_clim/'
    #cart_out_figs = cart_out + f'/check_{'-'.join(exps)}/'
    
    map_all = read_output_map(exps, user = user, read_again = read_again, cart_exp = cart_exp, cart_out = cart_out_nc, atm_only = atm_only, atmvars = atmvars, ocevars = ocevars, year_clim = year_clim, density=density, flag_omega=flag_omega)

    dataset = map_all[f'{domain}_map']
    dataset = {co: dataset[co] for co in dataset if dataset[co] is not None}
    dataset = create_ds_exp(dataset)

    if domain == 'oce':

        lats = dataset[vnames[0]].nav_lat
        lons = dataset[vnames[0]].nav_lon
        areas = get_areas_nemo(exps[0], 'itcv', cart_exp = cart_exp)
        mask = get_mask_nemo(exps[0], 'itcv', cart_exp = cart_exp, grid = 'T')
        nx, ny = np.shape(areas)
        areas_ds = var.sel(exp=exps[0])[0].copy()
        areas_ds.values = areas*mask
    
    elif domain in ['atm','atm3d' ,'cre']:
        
        lats = dataset[vnames[0]].lat
        lons = dataset[vnames[0]].lon
        nx = len(lats)
        ny = len(lons)
        weights = np.cos(np.deg2rad(lats))
        areas_ds = weights
    
    area_reg = areas_ds.where(((lats < lat_bounds[1]) & (lats > lat_bounds[0])), np.nan)
    
    if(lon_bounds[0] < lon_bounds[1]):
        area_box = area_reg.where(((lons < lon_bounds[1]) & (lons > lon_bounds[0])), np.nan)
    else:
        area_box = area_reg.where(((lons < lon_bounds[1]) | (lons > lon_bounds[0])), np.nan)
    
    results = {}

    for vname in vnames:

        var = dataset[vname]
        if(vname == 'wap'):
            var = var.sel(pressure_levels=50000)
        trend = np.zeros([nx,ny])
        intercept = np.zeros([nx,ny])

        for i in range(nx):
            for j in range(ny):

                trend[i,j], intercept[i,j], _, _, _ = stats.linregress(np.arange(0,np.shape(var.sel(exp=exps[1]))[0]), var.sel(exp=exps[1])[:,i,j])

        delta = (var.sel(exp=exps[0]) - (np.arange(0,len(var.sel(exp=exps[0]))))[:, np.newaxis, np.newaxis]*trend-intercept).rolling(year=30).mean()
        #delta = (var.sel(exp=exps[0]) - (np.arange(0,300))[:, np.newaxis, np.newaxis]*trend).rolling(year=30).mean()

    
        # fig,ax = plt.subplots(1,1,subplot_kw={'projection': ccrs.PlateCarree()})
        # d = ax.pcolormesh(lons, lats, area_box, transform=ccrs.PlateCarree())
        # ax.coastlines()
        # plt.show()
        if vname =='wap' or vname =='omega':
            clima = var.sel(exp=exps[1]).mean(dim='year')
            omega_up = delta.where(clima>0, np.nan)
            #omega_up = omega_up.where(delta<0, np.nan)

            fig,ax = plt.subplots(1,2,subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)})
            d = ax[0].pcolormesh(lons, lats, omega_up[85]*area_box, cmap='PuOr',transform=ccrs.PlateCarree())
            ax[0].coastlines()
            ax[0].set_extent([70, 290, -30, 30], crs=ccrs.PlateCarree())
            #plt.colorbar(d, ax=ax[0])
            omega = var.sel(exp=exps[0]).rolling(year=30).mean(dim='year')[85]
            # e = ax[1].pcolormesh(lons, lats, delta.where(omega<-0.02, np.nan)[85], cmap='PuOr', transform=ccrs.PlateCarree())
            # ax[1].coastlines()
            # plt.show()

            var_box = (omega_up*area_box).sum(axis=(1,2))/area_box.sum()
        else:
            if domain == 'oce':
                var_box = (delta*area_box).sum(axis=(1,2))/np.nansum(area_box)
            else :
                var_box = (delta*area_box).sum(axis=(1,2))/area_box.sum()

        results[vname] = var_box[85].values

    return results

def extract_variables_at_level(input_file, output_file, variables, level, level_dim='pressure_levels', level_units='Pa'):
    """
    Extract variables at a specific pressure level and save to a new netCDF file.

    Parameters
    ----------
    input_file  : str  - path to input netCDF file
    output_file : str  - path to output netCDF file
    variables   : list - variable names to extract, e.g. ['omega', 'ua']
    level       : float - level value to extract, e.g. 50000 (Pa) or 500 (hPa)
    level_dim   : str  - name of the vertical dimension in the file
    level_units : str  - just for metadata, 'Pa' or 'hPa'
    """
    ds = xr.open_dataset(input_file)

    extracted = {}
    for var in variables:
        if var not in ds:
            raise KeyError(f"Variable '{var}' not found in {input_file}. Available: {list(ds.data_vars)}")
        extracted[var] = ds[var].sel({level_dim: level}, method='nearest')

    ds_out = xr.Dataset(extracted)
    ds_out.attrs = ds.attrs  # carry over global attributes
    ds_out.attrs['extracted_level'] = f"{level} {level_units}"

    ds_out.to_netcdf(output_file)
    print(f"Saved {variables} at {level} {level_units} → {output_file}")

    return ds_out

def compute_atm_streamfunction(exps, user = None, cart_exp = '/ec/res4/scratch/{}/ece4/', cart_out = './output/', atm_only = False, atmvars = 'rsut rlut rsdt tas'.split(), ocevars = 'tos heatc qt_oce sos'.split(), year_clim = None, density=False):

    cart_out_nc = cart_out + '/exps_clim/'
    
    map_all = read_output_map(exps, user = user, read_again = [], cart_exp = cart_exp, cart_out = cart_out_nc, atm_only = atm_only, atmvars = atmvars, ocevars = ocevars, year_clim = year_clim, density=density)

    dataset = map_all[f'atm3d_map']
    dataset = {co: dataset[co] for co in dataset if dataset[co] is not None}

    dataset = create_ds_exp(dataset)
    var = dataset['va']
    var_u = dataset['ua']

    #streamfunction computation
    var_zonal = var.sel(exp=exps[0]).mean(dim='lon')
    v = var_zonal.transpose('year','lat', 'pressure_levels')
    
    plev = v.pressure_levels

    a = 6378000 #m
    g = 9.81  #m/s2

    dp = xr.DataArray(np.gradient(plev),dims=["pressure_levels"],coords={"pressure_levels": plev})

    v_int = (v * dp).cumsum(dim='pressure_levels')
    coslat = np.cos(np.deg2rad(v.lat))
    psi = (2 * np.pi * a / g) * coslat * v_int
    psi.name = "psi"
    psi.attrs["long_name"] = "Atmospheric mass streamfunction"
    psi.attrs["units"] = "kg s-1"

    # ── Omega (vertical pressure velocity) ───────────────────────────────────
    # Select experiment and get full 3D (lat, lon, lev) fields
    ua = var_u.sel(exp=exps[0])   # (year, lat, lon, pressure_levels)
    va = var.sel(exp=exps[0])     # (year, lat, lon, pressure_levels)

    # Spherical geometry: convert lat/lon to radians
    lat_rad = np.deg2rad(ua.lat)   # (lat,)
    lon_rad = np.deg2rad(ua.lon)   # (lon,)

    # Grid spacings in radians
    dlat = xr.DataArray(np.gradient(lat_rad), dims=['lat'], coords={'lat': ua.lat})
    dlon = xr.DataArray(np.gradient(lon_rad), dims=['lon'], coords={'lon': ua.lon})

    # Spherical divergence: (1/a*cos(lat)) * (du/dlon + d(v*cos(lat))/dlat)
    coslat_2d = np.cos(lat_rad)   # broadcast over lon automatically

    du_dlon = ua.differentiate('lon') / dlon / (a * coslat_2d)
    vcoslat  = va * coslat_2d
    dvcoslat_dlat = vcoslat.differentiate('lat') / dlat / (a * coslat_2d)
    div = du_dlon + dvcoslat_dlat   # (year, lat, lon, pressure_levels)

    # Integrate downward from TOA: omega(p) = -integral_0^p div dp'
    # Pressure levels must be ordered top→bottom (ascending pressure values)
    plev_vals = ua.pressure_levels.values
    if plev_vals[0] > plev_vals[-1]:
        # levels are bottom→top: flip, integrate, flip back
        div = div.isel(pressure_levels=slice(None, None, -1))

    dp_3d = xr.DataArray(
        np.gradient(div.pressure_levels.values),
        dims=['pressure_levels'],
        coords={'pressure_levels': div.pressure_levels}
    )
    omega = -(div * dp_3d).cumsum(dim='pressure_levels')

    # Flip back if we reversed earlier
    if plev_vals[0] > plev_vals[-1]:
        omega = omega.isel(pressure_levels=slice(None, None, -1))
        omega = omega.assign_coords(pressure_levels=ua.pressure_levels)

    omega.name = "omega"
    omega.attrs["long_name"] = "Vertical pressure velocity"
    omega.attrs["units"] = "Pa s-1"

    atm3d = xr.open_mfdataset(cart_out_nc + f'map_atm3d_tuning_{exps[0]}.nc') 
    atm3d['psi'] = psi
    atm3d['omega'] = omega
    atm3d.to_netcdf(cart_out_nc + f'map_atm3d_tuning_{exps[0]}_analysis.nc')

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Left: streamfunction (zonal mean, pressure vs lat)
    ax = axes[0]
    cf = ax.contourf(v.lat, plev / 100, psi.mean(dim='year').T, levels=20, cmap='RdBu_r')
    ax.invert_yaxis()
    plt.colorbar(cf, ax=ax, label="kg s$^{-1}$")
    ax.set_xlabel("Latitude")
    ax.set_ylabel("Pressure (hPa)")
    ax.set_title("Mean Streamfunction")

    # Right: omega at 500 hPa (lat-lon map)
    ax = axes[1]
    omega_500 = omega.mean(dim='year').sel(pressure_levels=50000, method='nearest')  # Pa → nearest 500 hPa
    cf2 = ax.contourf(omega.lon, omega.lat, omega_500, levels=20, cmap='RdBu_r', vcenter=0, vmax=0.1)
    plt.colorbar(cf2, ax=ax, label="Pa s$^{-1}$")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("$\\omega$ at 500 hPa (mean)")

    plt.tight_layout()
    plt.show()

def compute_heat_transport(exps, user=None, cart_exp='/ec/res4/scratch/{}/ece4/', cart_out='./output/',
                           atm_only=False,
                           atmvars='rsut rlut rsdt tas hfss hfls rlns rsns'.split(),
                           ocevars='tos qt_oce'.split(), year_clim=None, density=False):
    """
    Compute implied meridional heat transports (THT, AHT, OHT) from TOA and surface fluxes.
    Reads explicit ocean heat transport (hfbasin) from netCDF output.
    Saves all timeseries to netCDF and plots the time-mean transports.
    """
    cart_out_nc = cart_out + '/exps_clim/'

    # ── Read atmospheric data ─────────────────────────────────────────────────
    map_all = read_output_map(exps, user=user, read_again=[], cart_exp=cart_exp, cart_out=cart_out_nc,
                              atm_only=atm_only, atmvars=atmvars, ocevars=ocevars,
                              year_clim=year_clim, density=density)
    atm_dataset = map_all['atm_map']
    atm_dataset = {co: atm_dataset[co] for co in atm_dataset if atm_dataset[co] is not None}
    atm_dataset = create_ds_exp(atm_dataset)

    # ── Build flux fields ─────────────────────────────────────────────────────
    ds       = atm_dataset.sel(exp=exps)
    toa_net  = ds['rsdt'] - ds['rsut'] - ds['rlut']
    surf_net = -(ds['hfss'] + ds['hfls'] - ds['rsns'] - ds['rlns'])
    atm_div  = toa_net - surf_net

    # ── Integration function ──────────────────────────────────────────────────
    a       = 6378000
    lat     = toa_net.lat
    lat_rad = np.deg2rad(lat)
    dlat    = xr.DataArray(np.gradient(lat_rad), dims=['lat'], coords={'lat': lat})
    coslat  = np.cos(lat_rad)

    def integrate_transport(field):
        zm        = field.mean(dim='lon')
        zm_mean   = (zm * coslat).sum(dim='lat') / coslat.sum()
        zm_corr   = zm - zm_mean
        integrand = zm_corr * 2 * np.pi * a**2 * coslat * dlat
        transport = integrand.cumsum(dim='lat') / 1e15
        transport.attrs['units'] = 'PW'
        return transport

    # ── Compute implied transports ────────────────────────────────────────────
    tht = integrate_transport(toa_net)
    aht = integrate_transport(atm_div)
    oht = tht - aht

    tht.name = 'THT'; tht.attrs['long_name'] = 'Total meridional heat transport'
    aht.name = 'AHT'; aht.attrs['long_name'] = 'Atmospheric meridional heat transport (implied)'
    oht.name = 'OHT'; oht.attrs['long_name'] = 'Ocean meridional heat transport (implied)'

    # ── Read explicit ocean heat transport from hfbasin ───────────────────────
    exp_name = exps[0]
    user_str = user if user is not None else 'itcv'
    hfbasin_path = cart_exp.format(user_str) + f'{exp_name}/output/nemo/*diaptr2d*.nc'

    try:
        ds_hfbasin  = xr.open_mfdataset(hfbasin_path, combine='by_coords')

        # hfbasin is typically (time, basin, lat) — select global basin (basin=0)
        # basin indices: 0=global, 1=Atlantic, 2=Indo-Pacific (check your model convention)
        oht_explicit = ds_hfbasin['hfbasin'].isel(basin=0)   # global ocean, (time, lat)
        # rename time dim to year if needed

        oht_explicit = oht_explicit.groupby('time_counter.year').mean().compute()
        
        # convert to PW if in W (hfbasin is usually already in W)
        oht_explicit = oht_explicit
        oht_explicit.name = 'OHT_explicit'
        oht_explicit.attrs['long_name'] = 'Ocean meridional heat transport (explicit, hfbasin global)'
        oht_explicit.attrs['units']     = 'PW'

        # also save Atlantic basin if available
        oht_atl = ds_hfbasin['hfbasin'].isel(basin=1) 
        oht_atl = oht_atl.groupby('time_counter.year').mean().compute()

        oht_atl.name = 'OHT_atl'
        oht_atl.attrs['long_name'] = 'Ocean meridional heat transport (explicit, hfbasin Atlantic)'
        oht_atl.attrs['units']     = 'PW'

        has_hfbasin = True
        print(f"Read hfbasin → {hfbasin_path}")

    except Exception as e:
        print(f"Warning: could not read hfbasin: {e}")
        has_hfbasin = False

    # ── Save to netCDF ────────────────────────────────────────────────────────
    ds_out = xr.Dataset({'THT': tht, 'AHT': aht, 'OHT': oht})
    if has_hfbasin:
        ds_out['OHT_explicit'] = oht_explicit
        ds_out['OHT_atl']      = oht_atl

    ds_out.attrs['description'] = f'Meridional heat transport timeseries, exp={exp_name}'
    outfile = cart_out_nc + f'heat_transport_{exp_name}.nc'
    ds_out.to_netcdf(outfile)
    print(f"Saved heat transports → {outfile}")

    # ── Plot time mean ────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    ax = axes[0]
    ax.plot(lat, tht.mean(dim='year')[0], 'k-',  linewidth=2,   label='THT (implied)')
    ax.plot(lat, aht.mean(dim='year')[0], 'r-',  linewidth=1.5, label='AHT (implied)')
    ax.plot(lat, oht.mean(dim='year')[0], 'b-',  linewidth=1.5, label='OHT (implied)')
    if has_hfbasin:
        ax.plot(oht_explicit.nav_lat, oht_explicit.mean(dim='year'), 'b--', linewidth=1.5, label='OHT (explicit)')
        ax.plot(oht_atl.nav_lat,      oht_atl.mean(dim='year'),      'g--', linewidth=1.5, label='OHT Atlantic (explicit)')
    ax.axhline(0, color='k', linewidth=0.5, linestyle='--')
    ax.axvline(0, color='k', linewidth=0.5, linestyle='--')
    ax.set_xlabel('Latitude')
    ax.set_ylabel('Heat transport (PW)')
    ax.set_title(f'Meridional heat transport — {exp_name}')
    ax.legend()

    # ax = axes[1]
    # if has_hfbasin:
    #     diff = oht.mean(dim='year') - oht_explicit.mean(dim='year').interp(y=lat)
    #     ax.plot(lat, diff, 'b-', label='OHT implied - explicit')
    #     ax.axhline(0, color='k', linewidth=0.5, linestyle='--')
    #     ax.set_xlabel('Latitude')
    #     ax.set_ylabel('Difference (PW)')
    #     ax.set_title('OHT residual (implied - explicit)')
    #     ax.legend()
    # else:
    #     ax.plot(lat, toa_net.mean(dim=['lon', 'year']),  label='TOA net',        color='orange')
    #     ax.plot(lat, surf_net.mean(dim=['lon', 'year']), label='Surface net',    color='teal')
    #     ax.plot(lat, atm_div.mean(dim=['lon', 'year']),  label='Atm divergence', color='purple')
    #     ax.axhline(0, color='k', linewidth=0.5, linestyle='--')
    #     ax.set_xlabel('Latitude')
    #     ax.set_ylabel('W m$^{-2}$')
    #     ax.set_title('Zonal mean energy fluxes')
    #     ax.legend()

    plt.tight_layout()
    plt.show()

    return ds_out

def compute_trend(vars, exps, dataset):
    
    for var in vars:
        for exp in exps:
            var_pi = dataset[var].sel(exp=exp)
            mask = ~np.isnan(var_pi)
            print(f'-------- {exp} --------')
        
            # test full length trend
            trend = stats.linregress(np.arange(0,np.shape(var_pi[mask])[0]), var_pi[mask])[0]
            print(f'{var} trend for {exp}: {trend}')
            print(f'{var} change in {np.shape(var_pi[mask])[0]} years: {trend*np.shape(var_pi[mask])[0]}\n')

            # test 500 years trend
            trend = stats.linregress(np.arange(0,499), var_pi[mask][:499])[0]
            print(f'{var} trend for {exp} for first 500 ys: {trend}')
            print(f'{var} change in 500 years: {trend*500}\n')

            # test 300 years trend
            trend = stats.linregress(np.arange(0,300), var_pi[mask][-300:])[0]
            print(f'{var} trend for {exp} for last 300 ys: {trend}')
            print(f'{var} change in 300 years: {trend*300}\n')

            # test trend in 300 ys before 500
            trend = stats.linregress(np.arange(0,299), var_pi[mask][200:499])[0]
            print(f'{var} trend for {exp} for 200-500 ys: {trend}')
            print(f'{var} change in 300 years: {trend*300}\n')

def check_pi_state(clim_all, exps):
    """
    Compute trends for some variables to check experiment evolution.
    """
    vars = ['tas']
    dataset = clim_all[f'atm_mean']
    dataset = {co: dataset[co] for co in dataset if dataset[co] is not None}
    dataset = create_ds_exp(dataset)
    compute_trend(vars, exps, dataset)

    vars = ['tos', 'heatc']
    dataset = clim_all[f'oce_mean']
    dataset = {co: dataset[co] for co in dataset if dataset[co] is not None}
    dataset = create_ds_exp(dataset)
    compute_trend(vars, exps, dataset)


# ============================================================
################################################ MAIN FUNCTION ###########################

def compare_multi_exps(exps, user = None, read_again = [], cart_exp = '/ec/res4/scratch/{}/ece4/', cart_out = './output/', imbalance = 0., ref_exp = None, atm_only = False, atmvars = 'rsut rlut rsdt tas pr'.split(), ocevars = 'tos heatc qt_oce sos mldr10_1'.split(), icevars = 'siconc sivolu sithic'.split(), year_clim = None, plot_diffref=False, plot_param=False, param_map={}, skip_first_year=False, exp_type = 'PD', density=False, colors=None, rolling = None, file_lists = None, plot_zonal_vars = [], ongoing = []):
    """
    Runs all multi-exps diagnostics.

    exps: list of experiments to consider
    cart_exp: base dir for experiments (defaults as $SCRATCH on hpc2020)
    user: to set experiment dir using cart_exp template. If a list, specifies a different user for every exp
    read_again: list of exps to read again. If set, overwrites existing clims for exp to update them (useful if sims are still running)
    """
    if cart_out is None:
        raise ValueError('cart_out not specified!')
    
    if not os.path.exists(cart_out): os.mkdir(cart_out)

    cart_out_nc = cart_out + '/exps_clim/'
    cart_out_figs = cart_out + f"/check_{'-'.join(exps)}/"

    if not os.path.exists(cart_out_nc): os.mkdir(cart_out_nc)
    if not os.path.exists(cart_out_figs): os.mkdir(cart_out_figs)

    ### read outputs for all exps
    clim_all = read_output(exps, user = user, read_again = read_again, cart_exp = cart_exp, cart_out = cart_out_nc, atm_only = atm_only, atmvars = atmvars, ocevars = ocevars, icevars = icevars, year_clim = year_clim, density=density, file_lists = file_lists, ongoing = ongoing)

    coupled = False
    if 'amoc_ts' in clim_all: coupled = True

    #check_pi_state(clim_all, exps)
    allfigs = []
    ### Gregory and amoc gregory
    fig_greg = plot_greg(clim_all['atm_mean'], exps, imbalance = imbalance, ylim = None, cart_out = cart_out_figs, exp_type = exp_type, year_clim = year_clim, colors=colors)
    allfigs = [fig_greg]

    if coupled:
        if clim_all['amoc_ts'] is not None:
            fig_amoc_greg = plot_amoc_vs_gtas(clim_all, exps, lw = 0.25, cart_out = cart_out_figs, exp_type = exp_type, year_clim = year_clim, colors=colors)
            allfigs.append(fig_amoc_greg)

            fig_amoc_all = plot_amoc_2d_all(clim_all['amoc_mean'], exps, cart_out = cart_out_figs)
            allfigs.append(fig_amoc_all)

            fig_amoc_ts = plot_var_ts(clim_all, 'amoc', 'amoc', cart_out = cart_out_figs, rolling=rolling, colors=colors)
            allfigs.append(fig_amoc_ts)

    fig_tas2 = plot_var_ts(clim_all, 'atm', 'tas', cart_out = cart_out_figs, rolling=rolling, colors=colors)
    #fig_tas_map  = plot_var_map(clim_all, 'atm', 'tas', ref_exp=ref_exp,cart_out = cart_out_figs, clevels=np.arange(-5,6,1))
    #fig_toa_map  = plot_toa_map(clim_all, 'atm', 'tas', ref_exp=ref_exp,cart_out = cart_out_figs, clevels=np.arange(-5,6,1))

    #fig_ice_map  = plot_var_map(clim_all, 'ice', 'siconc', ref_exp=ref_exp,cart_out = cart_out_figs, clevels=np.arange(-5,6,1))

    #fig_tos = plot_var_ts(clim_all, 'oce', 'tos', cart_out = cart_out_figs, rolling=rolling, colors=colors)
    ##### CAN ADD NEW DIAGS HERE
    if coupled:
        #fig_n2 = plot_var_ts_3d(clim_all, 'rho', 'Nsquared', cart_out = cart_out_figs, rolling=rolling, colors=colors)
        fig_toa = plot_var_ts(clim_all, 'atm', 'toa_net', cart_out = cart_out_figs, rolling=rolling, colors=colors)
        #fig_tas2 = plot_var_ts(clim_all, 'atm', 'tas', cart_out = cart_out_figs, rolling=rolling, colors=colors)
        fig_tos = plot_var_ts(clim_all, 'oce', 'tos', cart_out = cart_out_figs, rolling=rolling, colors=colors)
        #fig_heatc = plot_var_ts(clim_all, 'oce', 'heatc', cart_out = cart_out_figs, rolling=rolling, colors=colors)
        # fig_qtoce = plot_var_ts(clim_all, 'oce', 'qt_oce', cart_out = cart_out_figs, rolling=rolling, colors=colors)
        # fig_enebal = plot_var_ts(clim_all, 'oce', 'enebal', cart_out = cart_out_figs, rolling=rolling, colors=colors)
        # fig_siv =plot_var_ts(clim_all, 'ice', 'sivolu_N', cart_out = cart_out_figs, rolling=rolling, colors=colors)
        #fig_sic = plot_var_ts(clim_all, 'ice', 'siconc_N', cart_out = cart_out_figs, rolling=rolling, colors=colors)
        # fig_siv2 = plot_var_ts(clim_all, 'ice', 'sivolu_S', cart_out = cart_out_figs, rolling=rolling, colors=colors)
        #fig_sic2 = plot_var_ts(clim_all, 'ice', 'siconc_S', cart_out = cart_out_figs, rolling=rolling, colors=colors)
        # allfigs += [fig_tos, fig_heatc, fig_qtoce, fig_enebal, fig_siv, fig_sic, fig_siv2, fig_sic2]
        if density:
            #fig_rho = plot_var_ts_3d(clim_all, 'rho', 'density', cart_out = cart_out_figs, rolling=rolling)
            #fig_den = plot_var_profile(clim_all, 'rho', 'density',  cart_out = cart_out_figs, colors=colors)
            fig_n2 = plot_var_profile(clim_all, 'rho', 'Nsquared', ref_exp=ref_exp, vcoord='depth_mid', cart_out = cart_out_figs, colors=colors)
            #fig_n2so = plot_var_region(clim_all, 'rho', 'Nsquared',[-60,-30],ref_exp=ref_exp, vcoord='depth_mid', cart_out = cart_out_figs, colors=colors,cart_exp = cart_exp)
            #fig_n2zonal  = plot_zonal_profile(clim_all, 'rho', 'Nsquared', ref_exp=ref_exp, vcoord='depth_mid', cart_out = cart_out_figs, colors=colors)
            #fig_siconc_map  = plot_var_map(clim_all, 'ice', 'siconc', ref_exp=ref_exp,cart_out = cart_out_figs, clevels=np.arange(-0.5,0.6,0.1))
            #fig_tos_map  = plot_var_map(clim_all, 'atm', 'tas', ref_exp=ref_exp,cart_out = cart_out_figs, clevels=np.arange(-1,1.1,0.1))
            #fig_cre = plot_cre_zonal_map(clim_all, exps, ref_exp=ref_exp, cart_out = cart_out_figs)
            
            allfigs += [fig_n2]
            
    fig_tas = plot_zonal_tas_vs_ref(clim_all['atm_clim'], exps = exps, ref_exp = ref_exp, cart_out = cart_out_figs, colors=colors)
    allfigs.append(fig_tas)
    
    for var in atmvars:
        if var not in 'rsut rlut rsdt tas'.split():
            fig = plot_var_ts(clim_all, 'atm', var, cart_out = cart_out_figs, rolling=rolling, colors=colors)
            allfigs.append(fig)

        if var in plot_zonal_vars:
            fig = plot_zonal_var(clim_all['atm_clim'], exps = exps, var = var, ref_exp = ref_exp, colors=colors)
            allfigs.append(fig)

    ###### CAN ADD NEW DIAGS HERE
    if coupled:
        for var in ocevars:
            fig = plot_var_ts(clim_all, 'oce', var, cart_out = cart_out_figs, rolling=rolling, colors=colors)
            allfigs.append(fig)

        fig_enebal = plot_var_ts(clim_all, 'oce', 'enebal', cart_out = cart_out_figs, rolling=rolling, colors=colors)
        allfigs.append(fig_enebal)
        
        for var in icevars:
            for emi in ['N', 'S']:
                fig = plot_var_ts(clim_all, 'ice', var+f'_{emi}', cart_out = cart_out_figs, rolling=rolling, colors=colors)
                allfigs.append(fig)


    # --- Optional diagnostics for tuning experiments
    if plot_diffref:
        figs_diffref = plot_zonal_fluxes_vs_ref(
            clim_all['atm_clim'], exps=exps, ref_exp=ref_exp, cart_out=cart_out_figs
        )
        allfigs += figs_diffref

    if plot_param:
        if 'atm_clim' not in clim_all:
            raise KeyError("Expected 'atm_clim' in clim_all, but not found.")
        if skip_first_year:
            for exp, ds in clim_all['atm_clim'].items():
                if ds is not None and 'year' in ds.coords:
                    clim_all['atm_clim'][exp] = ds.isel(year=slice(1, None))

        figs_param = plot_zonal_fluxes_by_param(
            atm_clim=clim_all['atm_clim'],
            ref_exp=ref_exp,
            param_map=param_map,
            cart_out=cart_out_figs,
            plot_anomalies=True,
            weighted=False
        )
        allfigs += figs_param

    print(f'Done! Check results in {cart_out_figs}')

    return clim_all, allfigs


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main(config_path = None):
    if config_path is None:
        # Set up command line argument parser
        parser = argparse.ArgumentParser(description='Load configuration from YAML file')
        parser.add_argument('config', type=str, nargs='?', default='config.yml', help='Path to YAML configuration file')

        args = parser.parse_args()

        config_path = args.config
    
    # Load and parse configuration
    config = load_config(config_path)

    exps = config.get('exps', [])
    user = config.get('user', os.getenv('USER'))
    read_again = config.get('read_again', [])
    cart_exp = config.get('cart_exp', '/ec/res4/scratch/{}/ece4/')
    cart_out = config.get('cart_out')
    imbalance = config.get('imbalance')
    ref_exp = config.get('ref_exp')
    plot_param = config.get('plot_param', False)
    plot_diffref = config.get('plot_diffref', False)
    param_map = config.get('param_map', {})
    skip_first_year = config.get('skip_first_year', False)
    

    if user is None:
        user = os.getenv('USER')
    
    # Example: Print loaded configuration
    print(f"Experiments: {exps}")
    print(f"User: {user}")
    print(f"Read again: {read_again}")
    print(f"Cart exp: {cart_exp}")
    print(f"Cart out: {cart_out}")

    clim_all, figs = compare_multi_exps(exps, user = user, read_again = read_again, cart_exp = cart_exp, cart_out = cart_out, imbalance = imbalance, ref_exp = ref_exp, plot_param=plot_param, plot_diffref=plot_diffref, param_map=param_map,skip_first_year=skip_first_year)

    return clim_all, figs
    

# Main execution
if __name__ == '__main__':
    main()