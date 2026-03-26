import xarray as xr
from matplotlib import pyplot as plt
import numpy as np
import os
import xarray_regrid
import pandas as pd
import xmca
from xmca.array import MCA  # numpy
from xmca.xarray import xMCA  # numpy

import matplotlib.cm as cm
from matplotlib.patches import Patch
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import matplotlib.colors as mcolors 
import glob
import cmocean as cmo
from scipy import stats
import cartopy.crs as ccrs
import matplotlib.gridspec as gridspec # GRIDSPEC !

import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
#import xesmf as xe

import yaml
import argparse
from pathlib import Path

###########################################################################################################

datadir = '../data/'
cart_out = './output/'
cart_exp = '/ec/res4/scratch/{}/ece4/'

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
    colors = [cmap(i / (n - 1)) for i in range(n)]
    return colors


def add_diahsb_init_to_restart(rest_file, rest_file_new = None):
    """
    Adds missing fields to a restart (produced before 4.1.2) and creates a new restart (compatible with 4.1.2), filling missing variables with zeros.
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
    for var in ['surf_ini', 'ssh_ini']:
        rest_oce[var] = xr.DataArray(
        np.zeros((nt, ny, nx)),
        dims=['time_counter', 'y', 'x'])

    # Add 3D variables (time_counter, nav_lev, y, x)
    for var in ['e3t_ini', 'tmask_ini', 'hc_loc_ini', 'sc_loc_ini']:
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
    gh_flux = get_ghflux(exp, user)

    heat_trend = ds_time_mean['heatc'].diff('year')/year_sec/tot_area
    ds_time_mean['enebal'] = heat_trend - ds_time_mean.qt_oce - gh_flux # source of energy in the ocean

    if compute:
        return ds_time_mean.compute()
    else:
        return ds_time_mean


def get_vmask_nemo(exp, user, cart_exp = cart_exp, v_grid = 'deptht'):
    #ocean areas

    if(v_grid == 'deptht'):
        fileoce = xr.load_dataset(cart_exp.format(user) + f'/{exp}/output/nemo/' + f'{exp}_oce_1m_T_1850-1850.nc')
        thetao = fileoce.thetao[0]
        vmask = thetao/thetao
    else:
        fileoce = xr.load_dataset('../density/density_fields/' + f'{exp}/{exp}_1850_density.nc')
        #fileoce = fileoce.rename({f'x_grid_T': 'x', f'y_grid_T': 'y'})
        density = fileoce.Nsquared[0]
        vmask = density/density

    return vmask


def global_mean_oce_3d(ds, exp, user, vars, cart_exp = cart_exp, compute = True, depth_mean=False,grid = 'T', singlelevel = False, lev = 0):
    
    area = get_areas_nemo(exp, user, cart_exp = cart_exp, grid = grid)
    
    #ds = ds.rename({f'x_grid_{grid}': 'x', f'y_grid_{grid}': 'y'}) # for 4.1.0

    ds_time_mean = ds[vars].copy()

    if(singlelevel):
        for var in ds.data_vars:
            if var in vars:
                if(var == 'Nsquared'):
                    v_grid = 'depth_mid'
                else:
                    v_grid = 'deptht'
                v_mask = get_vmask_nemo(exp, user, cart_exp = cart_exp, v_grid = v_grid)
                v_area = (v_mask*area).sum(axis=(1,2))
                ds_time_mean[var] = (ds[var]*area*v_mask).sum(['x', 'y']).compute()
                ds_time_mean[var] = ds_time_mean[var]/v_area[lev].values                
    else:
        for var in ds.data_vars:
            if var in vars:
                if(var == 'Nsquared'):
                    v_grid = 'depth_mid'
                else:
                    v_grid = 'deptht'
                v_mask = get_vmask_nemo(exp, user, cart_exp = cart_exp, v_grid = v_grid)
                v_area = (v_mask*area).sum(axis=(1,2))                
                ds_time_mean[var] = (ds[var]*area*v_mask).sum(['x', 'y']).compute()
                ds_time_mean[var] = (ds_time_mean[var]/v_area.values).compute()

                print(ds_time_mean)
                if depth_mean:
                    # get layer thicknesses as weights
                    e3t = ds_time_mean['e3t']  # shape: (lev, y, x)
                    # weighted sum over depth: weight each level by its thickness * mask
                    thick_weights = (e3t * v_mask).sum(axis=(1, 2))   # total weighted thickness per level
                    total_depth = thick_weights.sum(axis=0)            # scalar: total ocean depth
                    ds_time_mean[var] = (ds_time_mean[var] * thick_weights.values).sum(dim=v_grid) / total_depth.values

    
    return ds_time_mean
    
def global_mean_oce_3d_region(ds, exp, user, vars, cart_exp = cart_exp, lats=None, compute = True, grid = 'T', singlelevel = False, lev = 0):
    
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
                v_mask = get_vmask_nemo(exp, user, cart_exp = cart_exp, v_grid = v_grid)
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
                area_reg = (area*mask_reg).sel(x=slice(lats[0], lats[1]))
                v_mask = get_vmask_nemo(exp, user, cart_exp = cart_exp, v_grid = v_grid)
                v_mask = v_mask.sel(x=slice(lats[0], lats[1]))
                v_area = (v_mask*area_reg).sum(axis=(1,2))
                ds_time_mean[var] = (ds[var].sel(x=slice(lats[0],lats[1]))*area_reg*v_mask).sum(['x', 'y']).compute()
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
        ds_time_mean[var + '_N'] = ds_time_mean[var + '_N']/ds_time_mean['siconc_N']
        ds_time_mean[var + '_S'] = ds_time_mean[var + '_S']/ds_time_mean['siconc_S']

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
    else:
        atmclim = ds.sel(year = slice(year_clim[0], year_clim[1])).mean('year')
    atmmean = global_mean(ds, compute = True)

    if cart_out is not None:
        atmclim.to_netcdf(cart_out + f'clim_tuning_{exp}.nc')
        atmmean.to_netcdf(cart_out + f'mean_tuning_{exp}.nc')

    return atmclim, atmmean

def compute_atm_map(ds, exp, cart_out = cart_out, atmvars = 'rsut rlut rsdt tas pr'.split(), year_clim = None):
    ds = ds.rename({'time_counter': 'time'})
    ds = ds[atmvars].groupby('time.year').mean().compute()

    if cart_out is not None:
        ds.to_netcdf(cart_out + f'map_tuning_{exp}.nc')

    return ds

def compute_oce_map(ds, exp, user, cart_exp = cart_exp, cart_out = cart_out, ocevars = 'tos heatc qt_oce sos'.split(), year_clim = None, grid = 'T'):
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
    # print(ds.data_vars)
    ds = ds[ocevars].groupby('time.year').mean()
    if f'x_grid_{grid}_inner' in ds.dims:
        ds = ds.rename({f'x_grid_{grid}_inner': 'x', f'y_grid_{grid}_inner': 'y'})
    if f'x_grid_{grid}' in ds.dims:
        ds = ds.rename({f'x_grid_{grid}': 'x', f'y_grid_{grid}': 'y'})

    if year_clim is None:
        print('Using last 20 years for climatology')
        oceclim = ds.isel(year = slice(-20, None)).mean('year').compute()
    else:
        oceclim = ds.sel(year = slice(year_clim[0], year_clim[1])).mean('year').compute()

    ocemean = global_mean_oce_2d(ds, exp, user, cart_exp, compute = True)
    
    if cart_out is not None:
        oceclim.to_netcdf(cart_out + f'clim_oce_tuning_{exp}.nc')
        ocemean.to_netcdf(cart_out + f'mean_oce_tuning_{exp}.nc')

    return oceclim, ocemean


def compute_ice_clim(ds, exp, user, cart_exp = cart_exp, cart_out = cart_out, icevars = 'sithic sivolu siconc'.split(), year_clim = None):
    ds = ds.rename({'time_counter': 'time'})
    ds = ds[icevars].groupby('time.year').mean()

    if year_clim is None:
        print('Using last 20 years for climatology')
        iceclim = ds.isel(year = slice(-20, None)).mean('year').compute()
    else:
        iceclim = ds.sel(year = slice(year_clim[0], year_clim[1])).mean('year').compute()

    icemean = global_mean_ice(ds, exp, user, cart_exp, compute = True)

    if cart_out is not None:
        iceclim.to_netcdf(cart_out + f'clim_ice_tuning_{exp}.nc')
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
    else:
        amoc_mean = ds.sel(year = slice(year_clim[0], year_clim[1])).mean('year')
    amoc_mean = amoc_mean.squeeze()
    
    amoc_mean = amoc_mean.compute()
    amoc_ts = amoc_ts.compute()

    if cart_out is not None:
        amoc_mean.to_netcdf(cart_out + f'amoc_2d_tuning_{exp}.nc')
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

def compute_cre_clim(ds, exp, cart_out = cart_out, atmvars = 'rlnt rlntcs rsnt rsntcs'.split(), year_clim = None):
    ds = ds.rename({'time_counter': 'time'})
    ds = ds[atmvars].groupby('time.year').mean().compute()

    if year_clim is None:
        print('Using last 20 years for climatology')
        atmclim = ds.isel(year = slice(-20, None)).mean('year')
    else:
        atmclim = ds.sel(year = slice(year_clim[0], year_clim[1])).mean('year')
    
    if cart_out is not None:
        atmclim.to_netcdf(cart_out + f'clim_cre_tuning_{exp}.nc')

    return atmclim

def calc_amoc_ts(data, ax = None, exp_name = 'exp', depth_min = 500., depth_max = 2000., lat_min = 38, lat_max = 50, ylim = (5, 20), plot = False, basin = 2):

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

def file_list(exp, user, cart_exp = '/ec/res4/scratch/{}/ece4/', remove_last_year = False, coupled = True, density= False):
    cart = f'{cart_exp.format(user)}/{exp}/output/oifs_remap/'
    filz_exp = cart + f'{exp}_atm_cmip6_1m_*.nc'

    cart = f'{cart_exp.format(user)}/{exp}/output/nemo/'
    filz_amoc = cart + f'{exp}_oce_1m_diaptr3d_*.nc'
    filz_nemo = cart + f'{exp}_oce_1m_T_*.nc'
    filz_ice = cart + f'{exp}_ice_1m_*.nc'

    # ftv3_oce_1m_diaptr2d_1991-1991.nc -> hf_basin

    if remove_last_year:
        # Still running, remove last year
        fils = glob.glob(filz_exp)
        fils.sort()
        filz_exp = fils[:-1]

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

        return    filz_exp, filz_nemo, filz_amoc, filz_ice, filz_rho

    else:
        return filz_exp, filz_nemo, filz_amoc, filz_ice

def read_output_map(exps, user = None, read_again = [], cart_exp = cart_exp, cart_out = cart_out, atmvars = 'rsut rlut rsdt tas rlnt rlntcs rsnt rsntcs'.split(), ocevars = 'tos qt_oce'.split(), icevars='siconc'.split(), atm_only = False, year_clim = None, density=False, density_only=False):
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

    filz_exp = dict()
    filz_nemo = dict()
    filz_ice = dict()

    for exp, us in zip(exps, user):
            filz_exp[exp], filz_nemo[exp], _, filz_ice[exp] = file_list(exp, us, cart_exp = cart_exp)
        
    atmmap_exp = dict()
    ocemap_exp = dict()
    icemap_exp = dict()

    if(atm_only):
        coupled = False
    else:
        coupled = True

    for exp, us in zip(exps, user):
        print(exp)
        if os.path.exists(cart_out + f'map_ice_tuning_{exp}.nc') and exp not in read_again:
            print('Already computed, reading clim..')
            existing_atm = xr.load_dataset(cart_out + f'map_tuning_{exp}.nc')

            existing_vars = list(existing_atm.data_vars)
            
            missing_vars = [v for v in atmvars if v not in existing_vars]
            
            if len(missing_vars) == 0:
                atmmap_exp[exp] = existing_atm
            
            else:
                print(f'Computing missing ATM vars: {missing_vars}')
                ds = xr.open_mfdataset(filz_exp[exp],use_cftime=True,chunks={'time_counter': 240})

                new_atm = compute_atm_map(ds,exp,cart_out=cart_out,atmvars=missing_vars,year_clim=year_clim)

                # merge old + new
                updated_atm = xr.merge([existing_atm, new_atm])
                updated_atm.to_netcdf(cart_out + f'map_tuning_{exp}.nc', mode='w')

                atmmap_exp[exp] = updated_atm

            if coupled:
                if os.path.exists(cart_out + f'map_oce_tuning_{exp}.nc'):
                    ocemap_exp[exp] = xr.load_dataset(cart_out + f'map_oce_tuning_{exp}.nc')
                    icemap_exp[exp] = xr.load_dataset(cart_out+f'map_ice_tuning_{exp}.nc')
                
        else:
            print('Computing clim...')

            if os.path.exists(cart_out + f'map_tuning_{exp}.nc'):

                atmmap_exp[exp] = xr.load_dataset(cart_out + f'map_tuning_{exp}.nc', use_cftime=True, chunks = {'time_counter': 240})
                ocemap_exp[exp] = xr.load_dataset(cart_out + f'map_oce_tuning_{exp}.nc', use_cftime=True, chunks = {'time_counter': 240})

                ds = xr.open_mfdataset(filz_ice[exp], use_cftime=True, chunks = {'time_counter': 240})
                icemap_exp[exp] = compute_ice_map(ds, exp, us, cart_exp = cart_exp, cart_out = cart_out, icevars = icevars, year_clim = year_clim)

            else:
                ds = xr.open_mfdataset(filz_exp[exp], use_cftime=True, chunks = {'time_counter': 240})

                # ATM CLIM
                atmmap_exp[exp] = compute_atm_map(ds, exp, cart_out = cart_out, atmvars = atmvars, year_clim = year_clim)

                if coupled:
                    # OCE CLIM
                    ds = xr.open_mfdataset(filz_nemo[exp], use_cftime=True, chunks = {'time_counter': 240})
                    ocemap_exp[exp] = compute_oce_map(ds, exp, us, cart_exp = cart_exp, cart_out = cart_out, ocevars = ocevars, year_clim = year_clim)

                    ds = xr.open_mfdataset(filz_ice[exp], use_cftime=True, chunks = {'time_counter': 240})
                    icemap_exp[exp] = compute_ice_map(ds, exp, us, cart_exp = cart_exp, cart_out = cart_out, icevars = icevars, year_clim = year_clim)

                
    clim_all = dict()
    clim_all['atm_map'] = atmmap_exp
    if coupled:
        clim_all['oce_map'] = ocemap_exp
        clim_all['ice_map'] = icemap_exp
    return clim_all

def read_output(exps, user = None, read_again = [], cart_exp = cart_exp, cart_out = cart_out, atmvars = 'rsut rlut rsdt tas pr'.split(), ocevars = 'tos heatc qt_oce sos hfds'.split(), icevars = 'siconc sivolu sithic'.split(), atm_only = False, year_clim = None, density=False, density_only=False):
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

    filz_exp = dict()
    filz_amoc = dict()
    filz_rho = dict()
    filz_nemo = dict()
    filz_ice = dict()

    if density or density_only:
        for exp, us in zip(exps, user):
            filz_exp[exp], filz_nemo[exp], filz_amoc[exp], filz_ice[exp], filz_rho[exp] = file_list(exp, us, cart_exp = cart_exp, density=True)
    else:
        for exp, us in zip(exps, user):
            filz_exp[exp], filz_nemo[exp], filz_amoc[exp], filz_ice[exp] = file_list(exp, us, cart_exp = cart_exp)
    
    if density_only:
        clim_all = dict()
        rhomean_exp = dict()
        rhoclim_exp = dict()

        for exp, us in zip(exps, user):
            print(f'{exp} (density-only mode)')

            if (os.path.exists(cart_out + f'clim_rho_tuning_{exp}.nc')
                and os.path.exists(cart_out + f'mean_rho_tuning_{exp}.nc')
                and exp not in read_again):

                print('Reading existing density diagnostics')
                rhoclim_exp[exp] = xr.open_dataset(cart_out + f'clim_rho_tuning_{exp}.nc')
                rhomean_exp[exp] = xr.open_dataset(cart_out + f'mean_rho_tuning_{exp}.nc')
                continue

            print('Computing density diagnostics')

            try:
                ds = xr.open_mfdataset(filz_rho[exp],use_cftime=True, chunks={'time': 20},
                )
            except OSError as err:
                print(err)
                print('Run still ongoing, removing last year')
                _, _, _, _, filz_rho[exp] = file_list(exp,us,cart_exp=cart_exp,remove_last_year=True,density=True)
                ds = xr.open_mfdataset(filz_rho[exp],use_cftime=True,chunks={'time': 20})

            rhomean_exp[exp], rhoclim_exp[exp] = compute_rho_clim(ds,exp,us,cart_exp=cart_exp,cart_out=cart_out,year_clim=year_clim)

        clim_all['rho_mean'] = rhomean_exp
        clim_all['rho_clim'] = rhoclim_exp
        return clim_all
    
    atmmean_exp = dict()
    atmclim_exp = dict()
    creclim_exp = dict()
    oceclim_exp = dict()
    ocemean_exp = dict()
    amoc_mean_exp = dict()
    amoc_ts_exp = dict()
    iceclim_exp = dict()
    icemean_exp = dict()
    rhomean_exp = dict()
    rhoclim_exp = dict()

    for exp, us in zip(exps, user):
        print(exp)
        coupled = False
        if not atm_only: 
            if len(glob.glob(filz_nemo[exp])) > 0 or os.path.exists(cart_out + f'clim_oce_tuning_{exp}.nc') or os.path.exists(cart_out + f'oce_tuning_{exp}.nc'):
                print('coupled')
                coupled = True
            else:
                print(f'NO files matching pattern: {filz_nemo[exp]}. Assuming atm-only')

        if os.path.exists(cart_out + f'clim_tuning_{exp}.nc') and exp not in read_again:
            print('Already computed, reading clim..')
            atmclim_exp[exp] = xr.load_dataset(cart_out + f'clim_tuning_{exp}.nc')
            atmmean_exp[exp] = xr.load_dataset(cart_out + f'mean_tuning_{exp}.nc')

            creclim_exp[exp] = xr.load_dataset(cart_out + f'clim_cre_tuning_{exp}.nc')

            if coupled:
                amoc_ts_exp[exp] = xr.load_dataset(cart_out + f'amoc_ts_tuning_{exp}.nc')
                amoc_mean_exp[exp] = xr.load_dataset(cart_out + f'amoc_2d_tuning_{exp}.nc')

                if 'time_counter' in amoc_ts_exp[exp].dims: # legacy adapt to new structure
                    amoc_ts_exp[exp] = amoc_ts_exp[exp].groupby('time_counter.year').mean()
                if isinstance(amoc_ts_exp[exp], xr.Dataset):
                    amoc_ts_exp[exp] = amoc_ts_exp[exp]['msftyz']

                if os.path.exists(cart_out + f'clim_oce_tuning_{exp}.nc'):
                    oceclim_exp[exp] = xr.load_dataset(cart_out + f'clim_oce_tuning_{exp}.nc')
                    ocemean_exp[exp] = xr.load_dataset(cart_out + f'mean_oce_tuning_{exp}.nc')
                    iceclim_exp[exp] = xr.load_dataset(cart_out + f'clim_ice_tuning_{exp}.nc')
                    icemean_exp[exp] = xr.load_dataset(cart_out + f'mean_ice_tuning_{exp}.nc')
                else: # legacy for old exps
                    oceclim_exp[exp] = xr.load_dataset(cart_out + f'oce_tuning_{exp}.nc')
                    ocemean_exp[exp] = None

                if os.path.exists(cart_out + f'clim_rho_tuning_{exp}.nc'):
                    rhoclim_exp[exp] = xr.open_dataset(cart_out + f'clim_rho_tuning_{exp}.nc')
                    rhomean_exp[exp] = xr.open_dataset(cart_out + f'mean_rho_tuning_{exp}.nc')
                    density=True
        
        elif os.path.exists(cart_out + f'clim_tuning_{exp}.nc') and exp in read_again:
            print('Updating existing diagnostics with new data...')
            
            # Load existing data
            atmclim_old = xr.load_dataset(cart_out + f'clim_tuning_{exp}.nc')
            atmmean_old = xr.load_dataset(cart_out + f'mean_tuning_{exp}.nc')

            creclim_old = xr.load_dataset(cart_out + f'clim_cre_tuning_{exp}.nc')
            
            # Get last year from existing data
            last_year = int(atmmean_old.year[-1].values)
            print(f'Last year in saved data: {last_year}')
            
            # Read only new data starting from last_year + 1
            try:
                ds = xr.open_mfdataset(filz_exp[exp], use_cftime=True, chunks = {'time_counter': 240})
            except OSError as err:
                print(err)
                print('Run still ongoing, removing last year')
                filz_exp[exp], filz_nemo[exp], filz_amoc[exp], filz_ice[exp] = file_list(exp, us, cart_exp = cart_exp, remove_last_year = True, coupled = coupled)
                ds = xr.open_mfdataset(filz_exp[exp], use_cftime=True, chunks = {'time_counter': 240})
                
            ds_new = ds.sel(time_counter = slice(f'{last_year+1}0101', None))

            if len(ds_new.time_counter) == 0:
                print('No new data available, using existing diagnostics')
                atmclim_exp[exp] = atmclim_old
                atmmean_exp[exp] = atmmean_old

                creclim_exp[exp] = creclim_old
            else:
                print(f'Found {len(ds_new.time_counter)} new time steps')
                # Compute diagnostics for new data only
                atmclim_new, atmmean_new = compute_atm_clim(ds_new, exp, cart_out = None, atmvars = atmvars, year_clim = year_clim)

                creclim_new = compute_cre_clim(ds_new, exp, cart_out = None, year_clim = year_clim)
                
                # Concatenate old and new data
                atmclim_exp[exp] = atmclim_new
                atmmean_exp[exp] = xr.concat([atmmean_old, atmmean_new], dim='year')

                creclim_exp[exp] = creclim_new
                
                # Save updated data
                atmclim_exp[exp].to_netcdf(cart_out + f'clim_tuning_{exp}.nc')
                atmmean_exp[exp].to_netcdf(cart_out + f'mean_tuning_{exp}.nc')

                creclim_exp[exp].to_netcdf(cart_out + f'clim_cre_tuning_{exp}.nc')
            
            if os.path.exists(cart_out + f'clim_rho_tuning_{exp}.nc'):
                    rhoclim_exp[exp] = xr.open_dataset(cart_out + f'clim_rho_tuning_{exp}.nc')
                    rhomean_exp[exp] = xr.open_dataset(cart_out + f'mean_rho_tuning_{exp}.nc')
                    density=True
                    
            if coupled:
                # Load existing ocean/ice data
                if os.path.exists(cart_out + f'clim_oce_tuning_{exp}.nc'):
                    oceclim_old = xr.load_dataset(cart_out + f'clim_oce_tuning_{exp}.nc')
                    ocemean_old = xr.load_dataset(cart_out + f'mean_oce_tuning_{exp}.nc')
                    iceclim_old = xr.load_dataset(cart_out + f'clim_ice_tuning_{exp}.nc')
                    icemean_old = xr.load_dataset(cart_out + f'mean_ice_tuning_{exp}.nc')
                    amoc_ts_old = xr.load_dataset(cart_out + f'amoc_ts_tuning_{exp}.nc')
                    amoc_mean_old = xr.load_dataset(cart_out + f'amoc_2d_tuning_{exp}.nc')

                    if 'time_counter' in amoc_ts_old.dims:
                        amoc_ts_old = amoc_ts_old.groupby('time_counter.year').mean()
                    if isinstance(amoc_ts_old, xr.Dataset):
                        amoc_ts_old = amoc_ts_old['msftyz']
                    
                    last_year_oce = int(ocemean_old.year[-1].values)
                    # OCE
                    ds = xr.open_mfdataset(filz_nemo[exp], use_cftime=True, chunks = {'time_counter': 240})
                    ds_new = ds.sel(time_counter = slice(f'{last_year_oce+1}0101', None))
                    
                    if len(ds_new.time_counter) > 0:
                        oceclim_new, ocemean_new = compute_oce_clim(ds_new, exp, us, cart_exp = cart_exp, cart_out = None, ocevars = ocevars, year_clim = year_clim)
                        oceclim_exp[exp] = oceclim_new
                        ocemean_exp[exp] = xr.concat([ocemean_old, ocemean_new], dim='year')
                        oceclim_exp[exp].to_netcdf(cart_out + f'clim_oce_tuning_{exp}.nc')
                        ocemean_exp[exp].to_netcdf(cart_out + f'mean_oce_tuning_{exp}.nc')
                    else:
                        print('No new data available, using existing diagnostics')
                        oceclim_exp[exp] = oceclim_old
                        ocemean_exp[exp] = ocemean_old
                    
                    # ICE
                    last_year_ice = int(icemean_old.year[-1].values)
                    ds = xr.open_mfdataset(filz_ice[exp], use_cftime=True, chunks = {'time_counter': 240})
                    ds_new = ds.sel(time_counter = slice(f'{last_year_ice+1}0101', None))
                    
                    if len(ds_new.time_counter) > 0:
                        iceclim_new, icemean_new = compute_ice_clim(ds_new, exp, us, cart_exp = cart_exp, cart_out = None, icevars = icevars, year_clim = year_clim)
                        iceclim_exp[exp] = iceclim_new
                        icemean_exp[exp] = xr.concat([icemean_old, icemean_new], dim='year')
                        iceclim_exp[exp].to_netcdf(cart_out + f'clim_ice_tuning_{exp}.nc')
                        icemean_exp[exp].to_netcdf(cart_out + f'mean_ice_tuning_{exp}.nc')
                    else:
                        print('No new data available, using existing diagnostics')
                        iceclim_exp[exp] = iceclim_old
                        icemean_exp[exp] = icemean_old
                    
                    # AMOC
                    last_year_amoc = int(amoc_ts_old.year[-1].values)
                    ds = xr.open_mfdataset(filz_amoc[exp], use_cftime=True, chunks = {'time_counter': 240})
                    ds_new = ds.sel(time_counter = slice(f'{last_year_amoc+1}0101', None))
                    
                    if len(ds_new.time_counter) > 0:
                        amoc_mean_new, amoc_ts_new = compute_amoc_clim(ds_new, exp, cart_out = None, year_clim = year_clim)
                        amoc_mean_exp[exp] = amoc_mean_new
                        amoc_ts_exp[exp] = xr.concat([amoc_ts_old, amoc_ts_new], dim='year')
                        amoc_mean_exp[exp].to_netcdf(cart_out + f'amoc_2d_tuning_{exp}.nc')
                        amoc_ts_exp[exp].to_netcdf(cart_out + f'amoc_ts_tuning_{exp}.nc')
                    else:
                        print('No new data available, using existing diagnostics')
                        amoc_mean_exp[exp] = amoc_mean_old
                        amoc_ts_exp[exp] = amoc_ts_old
                else:
                    # Legacy path - recompute everything
                    print('Legacy format detected, recomputing from scratch')
                    ds = xr.open_mfdataset(filz_nemo[exp], use_cftime=True, chunks = {'time_counter': 240})
                    oceclim_exp[exp], ocemean_exp[exp] = compute_oce_clim(ds, exp, us, cart_exp = cart_exp, cart_out = cart_out, ocevars = ocevars, year_clim = year_clim)
                    ds = xr.open_mfdataset(filz_ice[exp], use_cftime=True, chunks = {'time_counter': 240})
                    iceclim_exp[exp], icemean_exp[exp] = compute_ice_clim(ds, exp, us, cart_exp = cart_exp, cart_out = cart_out, icevars = icevars, year_clim = year_clim)
                    ds = xr.open_mfdataset(filz_amoc[exp], use_cftime=True, chunks = {'time_counter': 240})
                    amoc_mean_exp[exp], amoc_ts_exp[exp] = compute_amoc_clim(ds, exp, cart_out = cart_out, year_clim = year_clim)
        
        else:
            print('Computing clim...')

            try:
                ds = xr.open_mfdataset(filz_exp[exp], use_cftime=True, chunks = {'time_counter': 240})
            except OSError as err:
                print(err)
                print('Run still ongoing, removing last year')
                filz_exp[exp], filz_nemo[exp], filz_amoc[exp], filz_ice[exp] = file_list(exp, us, cart_exp = cart_exp, remove_last_year = True, coupled = coupled)

                ds = xr.open_mfdataset(filz_exp[exp], use_cftime=True, chunks = {'time_counter': 240})

            # ATM CLIM
            atmclim_exp[exp], atmmean_exp[exp] = compute_atm_clim(ds, exp, cart_out = cart_out, atmvars = atmvars, year_clim = year_clim)

            creclim_exp[exp] = compute_cre_clim(ds, exp, cart_out = cart_out, year_clim = year_clim)

            if coupled:
                # OCE CLIM
                
                ds = xr.open_mfdataset(filz_nemo[exp], use_cftime=True, chunks = {'time_counter': 240})
                oceclim_exp[exp], ocemean_exp[exp] = compute_oce_clim(ds, exp, us, cart_exp = cart_exp, cart_out = cart_out, ocevars = ocevars, year_clim = year_clim)

                ds = xr.open_mfdataset(filz_ice[exp], use_cftime=True, chunks = {'time_counter': 240})
                iceclim_exp[exp], icemean_exp[exp] = compute_ice_clim(ds, exp, us, cart_exp = cart_exp, cart_out = cart_out, icevars = icevars, year_clim = year_clim)

                ds = xr.open_mfdataset(filz_amoc[exp], use_cftime=True, chunks = {'time_counter': 240})
                amoc_mean_exp[exp], amoc_ts_exp[exp] = compute_amoc_clim(ds, exp, cart_out = cart_out, year_clim = year_clim)
                
                try:
                    ds = xr.open_mfdataset(filz_rho[exp], use_cftime=True, chunks = {'time': 20})
                    rhomean_exp[exp], rhoclim_exp[exp] = compute_rho_clim(ds, exp, us, cart_exp = cart_exp, cart_out = cart_out, year_clim = year_clim)
                    density = True
                except OSError as err:
                        print(err)
                        density = False

    clim_all = dict()
    clim_all['atm_clim'] = atmclim_exp
    clim_all['atm_mean'] = atmmean_exp
    clim_all['cre_clim'] = creclim_exp
    if coupled:
        clim_all['oce_clim'] = oceclim_exp
        clim_all['oce_mean'] = ocemean_exp
        clim_all['ice_clim'] = iceclim_exp
        clim_all['ice_mean'] = icemean_exp
        clim_all['amoc_mean'] = amoc_mean_exp

        # if 'time_counter' in clim_all['amoc_ts']['pal3'].dims:
        #     amoc_ts_exp = amoc_ts_exp.groupby('time_counter.year').mean()
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


def plot_greg(atmmean_exp, exps, cart_out = cart_out, exp_type = 'PI', n_end = 20, imbalance = -0.9, ylim = None, colors = None):
    """
    gregory plot
    """

    fig, ax = plt.subplots(figsize=(12, 8))

    if colors is None:
        colors = get_colors(exps)

    for exp, col in zip(exps, colors):
        ax.plot(atmmean_exp[exp].tas, atmmean_exp[exp].toa_net, label = exp, lw = 0.2, color = col)
        #ax.scatter(atmmean_exp[exp].tas.sel(year = slice(1990, 2000)).mean(), atmmean_exp[exp].toa_net.sel(year = slice(1990, 2000)).mean(), s = 1000, color = 'red', marker = 'o')
        x, y = atmmean_exp[exp].tas.isel(year = slice(-n_end, None)).mean(), atmmean_exp[exp].toa_net.isel(year = slice(-n_end, None)).mean()
        ax.scatter(x, y, s = 1000, color = col, marker = 'o', alpha = 0.5, zorder = 3)
        ax.text(x+0.1, y+0.1, exp, fontsize=12, ha='right', color = col)

    ### plot target shades
    xlim_tot = ax.get_xlim()
    ylim_tot = ax.get_ylim()
    
    #PD
    tas_clim_PD = 287.29
    net_toa_clim_PD = 0.6
    
    #PI
    tas_clim_PI = 286.65
    net_toa_clim_PI = 0.

    if exp_type == 'PI' or exp_type == 'all':
        tas_clim = tas_clim_PI
        net_toa_clim = net_toa_clim_PI
        ax.fill_betweenx(np.arange(ylim_tot[0], ylim_tot[1], 0.1), tas_clim - 0.15, tas_clim + 0.15, color = 'grey', alpha = 0.2, edgecolor = None, zorder = 0)
        ax.fill_between(np.arange(xlim_tot[0], xlim_tot[1], 0.1), net_toa_clim - imbalance - 0.15, net_toa_clim - imbalance + 0.15, color = 'grey', alpha = 0.2, edgecolor = None, zorder = 0)
    
    if exp_type == 'PD' or exp_type == 'all':
        tas_clim = tas_clim_PD
        net_toa_clim = net_toa_clim_PD
        ax.fill_betweenx(np.arange(ylim_tot[0], ylim_tot[1], 0.1), tas_clim - 0.15, tas_clim + 0.15, color = 'burlywood', alpha = 0.2, edgecolor = None, zorder = 0)
        ax.fill_between(np.arange(xlim_tot[0], xlim_tot[1], 0.1), net_toa_clim - imbalance - 0.15, net_toa_clim - imbalance + 0.15, color = 'burlywood', alpha = 0.2, edgecolor = None, zorder = 0)

    
    ax.set_xlabel('GTAS (K)')
    ax.set_ylabel('net TOA (W/m$^2$)')
    plt.legend()

    if ylim is not None:
        ax.set_ylim(ylim)

    fig.savefig(cart_out + f'check_tuning_{'-'.join(exps)}.pdf')
    plt.show()

    return fig

def plot_amoc_vs_gtas(clim_all, exps = None, cart_out = cart_out, exp_type = 'PI', n_end = 20, colors = None, labels = None, colors_legend = None, lw = 0.3, alpha = 0.5, background_color = None):
    fig, ax = plt.subplots(figsize=(12, 8))

    if exps is None:
        exps = clim_all['atm_mean'].keys()

    if colors is None:
        colors = get_colors(exps)

    # print('AAAAAA')
    # print(clim_all['amoc_ts'].keys())

    for exp, col in zip(exps, colors):
        if isinstance(clim_all['amoc_ts'][exp], xr.DataArray):
            y = clim_all['amoc_ts'][exp]
        else:
            y = clim_all['amoc_ts'][exp]['msftyz']

        x = clim_all['atm_mean'][exp]['tas']
        if 'time_counter' in y.dims:
            y = y.groupby('time_counter.year').mean().squeeze()

        ax.plot(x, y, label = exp, lw = lw, color = col)
        
        x, y = x.isel(year = slice(-n_end, None)).mean(), y.isel(year = slice(-n_end, None)).mean()
        ax.scatter(x, y, s = 1000, color = col, marker = 'o', edgecolors = col, alpha = 0.5, zorder = 3)
        ax.text(x+0.1, y+0.1, exp, fontsize=12, ha='right', color = col)

    ax.set_xlabel('GTAS (K)')
    ax.set_ylabel('AMOC max (Sv)')

    xlim_tot = ax.get_xlim()
    ylim_tot = ax.get_ylim()
    
    #PD
    tas_clim_PD = 287.29
    
    #PI
    tas_clim_PI = 286.65

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


    fig.savefig(cart_out + f'check_amoc_vs_gtas_{'-'.join(exps)}.pdf')
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

    fig.savefig(cart_out + f'check_{x_ds.name}_vs_{y_ds.name}_{'-'.join(exps)}.pdf')
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

    ceres_vars = ['toa_lw_all_mon', 'toa_sw_all_mon', 'toa_net_all_mon']#, 'solar_mon']
    okvars = ['rlut', 'rsut', 'toa_net']#, 'rsdt']

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

        fig.savefig(cart_out + f'check_radiation_vs_ceres_{'-'.join(exps)}{add}.pdf')
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

        figname = f'check_radiation_vs_ref_{ref_exp}_{var}_{"-".join(exps)}{add}.pdf'
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
    for exp, ax in zip(exps, axs.flatten()):
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

    for exp, ax in zip(exps, axs.flatten()):
        plot_amoc_2d(amoc_mean[exp], exp=exp, ax = ax)

    plt.tight_layout()
    fig.savefig(cart_out + f'check_amoc_2d_{'-'.join(exps)}.pdf')
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
        ax.set_ylabel(f'zonal temp diff wrt {ref_exp} (K)')
        fig.savefig(cart_out + f'check_zonal_tas_{'-'.join([exp for exp in exps if exp != ref_exp])}_vs_{ref_exp}.pdf')
    else:
        ax.set_ylabel('zonal temp (K)')
        fig.savefig(cart_out + f'check_zonal_tas_{'-'.join(exps)}.pdf')


    return fig

def plot_var_ts(clim_all, domain, vname, exps = None, ref_exp = None, rolling = None, norm_factor = 1., cart_out = cart_out, colors= None):
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

    fig.savefig(cart_out + f'check_ts_{domain}_{vname}_{'-'.join([exp for exp in exps])}.pdf')
    
    return fig


def plot_var_ts_3d(clim_all, domain, vname, exps = None, ref_exp = None, rolling = None, norm_factor = 1., cart_out = cart_out, colors=None):
    """
    Plots timeseries of var "vname" in domain "domain" for all exps.
    Now only for surface level
    Domain is one among: ['atm', 'oce', 'ice']
    """

    if domain not in ['atm', 'oce', 'ice', 'rho']:
        raise ValueError('domain should be one among: atm, oce, ice, rho')
    
    ts_dataset = clim_all[f'{domain}_mean']

    ts_dataset = {co: ts_dataset[co] for co in ts_dataset if ts_dataset[co] is not None}

    fig, ax = plt.subplots(figsize=(12, 8))

    if exps is None: exps = ts_dataset.keys()
    ts_dataset = create_ds_exp(ts_dataset)

    y_ref = None
    if ref_exp is not None:
        y_ref = norm_factor*ts_dataset.sel(exp = ref_exp)[vname]

    if colors is None:
        colors = get_colors(exps)

    for exp, col in zip(exps, colors):
        y = norm_factor*ts_dataset.sel(exp = exp)[vname]
        
        if y_ref is not None: y = y - y_ref

        # fix with averaged mean with level depth!! now only surface level!
        if rolling is not None:
            y[:,0].rolling(year = rolling).mean(axes=0).plot(label = exp, color = col, ax = ax)
        else:
            y[:,0].plot(label = exp, color = col, ax = ax)

        # ax.text(y.year[-1] + 5, np.nanmean(y.values[-30:]), exp, fontsize=12, ha='right', color = col) # not working for some evil reason
    
    ax.set_title('')
    ax.legend()

    fig.savefig(cart_out + f'check_ts_{domain}_{vname}_{'-'.join([exp for exp in exps])}.pdf')
    
    return fig

def plot_var_region(clim_all, domain, vname, lats, vcoord='deptht', exps = None, ref_exp = None, norm_factor = 1., cart_exp=cart_exp, cart_out = cart_out, colors=None):
    """
    Plots vertical profile of var "vname" in domain "domain" for all exps.

    Domain is one among: ['atm', 'oce', 'ice']
    """

    if domain not in ['oce', 'rho']:
        raise ValueError('domain should be one among: oce, rho')
    
    ts_dataset = clim_all[f'{domain}_clim']
    ts_dataset = {co: ts_dataset[co] for co in ts_dataset if ts_dataset[co] is not None}

    fig, ax = plt.subplots(figsize=(8, 8))
    if exps is None: exps = ts_dataset.keys()

    ts_dataset = create_ds_exp(ts_dataset)
    
    if ref_exp is not None:
        y_ref = ts_dataset.sel(exp = ref_exp)#[vname]
        y_ref = global_mean_oce_3d_region(y_ref,ref_exp,'itcv', vname, cart_exp, lats)[vname]

    if colors is None:
        colors = get_colors(exps)
    
    for exp, col in zip(exps, colors):
        y = ts_dataset.sel(exp = exp)#[vname]
        y = global_mean_oce_3d_region(y,exp,'itcv', vname, cart_exp, lats)[vname]

        if (vcoord == 'depth_mid'):
            v_levels = ts_dataset.sel(exp = exp)['density']['deptht']
            levels = (v_levels[1:].values + v_levels[:-1].values)/2
        else:
            levels = ts_dataset.sel(exp = exp)[vname][vcoord]

        if y_ref is not None: y = (y - y_ref)/y_ref

        ax.plot(y, -levels, label=exp, color= col)
        # ax.text(y.year[-1] + 5, np.nanmean(y.values[-30:]), exp, fontsize=12, ha='right', color = col) # not working for some evil reason
    
    ax.set_title('')
    ax.legend()
    ax.set_ylabel('Depth (m)')
    ax.set_xlabel(vname)

    power = 1/2  # o 1/1.5
    fwd = lambda y: np.sign(y) * (abs(y) ** power)
    inv = lambda y: np.sign(y) * (abs(y) ** (1/power))
    ax.set_yscale('function', functions=(fwd, inv))

    fig.savefig(cart_out + f'check_profileregion_{domain}_{vname}_{'-'.join([exp for exp in exps])}.pdf')
    
    return fig

def zonal_mean_irregular_xarray(data,lat, fix=False):
    
    if(fix):
        nlat=90
        lats = np.linspace(-89,89,nlat)
    else:
        nlat = lat.shape[0]
        lats = np.linspace(-90,90, nlat)
    step = (lats[1]-lats[0])/2 + 0.16

    data_zonal = np.zeros([data.shape[0], nlat])

    for i, lat in enumerate(lats):
        #print(lat)
        temp = data.where(((data.nav_lat < lat+step) & (data.nav_lat > lat-step)), np.nan)

        data_zonal[:,i] = np.nanmean(temp, axis=(1,2))

    return data_zonal, lats

def plot_var_profile(clim_all, domain, vname, vcoord='deptht', exps = None, ref_exp = None, norm_factor = 1., cart_out = cart_out, colors=None):
    """
    Plots vertical profile of var "vname" in domain "domain" for all exps.

    Domain is one among: ['atm', 'oce', 'ice']
    """
    
    if domain not in ['oce', 'rho']:
        raise ValueError('domain should be one among: oce, rho')
    
    ts_dataset = clim_all[f'{domain}_mean']
    ts_dataset = {co: ts_dataset[co] for co in ts_dataset if ts_dataset[co] is not None}

    fig, ax = plt.subplots(figsize=(8, 8))
    if exps is None: exps = ts_dataset.keys()

    ts_dataset = create_ds_exp(ts_dataset)

    if ref_exp is not None:
        y_ref = ts_dataset.sel(exp = ref_exp)[vname]
    
    if colors is None:
        colors = get_colors(exps)
    
    for exp, col in zip(exps, colors):
        y = ts_dataset.sel(exp = exp)[vname]
    
        if (vcoord == 'depth_mid'):
            v_levels = ts_dataset.sel(exp = exp)['density']['deptht']
            levels = (v_levels[1:].values + v_levels[:-1].values)/2
        else:
            levels = ts_dataset.sel(exp = exp)[vname][vcoord]

        if ref_exp is not None: y = (y - y_ref) #/y_ref

        ax.plot(y.mean(axis=0), -levels, label=exp, color=col)
        # ax.text(y.year[-1] + 5, np.nanmean(y.values[-30:]), exp, fontsize=12, ha='right', color = col) # not working for some evil reason
    
    ax.set_title('')
    ax.legend()
    ax.set_ylabel('Depth (m)')
    ax.set_xlabel(vname)

    power = 1/2  # o 1/1.5
    fwd = lambda y: np.sign(y) * (abs(y) ** power)
    inv = lambda y: np.sign(y) * (abs(y) ** (1/power))
    ax.set_yscale('function', functions=(fwd, inv))

    #fig.savefig(cart_out + f'check_profile_{domain}_{vname}_{'-'.join([exp for exp in exps])}.pdf')
    
    return fig


def plot_zonal_profile(clim_all, domain, vname, vcoord='deptht', exps = None, ref_exp = None, norm_factor = 1., cart_out = cart_out, colors=None):
    """
    Plots vertical profile of var "vname" in domain "domain" for all exps.

    Domain is one among: ['atm', 'oce', 'ice']
    """
    decades = np.arange(-6, -3)   # 1e-8 → 1e-4

    # 1–5 spacing per decade
    pos_levels = np.sort(np.concatenate([1e0 * 10.0**decades,5e0 * 10.0**decades]))
    pos_levels = pos_levels[pos_levels <= 1e-4]

    neg_levels = -pos_levels[::-1]
    neg_levels = neg_levels[neg_levels >=-1e-4]

    #llevels = np.concatenate((neg_levels, [-5e-9,-1e-9],[0.0],[1e-9,5e-9,], pos_levels))
    llevels = np.concatenate((neg_levels,[ -5e-7,-1e-7 ],[0.0],[1e-7,5e-7],  pos_levels))

    # Keep only within desired max
    N = 18 #22
    center_bin = 8 #10 

    neg_colors = plt.cm.RdBu_r(np.linspace(0, 0.5, center_bin, endpoint=False))
    pos_colors = plt.cm.RdBu_r(np.linspace(0.5, 1, N - center_bin, endpoint=True))

    colors_combined = np.vstack([neg_colors, pos_colors])
    cmap = mcolors.ListedColormap(colors_combined)
    norm = mcolors.BoundaryNorm(llevels, ncolors=N, clip=True)

    if domain not in ['oce', 'rho']:
        raise ValueError('domain should be one among: oce, rho')
    
    ts_dataset = clim_all[f'{domain}_clim']
    ts_dataset = {co: ts_dataset[co] for co in ts_dataset if ts_dataset[co] is not None}
    
    if exps is None: exps = ts_dataset.keys()
    nx = int(np.ceil(np.sqrt(len(exps))))
    ny = int(np.ceil(len(exps)/nx))
    fig, axs = plt.subplots(nx, ny, figsize = (14, 12)) 

    ts_dataset = create_ds_exp(ts_dataset)

    if ref_exp is not None:
        y_ref = ts_dataset.sel(exp = ref_exp)[vname]
        z_ref, lats = zonal_mean_irregular_xarray(y_ref, y_ref.nav_lat)
    
    if colors is None:
        colors = get_colors(exps)
    
    for exp, ax in zip(exps, axs.flatten()):

        if exp != ref_exp:
            y = ts_dataset.sel(exp = exp)[vname]
        
            if (vcoord == 'depth_mid'):
                v_levels = ts_dataset.sel(exp = exp)['density']['deptht']
                levels = (v_levels[1:].values + v_levels[:-1].values)/2
            else:
                levels = ts_dataset.sel(exp = exp)[vname][vcoord]
            
            z, lats = zonal_mean_irregular_xarray(y, y.nav_lat)

            if y_ref is not None: z = (z - z_ref) #/z_ref

            # llevels = np.arange(-1,1.1,0.1)
            # norm = mcolors.BoundaryNorm(llevels,ncolors=plt.colormaps['RdBu_r'].N,clip=True)
            d = ax.contourf(lats, -levels/1000, z, levels=llevels, cmap=cmap, extend='both',norm=norm)
            #ax.set_ylabel('Depth (m)')
            ax.set_title(exp)

            cb2 = plt.colorbar(d, ax=ax, extend='both')
            # cb2.set_ticks([0, 1e-7,1e-6,1e-5, 1e-4, 1e-3])
            # cb2.set_ticklabels(['0','1e-7','1e-6','1e-5','1e-4','1e-3'], fontsize=12)
            # cb2.ax.set_ylabel(r'$N^2$ $(s^{-2}$)', fontsize=12)

    clevels = [0, 1e-7, 2.5e-7, 5e-7, 7.5e-7, 1e-6, 2.5e-6, 5e-6, 7.5e-6, 1e-5, 2.5e-5, 5e-5, 7.5e-5, 1e-4, 2.5e-4, 5e-4, 7.5e-4, 1e-3]
    divnorm = mcolors.BoundaryNorm(clevels,ncolors=plt.colormaps['RdBu_r'].N,clip=True)
    c = axs[0,0].contourf(lats, -levels, z_ref, cmap=cmo.cm.dense, levels=clevels, norm=divnorm)
    cb = plt.colorbar(c, ax=axs[0,0], extend='both')
    cb.set_ticks([0, 1e-7,1e-6,1e-5, 1e-4, 1e-3])
    cb.set_ticklabels(['0','1e-7','1e-6','1e-5','1e-4','1e-3'], fontsize=12)
    cb.ax.set_ylabel(r'$N^2$ $(s^{-2}$)', fontsize=12)
    axs[0,0].set_title(ref_exp)

    #fig.savefig(cart_out + f'check_profile_{domain}_{vname}_{'-'.join([exp for exp in exps])}.pdf')
    
    return fig

def plot_var_map(clim_all, domain, vname, exps=None, ref_exp = None, norm_factor = 1., cart_out = cart_out, clevels=None):
    
    """
    Plots 2d map of var "vname" in domain "domain" for all exps.
    """
    #clim_all = read_output()
    ts_dataset = clim_all[f'{domain}_map'] #_clim
    ts_dataset = {co: ts_dataset[co] for co in ts_dataset if ts_dataset[co] is not None}
    
    if exps is None: exps = ts_dataset.keys()
    nx = int(np.ceil(np.sqrt(len(exps))))
    ny = int(np.ceil(len(exps)/nx))
    fig, axs = plt.subplots(nx, ny, figsize = (14, 10), subplot_kw={'projection': ccrs.PlateCarree()}) 

    ts_dataset = create_ds_exp(ts_dataset)

    if ref_exp is not None:
        y_ref = ts_dataset.sel(exp = ref_exp)[vname]
    
    divnorm = mcolors.BoundaryNorm(clevels, ncolors=plt.colormaps['RdBu_r'].N, clip=True)
    
    for exp, ax in zip(exps, axs.flatten()):

        if exp != ref_exp:
            y = ts_dataset.sel(exp = exp)[vname]
            
            if y_ref is not None: y = (y - y_ref)#/y_ref
            
            d = ax.pcolormesh(ts_dataset.sel(exp = exp).nav_lon, ts_dataset.sel(exp = exp).nav_lat, y.sel(year=slice(1850+55,1850+85)).mean(axis=0),  transform=ccrs.PlateCarree(),cmap='RdBu_r', norm=divnorm)
            gl = ax.gridlines(draw_labels={"bottom": "x", "left": "y"}, color='gray', alpha=0.5)
            gl.xlabel_style = {'size': 10}
            gl.ylabel_style = {'size':10}    
            ax.set_title(exp)
            ax.coastlines()

            cb2 = plt.colorbar(d, ax=ax, extend='both', shrink=0.7)
            # cb2.set_ticks([0, 1e-7,1e-6,1e-5, 1e-4, 1e-3])
            # cb2.set_ticklabels(['0','1e-7','1e-6','1e-5','1e-4','1e-3'], fontsize=12)
            #cb2.ax.set_ylabel(r'(°C)', fontsize=12)


    # clevels = [0, 1e-7, 2.5e-7, 5e-7, 7.5e-7, 1e-6, 2.5e-6, 5e-6, 7.5e-6, 1e-5, 2.5e-5, 5e-5, 7.5e-5, 1e-4, 2.5e-4, 5e-4, 7.5e-4, 1e-3]
    # divnorm = mcolors.BoundaryNorm(clevels,ncolors=plt.colormaps['RdBu_r'].N,clip=True)
    c = axs[0,0].pcolormesh(ts_dataset.sel(exp = ref_exp).nav_lon,ts_dataset.sel(exp = ref_exp).nav_lat, y_ref.sel(year=slice(1850+55,1850+85)).mean(axis=0), cmap=cmo.cm.thermal,transform=ccrs.PlateCarree())#, norm=divnorm)
    cb = plt.colorbar(c, ax=axs[0,0], extend='both', shrink=0.7)
    #cb.set_ticks([0, 1e-7,1e-6,1e-5, 1e-4, 1e-3])
    #cb.set_ticklabels(['0','1e-7','1e-6','1e-5','1e-4','1e-3'], fontsize=12)
    #cb.ax.set_ylabel(r'(°C)', fontsize=12)
    axs[0,0].set_title(ref_exp)
    axs[0,0].coastlines()

    #fig.savefig(cart_out + f'siconc.pdf')
    plt.show()
    return fig

def plot_toa_map(clim_all, domain, vname, exps=None, ref_exp = None, norm_factor = 1., cart_out = cart_out, clevels=None):
    
    """
    Plots 2d map of var "vname" in domain "domain" for all exps.
    """
    #clim_all = read_output()
    ts_dataset = clim_all[f'{domain}_clim']
    ts_dataset = {co: ts_dataset[co] for co in ts_dataset if ts_dataset[co] is not None}
    
    if exps is None: exps = ts_dataset.keys()
    nx = int(np.ceil(np.sqrt(len(exps))))
    ny = int(np.ceil(len(exps)/nx))
    fig, axs = plt.subplots(nx, ny, figsize = (14, 10), subplot_kw={'projection': ccrs.PlateCarree()}) 

    ts_dataset = create_ds_exp(ts_dataset)

    if ref_exp is not None:
            y_ref = ts_dataset.sel(exp=ref_exp)['rsdt'] - ts_dataset.sel(exp=ref_exp)[ 'rsut']- ts_dataset.sel(exp=ref_exp)[ 'rlut']
    
    divnorm = mcolors.BoundaryNorm(clevels, ncolors=plt.colormaps['RdBu_r'].N, clip=True)
    
    for exp, ax in zip(exps, axs.flatten()):

        if exp != ref_exp:
            y = ts_dataset.sel(exp=exp)['rsdt'] - ts_dataset.sel(exp=exp)[ 'rsut']- ts_dataset.sel(exp=exp)[ 'rlut']

            if y_ref is not None: y = (y - y_ref)#/y_ref
            
            d = ax.pcolormesh(ts_dataset.sel(exp = exp).lon, ts_dataset.sel(exp = exp).lat, y,  transform=ccrs.PlateCarree(),cmap='Reds') #, norm=divnorm)
            gl = ax.gridlines(draw_labels={"bottom": "x", "left": "y"}, color='gray', alpha=0.5)
            gl.xlabel_style = {'size': 10}
            gl.ylabel_style = {'size':10}    
            ax.set_title(exp)
            ax.coastlines()

            cb2 = plt.colorbar(d, ax=ax, extend='both', shrink=0.7)
            # cb2.set_ticks([0, 1e-7,1e-6,1e-5, 1e-4, 1e-3])
            # cb2.set_ticklabels(['0','1e-7','1e-6','1e-5','1e-4','1e-3'], fontsize=12)
            #cb2.ax.set_ylabel(r'(°C)', fontsize=12)


    # clevels = [0, 1e-7, 2.5e-7, 5e-7, 7.5e-7, 1e-6, 2.5e-6, 5e-6, 7.5e-6, 1e-5, 2.5e-5, 5e-5, 7.5e-5, 1e-4, 2.5e-4, 5e-4, 7.5e-4, 1e-3]
    # divnorm = mcolors.BoundaryNorm(clevels,ncolors=plt.colormaps['RdBu_r'].N,clip=True)
    c = axs[0].pcolormesh(ts_dataset.sel(exp = ref_exp).lon,ts_dataset.sel(exp = ref_exp).lat, y_ref, cmap=cmo.cm.thermal,transform=ccrs.PlateCarree())#, norm=divnorm)
    cb = plt.colorbar(c, ax=axs[0], extend='both', shrink=0.7)
    #cb.set_ticks([0, 1e-7,1e-6,1e-5, 1e-4, 1e-3])
    #cb.set_ticklabels(['0','1e-7','1e-6','1e-5','1e-4','1e-3'], fontsize=12)
    #cb.ax.set_ylabel(r'(°C)', fontsize=12)
    axs[0].set_title(ref_exp)
    axs[0].coastlines()

    #fig.savefig(cart_out + f'check_profile_{domain}_{vname}_{'-'.join([exp for exp in exps])}.pdf')
    
    return fig

def plot_pattern_map(exps, user = None, read_again = [], cart_exp = '/ec/res4/scratch/{}/ece4/', cart_out = './output/',ref_exp_c=None, ref_exp = None, atm_only = False, atmvars = 'tas'.split(), ocevars = 'tos heatc qt_oce sos'.split(),  year_clim = None, density=False):
    
    """
    Plots timeseries of var "vname" in domain "domain" for all exps.

    Domain is one among: ['atm', 'oce', 'ice']
    """
    if cart_out is None:
        raise ValueError('cart_out not specified!')

    cart_out_nc = cart_out + '/exps_clim/'
    nx = 1
    ny = 1
    fig, axs = plt.subplots(nx, ny, figsize = (6,5), subplot_kw={'projection': ccrs.PlateCarree()}) 
    
    exps = exps +['1pc0', 'ctl0']

    map_all = read_output_map(exps, user = user+user, read_again = read_again, cart_exp = cart_exp, cart_out = cart_out_nc, atm_only = atm_only, atmvars = atmvars, ocevars = ocevars, year_clim = year_clim, density=density)

    oce_dataset = map_all[f'atm_map']
    oce_dataset = {co: oce_dataset[co] for co in oce_dataset if oce_dataset[co] is not None}

    if exps is None: exps = oce_dataset.keys()
    oce_dataset = create_ds_exp(oce_dataset)

    if isinstance(oce_dataset, xr.Dataset):
        tas = oce_dataset['tas']

    year1 = 1870
    year2 = 1980

    trend_T = np.zeros([90,180])
    intercept_T = np.zeros([90,180])

    trend_T0 = np.zeros([90,180])
    intercept_T0 = np.zeros([90,180])

    for i in range(np.shape(tas.sel(exp=ref_exp_c))[1]):
        for j in range(np.shape(tas.sel(exp=ref_exp_c))[2]):

            trend_T[i,j], intercept_T[i,j], _, _, _ = stats.linregress(np.arange(0,np.shape(tas.sel(exp=ref_exp_c))[0]), tas.sel(exp=ref_exp_c)[:,i,j])
            trend_T0[i,j], intercept_T0[i,j], _, _, _ = stats.linregress(np.arange(0,np.shape(tas.sel(exp='ctl0'))[0]), tas.sel(exp='ctl0')[:,i,j])


    deltaTAS_ref = tas.sel(exp=ref_exp)[:150] - np.arange(0,150)[:,np.newaxis, np.newaxis]*trend_T - intercept_T
    deltaTAS_c = tas.sel(exp='1pc0')[:150] - np.arange(0,150)[:,np.newaxis, np.newaxis]*trend_T0 - intercept_T0

    deltaT = deltaTAS_ref.sel(year=slice(year1,year1+10)).mean(axis=0)#*tot_area
    deltaT2 = deltaTAS_ref.sel(year=slice(year2,year2+10)).mean(axis=0)#*tot_area

    deltaT0 = deltaTAS_c.sel(year=slice(year1,year1+10)).mean(axis=0)#*tot_area
    deltaT20 = deltaTAS_c.sel(year=slice(year2,year2+10)).mean(axis=0)#*tot_area
    
    clevels = np.arange(-2,2.25,0.25)
    divnorm = mcolors.BoundaryNorm(clevels,ncolors=plt.colormaps['RdBu_r'].N,clip=True)
    c = axs.pcolormesh(deltaTAS_ref.lon,deltaTAS_ref.lat, deltaT2 - deltaT - (deltaT20 -deltaT0)  , cmap='RdBu_r',transform=ccrs.PlateCarree(),norm=divnorm)
    cb = plt.colorbar(c, ax=axs, extend='both', shrink=0.7)
    gl = axs.gridlines(draw_labels={"bottom": "x", "left": "y"}, color='gray', alpha=0.5)
    gl.xlabel_style = {'size': 10}
    gl.ylabel_style = {'size':10}
    #cb.set_ticks([0, 1e-7,1e-6,1e-5, 1e-4, 1e-3])
    #cb.set_ticklabels(['0','1e-7','1e-6','1e-5','1e-4','1e-3'], fontsize=12)
    cb.ax.set_ylabel(r'(W/m2/K)', fontsize=12)
    axs.set_title(ref_exp)
    axs.coastlines()

    # clevels = np.arange(0,3.25,0.25)
    # divnorm = mcolors.BoundaryNorm(clevels,ncolors=plt.colormaps['RdBu_r'].N,clip=True)
    # c = axs[0].pcolormesh(deltaTAS_ref.lon,deltaTAS_ref.lat, deltaT , cmap='Reds',transform=ccrs.PlateCarree(),norm=divnorm)
    # cb = plt.colorbar(c, ax=axs[0], extend='both', shrink=0.7)
    # gl = axs[0].gridlines(draw_labels={"bottom": "x", "left": "y"}, color='gray', alpha=0.5)
    # gl.xlabel_style = {'size': 10}
    # gl.ylabel_style = {'size':10}
    # #cb.set_ticks([0, 1e-7,1e-6,1e-5, 1e-4, 1e-3])
    # #cb.set_ticklabels(['0','1e-7','1e-6','1e-5','1e-4','1e-3'], fontsize=12)
    # cb.ax.set_ylabel(r'(W/m2/K)', fontsize=12)
    # axs[0].set_title(ref_exp)
    # axs[0].coastlines()

    # clevels = np.arange(0,11,1)
    # divnorm = mcolors.BoundaryNorm(clevels,ncolors=plt.colormaps['RdBu_r'].N,clip=True)
    # c = axs[1].pcolormesh(deltaTAS_ref.lon,deltaTAS_ref.lat, deltaT2, cmap='Reds',transform=ccrs.PlateCarree(),norm=divnorm)
    # cb = plt.colorbar(c, ax=axs[1], extend='both', shrink=0.7)
    # #cb.set_ticks([0, 1e-7,1e-6,1e-5, 1e-4, 1e-3])
    # #cb.set_ticklabels(['0','1e-7','1e-6','1e-5','1e-4','1e-3'], fontsize=12)
    # gl = axs[1].gridlines(draw_labels={"bottom": "x", "left": "y"}, color='gray', alpha=0.5)
    # gl.xlabel_style = {'size': 10}
    # gl.ylabel_style = {'size':10}
    # cb.ax.set_ylabel(r'(W/m2/K)', fontsize=12)
    # axs[1].set_title(ref_exp)
    # axs[1].coastlines()

    # clevels = np.arange(0,11,1)
    # divnorm = mcolors.BoundaryNorm(clevels, ncolors=plt.colormaps['RdBu_r'].N, clip=True)

    # c = axs[2].pcolormesh(deltaTAS_ref.lon,deltaTAS_ref.lat, deltaT2 - deltaT, cmap='Reds',transform=ccrs.PlateCarree(),norm=divnorm)
    # cb = plt.colorbar(c, ax=axs[2], extend='both', shrink=0.7)
    # gl = axs[2].gridlines(draw_labels={"bottom": "x", "left": "y"}, color='gray', alpha=0.5)
    # gl.xlabel_style = {'size': 10}
    # gl.ylabel_style = {'size':10}
    # #cb.set_ticks([0, 1e-7,1e-6,1e-5, 1e-4, 1e-3])
    # #cb.set_ticklabels(['0','1e-7','1e-6','1e-5','1e-4','1e-3'], fontsize=12)
    # cb.ax.set_ylabel(r'(W/m2/K)', fontsize=12)
    # axs[2].set_title(ref_exp)
    # axs[2].coastlines()

    plt.show()


def plot_anom_map(exps, user = None, bx=None, var = None, index=None,  map70 = None, label = None, domain=None, cart_exp = '/ec/res4/scratch/{}/ece4/', cart_out = './output/', ref_exp = None, atmvars = ''.split(), ocevars = ''.split(), icevars ='siconc'.split(),  year_clim = None, ax=None, density=False):
    
    """
    Plots 2d map of var "vname" in domain "domain" for all exps.
    Need to update every time clevels and configuration depending on what I want to plot
    """
    cart_out_nc = cart_out + '/exps_clim/'
    exps = exps+['1pc0', 'ctl0']

    map_all = read_output_map(exps, user = user, read_again = [], cart_exp = cart_exp, cart_out = cart_out_nc, atm_only = False, atmvars = atmvars, ocevars = ocevars, icevars=icevars,year_clim = year_clim, density=density)

    ts_dataset = map_all[f'{domain}_map'] 
    ts_dataset = {co: ts_dataset[co] for co in ts_dataset if ts_dataset[co] is not None}
    ts_dataset = create_ds_exp(ts_dataset)

    # update with trend!!
    y = ts_dataset.sel(exp = exps[0])[var] - ts_dataset.sel(exp = ref_exp)[var].mean(axis=0)  #anomalies for each state
    y_base = ts_dataset.sel(exp = '1pc0')[var] - ts_dataset.sel(exp = 'ctl0')[var].mean(axis=0)
    
    if domain == 'atm':
        lats = ts_dataset.lat
        lons = ts_dataset.lon
    else:
        lats = ts_dataset.nav_lat
        lons = ts_dataset.nav_lon

    #preindustrial state

    #clevels_pi = np.arange(0,1.1,0.1) # for sea-ice plot
    clevels_pi = np.arange(-5,40,5) # for sst

    cmap = cmo.cm.thermal
    divnorm = mcolors.BoundaryNorm(clevels_pi, plt.colormaps['RdBu_r'].N, clip=True)
    c = ax[0].pcolormesh(lons, lats, ts_dataset.sel(exp = ref_exp)[var].mean(axis=0)-273.15, cmap=cmap, transform=ccrs.PlateCarree(), norm=divnorm)
    gl = ax[0].gridlines(draw_labels={"bottom": "x", "left": "y"}, color='gray', alpha=0.5)
    gl.xlabel_style = {'size': 10}
    gl.ylabel_style = {'size':10}    
    ax[0].set_title(exps[0])
    ax[0].coastlines()

    cb2 = plt.colorbar(c, ax=ax[0], extend='both', shrink=0.7)
    cb2.ax.set_ylabel(label, fontsize=12)

    periods = [1850+55, 1850+85] #[(1850+55, 1850+85), (1910, 1930), (1990, 2000)]

    # tas
    clevels = [np.arange(0, 5.5, 0.5), np.arange(-2, 2.5, 0.5)]
    blevels = [clevels_pi, clevels_pi] #,np.arange(0, 6.5, 0.5),np.arange(0, 13, 1)]
    #sea-ice
    #clevels = [clevels_pi, np.arange(-10,11,1)]
    #blevels = [np.arange(0,1.1, 0.1), np.arange(0,1.1, 0.1)]

    if(exps[0] == exps[2]):
            diff = y.sel(year=slice(periods[0], periods[1])).mean(axis=0)
            divnorm = mcolors.BoundaryNorm(clevels[1], plt.colormaps['Reds'].N, clip=True)
            
            #divnorm_70 = mcolors.TwoSlopeNorm(vmin=-0.5, vcenter=0, vmax=0.1) # for sea ice
            divnorm_70 = mcolors.BoundaryNorm(clevels[0], plt.colormaps['RdBu_r'].N, clip=True) 

    else:
            anom_ref = y_base.sel(year=slice(periods[0], periods[1])).mean(axis=0)
            diff = (y).sel(year=slice(periods[0], periods[1])).mean(axis=0) - anom_ref
            divnorm = mcolors.BoundaryNorm(clevels[1], plt.colormaps['RdBu_r'].N, clip=True)

            #divnorm_70 = mcolors.TwoSlopeNorm(vmin=-0.5, vcenter=0, vmax=0.1) 
            divnorm_70 = mcolors.BoundaryNorm(clevels[0], plt.colormaps['RdBu_r'].N, clip=True) 
    #for sea-ice
    #maps = [ts_dataset.sel(exp = exps[0])[var].sel(year=slice(periods[0], periods[1])).mean(axis=0), y.sel(year=slice(periods[0], periods[1])).mean(axis=0)]
    # for tas
    maps = [ y.sel(year=slice(periods[0], periods[1])).mean(axis=0), diff]
    
    # areas = get_areas_nemo(exps[0], 'itcv', cart_exp = cart_exp)
    # mask = get_mask_nemo(exps[0], 'itcv', cart_exp = cart_exp, grid = 'T')
    # tot_area = np.nansum(areas*mask)

    # seaice_map = y.sel(year=slice(periods[0], periods[1])).mean(axis=0)
    # seaice_free = np.ma.masked_where(ts_dataset.sel(exp = ref_exp)[var].mean(axis=0)==0, seaice_map)
    # seaice_north = np.ma.masked_where(ts_dataset.sel(exp = ref_exp)[var].nav_lat <0, seaice_free)
    # area_free = np.ma.masked_where(ts_dataset.sel(exp = ref_exp)[var].mean(axis=0)==0, areas*mask)
    # area_north = np.ma.masked_where(ts_dataset.sel(exp = ref_exp)[var].nav_lat <0, area_free)

    # totarea_free = np.nansum(area_free)
    # totarea_north = np.nansum(area_north)
    # print(np.round(np.nansum(seaice_free*area_free),3))
    # print(np.round((np.nansum(seaice_north*area_north)),3))

    norms = [divnorm_70, divnorm]
    cmaps = ['Reds', 'RdBu_r'] # both rdbu for sea-ice

    for k, (map, norm, cmap) in enumerate(zip(maps, norms, cmaps)):
            k+=1
            #year 70 anomaly
            c = ax[k].pcolormesh(lons, lats, map , cmap=cmap, transform=ccrs.PlateCarree(), norm=norm)
            gl = ax[k].gridlines(draw_labels={"bottom": "x", "left": "y"}, color='gray', alpha=0.5)
            gl.xlabel_style = {'size': 10}
            gl.ylabel_style = {'size':10}    
            ax[k].set_title(exps[0])
            ax[k].coastlines()

            cb2 = plt.colorbar(c, ax=ax[k], extend='both', shrink=0.7)
            cb2.ax.set_ylabel(label, fontsize=12)
            
    # if (exps[0] != exps[2]):
    #     y = y-y_base
    #     norm2 = mcolors.BoundaryNorm(np.arange(-2,2.2,0.2),plt.colormaps['Reds'].N, clip=True)
    #     d = bx.contourf( np.arange(0,300),y.lat, y.mean(dim='lon').transpose(), cmap='RdBu_r', levels=np.arange(-2,2.2,0.2), norm=norm2, extend='both')
    # else:    
    #     norm2 = mcolors.BoundaryNorm(np.arange(-3,24,3),plt.colormaps['Reds'].N, clip=True)
    #     d = bx.contourf( np.arange(0,300),y.lat, y.mean(dim='lon').transpose(), cmap='Reds', levels=np.arange(-3,24,3), norm=norm2)

    # bx.set_xlim(0,150)
    # bx.set_title(exps[0])

    # cb = plt.colorbar(d, ax=bx, orientation='vertical')

    #fig.savefig(cart_out + f'check_profile_{domain}_{vname}_{'-'.join([exp for exp in exps])}.pdf')
    map70[index] = y.sel(year=slice(periods[0], periods[1])).mean(axis=0)
    return map70, lats, lons

def plot_cre_zonal_map(clim_all, exps=None, ref_exp=None, cart_out=cart_out):

    print('Plotting CRE zonal')

    cre_ds = {k: v for k, v in clim_all['cre_clim'].items() if v is not None}
    if exps is None: exps = list(cre_ds.keys())

    cre_ds = create_ds_exp(cre_ds)

    comps = [('Net', 'CRE'), ('Shortwave', 'CRE_SW'), ('Longwave', 'CRE_LW')]

    fig = plt.figure(figsize=(18, len(exps)*7+4))
    gs = gridspec.GridSpec(len(exps)*2+4, 2, figure=fig, width_ratios=[1, 0.3], wspace=-0.1)
    clevels = np.arange(-5, 5.5, 0.5)
    divnorm = mcolors.BoundaryNorm(clevels, plt.colormaps['RdBu_r'].N, clip=True)

    cmap = plt.colormaps['RdBu_r'].resampled(10)
    cols = cmap(np.linspace(0, 1, 10))
    cneg, cpos = cols[2], cols[7]

    for i, exp in enumerate(exps):

        y = cre_ds.sel(exp=exp)['rsnt']+ cre_ds.sel(exp=exp)['rlnt']- cre_ds.sel(exp=exp)['rlntcs']- cre_ds.sel(exp=exp)['rsntcs']
        yref = cre_ds.sel(exp=ref_exp)['rsnt']+ cre_ds.sel(exp=ref_exp)['rlnt']- cre_ds.sel(exp=ref_exp)['rlntcs']- cre_ds.sel(exp=ref_exp)['rsntcs']

        if(exp==ref_exp):
            axm = fig.add_subplot(gs[2*i:2*i+2, 0], projection=ccrs.PlateCarree())
            axz = fig.add_subplot(gs[2*i:2*i+2, 1])
            diff = y
            
            nlevels = np.arange(-120,60,20)
            norm = mcolors.BoundaryNorm(nlevels, plt.colormaps['RdBu_r'].N, clip=True)

            pcm = axm.pcolormesh(cre_ds['rsnt'].lon, cre_ds['rsnt'].lat, diff, cmap='viridis',norm=norm,
                              transform=ccrs.PlateCarree())

            cbax = fig.add_axes([axm.get_position().x0,
                                 axm.get_position().y0 - 0.03,
                                 axm.get_position().width, 0.015])
            cb = plt.colorbar(pcm, cax=cbax, orientation='horizontal', extend='both', aspect=40)
            cb.set_label('(W m$^{-2}$)', fontsize=15)
            axz.set_xlabel('(W m$^{-2}$)', fontsize=15, labelpad=15)

        else:
            axm = fig.add_subplot(gs[2*i+1:2*i+3, 0], projection=ccrs.PlateCarree())
            axz = fig.add_subplot(gs[2*i+1:2*i+3, 1])
            diff = (y- yref)
        #diff = regrid_ifs(diff)
        #_, p = stats.ttest_ind(y[-win:], yref, equal_var=False)
        #diff = np.ma.masked_where(p > 0.05, diff)
        
            pcm = axm.pcolormesh(cre_ds['rsnt'].lon, cre_ds['rsnt'].lat, diff, cmap='RdBu_r',
                              norm=divnorm, 
                              transform=ccrs.PlateCarree())

        axm.set_title(exp, fontsize=15)
        axm.coastlines()
        gl = axm.gridlines(draw_labels={"bottom": "x", "left": "y"}, color='gray', alpha=0.5)
        gl.xlabel_style, gl.ylabel_style = {'size': 10}, {'size': 10}

        if (exp == ref_exp):
            zdiff= y.mean(axis=(1))
        else:   
            zdiff = (y - yref).mean(axis=(1))
            axz.set_xlim(-5, 5)
        axz.plot(zdiff, cre_ds.lat, color='k')
        axz.fill_betweenx(cre_ds['rsnt'].lat, zdiff, where=zdiff >= 0, color=cpos, alpha=0.7)
        axz.fill_betweenx(cre_ds['rsnt'].lat, zdiff, where=zdiff < 0, color=cneg, alpha=0.7)

        axz.set_ylim(-90, 90)
        axz.set_yticks([-60, -30, 0, 30, 60])
        axz.set_yticklabels([])
        axz.tick_params(labelsize=12, right=True, left=True, axis='y', direction='in')
        axz.tick_params(labelsize=11, axis='x')

        if i == len(exps) - 1:
            cbax = fig.add_axes([axm.get_position().x0,
                                 axm.get_position().y0 - 0.03,
                                 axm.get_position().width, 0.015])
            cb = plt.colorbar(pcm, cax=cbax, orientation='horizontal', extend='both', aspect=40)
            cb.set_label('(W m$^{-2}$)', fontsize=15)
            axz.set_xlabel('(W m$^{-2}$)', fontsize=15, labelpad=15)
    
    plt.show()
    fig.savefig(cart_out + f'check_profile_cre_{'-'.join([exp for exp in exps])}.pdf')

    return fig

def plot_zonal_ohue_correlation(clim_all, domain, vname, vcoord='deptht', exps = None, ref_exp = None, norm_factor = 1., cart_out = cart_out, colors=None):
    """
    Plots vertical profile of var "vname" in domain "domain" for all exps.
    Domain is one among: ['atm', 'oce', 'ice']
    """
    decades = np.arange(-6, -3)   # 1e-8 → 1e-4

    # 1–5 spacing per decade
    pos_levels = np.sort(np.concatenate([1e0 * 10.0**decades,5e0 * 10.0**decades]))
    pos_levels = pos_levels[pos_levels <= 1e-4]

    neg_levels = -pos_levels[::-1]
    neg_levels = neg_levels[neg_levels >=-1e-4]

    #llevels = np.concatenate((neg_levels, [-5e-9,-1e-9],[0.0],[1e-9,5e-9,], pos_levels))
    llevels = np.concatenate((neg_levels,[ -5e-7,-1e-7 ],[0.0],[1e-7,5e-7],  pos_levels))

    # Keep only within desired max
    N = 18 #22
    center_bin = 8 #10 

    neg_colors = plt.cm.RdBu_r(np.linspace(0, 0.5, center_bin, endpoint=False))
    pos_colors = plt.cm.RdBu_r(np.linspace(0.5, 1, N - center_bin, endpoint=True))

    colors_combined = np.vstack([neg_colors, pos_colors])
    cmap = mcolors.ListedColormap(colors_combined)
    norm = mcolors.BoundaryNorm(llevels, ncolors=N, clip=True)

    if domain not in ['oce', 'rho']:
        raise ValueError('domain should be one among: oce, rho')
    
    ts_dataset = clim_all[f'{domain}_clim']
    ts_dataset = {co: ts_dataset[co] for co in ts_dataset if ts_dataset[co] is not None}
    
    if exps is None: exps = ts_dataset.keys()

    fig, axs = plt.subplots(1,1, figsize = (8,5)) 

    ts_dataset = create_ds_exp(ts_dataset)

    if colors is None:
        colors = get_colors(exps)
    
    n2_zonal = []
    #ohue = [0.51541,0.47315, 0.56904, 0.60572, 0.78627, 0.68126, 0.54892]
    #ohue = [0.51541,0.47315, 0.56904, 0.60572, 0.68126, 0.54892]
    ohue = [0.84123329, 0.99007471, 1.0855393  , 1.19594174,0.92113063, 0.90884158]

    for exp in exps:

            y = ts_dataset.sel(exp = exp)[vname]
        
            if (vcoord == 'depth_mid'):
                v_levels = ts_dataset.sel(exp = exp)['density']['deptht']
                levels = (v_levels[1:].values + v_levels[:-1].values)/2
            else:
                levels = ts_dataset.sel(exp = exp)[vname][vcoord]
            
            z, lats = zonal_mean_irregular_xarray(y, y.nav_lat)
            n2_zonal.append(z)
            # llevels = np.arange(-1,1.1,0.1)
            # norm = mcolors.BoundaryNorm(llevels,ncolors=plt.colormaps['RdBu_r'].N,clip=True)
   
    n2 = np.zeros(np.shape(n2_zonal[0]))
    n2_zonal = np.array(n2_zonal)

    for i in range(np.shape(n2)[0]):
        for j in range(np.shape(n2)[1]):
           
            n2[i,j] = stats.linregress(n2_zonal[:,i,j], ohue)[2]

    #clevels = [0, 1e-7, 2.5e-7, 5e-7, 7.5e-7, 1e-6, 2.5e-6, 5e-6, 7.5e-6, 1e-5, 2.5e-5, 5e-5, 7.5e-5, 1e-4, 2.5e-4, 5e-4, 7.5e-4, 1e-3]
    #divnorm = mcolors.BoundaryNorm(clevels,ncolors=plt.colormaps['RdBu_r'].N,clip=True)
    c = axs.contourf(lats, -levels, n2, cmap='RdBu_r') #, levels=clevels, norm=divnorm)
    axs.set_ylim(-3500,0)
    cb = plt.colorbar(c, ax=axs, extend='both')
    #cb.set_ticks([0, 1e-7,1e-6,1e-5, 1e-4, 1e-3])
    #cb.set_ticklabels(['0','1e-7','1e-6','1e-5','1e-4','1e-3'], fontsize=12)
    #cb.ax.set_ylabel(r'$N^2$ $(s^{-2}$)', fontsize=12)
    #axs.set_title(ref_exp)

    #fig.savefig(cart_out + f'check_profile_{domain}_{vname}_{'-'.join([exp for exp in exps])}.pdf')
    
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

    n = len(slope_dict)
    ncols = 3
    nrows = int(np.ceil(n / ncols))

    fig, axs = plt.subplots(
        nrows, ncols,
        figsize=(4*ncols, 2.5*nrows),
        subplot_kw={'projection': ccrs.PlateCarree()}
    )
    axs = axs.flatten()

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
def calc_and_plot_slopes_from_raw(param_map, ref_exp='n000', user=None,
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

    slope_dict, r2_dict, anom_full_dict, slope_50pct_dict = {}, {}, {}, {}

    # --- load tuning values
    param_folder = '/ec/res4/hpcperm/ecme3038/ecearth/ecearth4/ECtuner/exps_413lr/'
    param_yaml = load_param_values(param_folder)

    def normalize_exp_key(exp, available_keys):
        exp_num = exp.replace('n', '').lstrip('0') or '0'
        key = exp_num.zfill(2)
        if key not in available_keys:
            raise KeyError(f"No matching experiment '{exp}' in YAML ({list(available_keys)})")
        return key

    for param, exps in param_map.items():
        
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
            ds = xr.open_mfdataset(filz, use_cftime=True, chunks={})
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

        # --- retrieve parameter values
        try:
            key_minus = normalize_exp_key(exp_minus, param_yaml.keys())
            key_ref   = normalize_exp_key(ref_exp,   param_yaml.keys())
            key_plus  = normalize_exp_key(exp_plus,  param_yaml.keys())
        except KeyError as e:
            print(f"Skip {param}: {e}")
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
        slope_per50pct = slope * (abs(p_ref) * 0.5 if p_ref not in [0, None, np.nan] else np.nan)
        slope_per50pct.name = f"{param}_slope_per50pct"
        slope_per50pct.attrs['units'] = 'W/m² per 50%'

        slope_dict[param] = slope
        r2_dict[param] = r2
        anom_full_dict[param] = anom_full
        slope_50pct_dict[param] = slope_per50pct

        r2_mean = slope.attrs.get('r2_mean', np.nan)
        r2_min  = slope.attrs.get('r2_min', np.nan)
        print(f" {param}: mean R²={r2_mean:.3f}, min R²={r2_min:.3f}")

    # --- Plot 1: slope per 50%
    print("\nPlot 1: Sensitivity normalized (W/m² per 50%)")
    plot_all_slopes(slope_50pct_dict, r2_dict=r2_dict, vmin=-3, vmax=3, cmap='RdBu_r', r2_thresh=r2_thresh,
                    filename='plot_slope_per50pct.png', label='TOA Net (W/m² per 50% param change)')

    # --- Plot 2: physical effect (total anomaly)
    print("\nPlot 2: Total effect minus→plus (W/m²)")
    plot_all_slopes(anom_full_dict, r2_dict=r2_dict, vmin=-10, vmax=10, cmap='RdBu_r', r2_thresh=r2_thresh,
                    filename='plot_anom_full.png', label='TOA Net anomaly (W/m²)')

    return slope_dict, r2_dict, slope_50pct_dict, anom_full_dict

### analysis function

def compute_ohue(exps, user = None, window=30, read_again = [], cart_exp = '/ec/res4/scratch/{}/ece4/', cart_out = './output/', imbalance = 0., ref_exp = None, atm_only = False, atmvars = 'rsut rlut rsdt tas pr'.split(), ocevars = 'tos heatc qt_oce sos'.split(), icevars = 'siconc sivolu sithic'.split(), year_clim = None, plot_diffref=False, plot_param=False, param_map={}, skip_first_year=False, exp_type = 'PD', density=False):
    """
    Plots timeseries of var "vname" in domain "domain" for all exps.

    Domain is one among: ['atm', 'oce', 'ice']
    """
    if cart_out is None:
        raise ValueError('cart_out not specified!')

    cart_out_nc = cart_out + '/exps_clim/'

    clim_all = read_output(exps, user = user, read_again = read_again, cart_exp = cart_exp, cart_out = cart_out_nc, atm_only = atm_only, atmvars = atmvars, ocevars = ocevars, icevars = icevars, year_clim = year_clim, density=density)

    toa_dataset = clim_all[f'atm_mean']
    oce_dataset = clim_all[f'oce_mean']

    toa_dataset = {co: toa_dataset[co] for co in toa_dataset if toa_dataset[co] is not None}
    oce_dataset = {co: oce_dataset[co] for co in oce_dataset if oce_dataset[co] is not None}

    if exps is None: exps = toa_dataset.keys()
    toa_dataset = create_ds_exp(toa_dataset)

    if exps is None: exps = oce_dataset.keys()
    oce_dataset = create_ds_exp(oce_dataset)

    if isinstance(toa_dataset, xr.Dataset):
        toa = toa_dataset['rsdt'] - toa_dataset[ 'rsut']- toa_dataset[ 'rlut']
        tas = toa_dataset['tas']

        tos = oce_dataset['tos']
        shf = oce_dataset['qt_oce']

    trend_N, intercept_N, _, _, _ = stats.linregress(np.arange(0,len(toa.sel(exp=exps[1]))), toa.sel(exp=exps[1]))
    trend_SST, intercept_SST, _,_,_ = stats.linregress(np.arange(0,len(toa.sel(exp=exps[1]))), tas.sel(exp=exps[1]))
    deltaN = (toa.sel(exp=exps[0]) - trend_N*np.arange(0,len(toa.sel(exp=exps[0])))-intercept_N)
    deltaSST = (tas.sel(exp=exps[0]) - trend_SST*np.arange(0,len(toa.sel(exp=exps[0])))-intercept_SST)
    
    ohue_atm = (deltaN.sel(year =slice(1850+55, 1850+85)).mean(axis=0)/deltaSST.sel(year =slice(1850+55, 1850+85)).mean(axis=0))
    ohue_atm_ts = deltaN.rolling(year=window).mean()/deltaSST.rolling(year=window).mean() 

    trend_SHF, intercept_SHF, _, _, _ = stats.linregress(np.arange(0,len(shf.sel(exp=exps[1]))), shf.sel(exp=exps[1]))
    trend_TOS, intercept_TOS, _,_,_ = stats.linregress(np.arange(0,len(tos.sel(exp=exps[1]))), tos.sel(exp=exps[1]))
    deltaSHF = (shf.sel(exp=exps[0]) - trend_SHF*np.arange(0,len(shf.sel(exp=exps[0])))-intercept_SHF)
    deltaTOS = (tos.sel(exp=exps[0]) - trend_TOS*np.arange(0,len(tos.sel(exp=exps[0])))-intercept_TOS)

    ohue_ts = deltaSHF.rolling(year=window).mean()/deltaTOS.rolling(year=window).mean() 
    ohue_oce = (deltaSHF.sel(year =slice(1850+55, 1850+85)).mean(axis=0)/deltaTOS.sel(year =slice(1850+55, 1850+85)).mean(axis=0))

    # deltaSHF = (shf.sel(exp=exps[0]) - shf.sel(exp=exps[1]).mean(axis=0))
    # deltaTOS = (tos.sel(exp=exps[0]) - tos.sel(exp=exps[1]).mean(axis=0))
    # ohue_notrend = deltaSHF.rolling(year=window).mean()/deltaTOS.rolling(year=window).mean() 
    
    #return ohue_atm_ts, ohue_ts
    return ohue_ts, ohue_atm, deltaSST.sel(year =slice(1850+55, 1850+85)).mean(axis=0), deltaN.sel(year =slice(1850+55, 1850+85)).mean(axis=0), ohue_oce, deltaTOS.sel(year =slice(1850+55, 1850+85)).mean(axis=0), deltaSHF.sel(year =slice(1850+55, 1850+85)).mean(axis=0)

def plot_ohue_zonal(exps, user = None, read_again = [], cart_exp = '/ec/res4/scratch/{}/ece4/', cart_out = './output/', color=None, window=40, imbalance = 0., ref_exp = None, atm_only = False, atmvars = ''.split(), ocevars = 'tos heatc qt_oce sos'.split(),  year_clim = None, ax=None, density=False):
    """
    Plots timeseries of var "vname" in domain "domain" for all exps.

    Domain is one among: ['atm', 'oce', 'ice']
    """
    if cart_out is None:
        raise ValueError('cart_out not specified!')

    cart_out_nc = cart_out + '/exps_clim/'

    map_all = read_output_map(exps, user = user, read_again = read_again, cart_exp = cart_exp, cart_out = cart_out_nc, atm_only = atm_only, atmvars = atmvars, ocevars = ocevars, year_clim = year_clim, density=density)

    oce_dataset = map_all[f'oce_map']
    oce_dataset = {co: oce_dataset[co] for co in oce_dataset if oce_dataset[co] is not None}

    if exps is None: exps = oce_dataset.keys()
    oce_dataset = create_ds_exp(oce_dataset)

    if isinstance(oce_dataset, xr.Dataset):
        shf = oce_dataset['qt_oce']

    clim_all = read_output(exps, user = user, read_again = read_again, cart_exp = cart_exp, cart_out = cart_out_nc, atm_only = atm_only, atmvars = atmvars, ocevars = ocevars, icevars = [], year_clim = year_clim, density=density)
    
    oce_mean = clim_all[f'oce_mean']
    oce_mean = {co: oce_mean[co] for co in oce_mean if oce_mean[co] is not None}

    oce_mean = create_ds_exp(oce_mean)

    if isinstance(oce_mean, xr.Dataset):
        tos = oce_mean['tos']

    deltaTOS = (tos.sel(exp=exps[0])-tos.sel(exp=exps[1]).mean(axis=0))
    deltaN = (shf.sel(exp=exps[0])-shf.sel(exp=exps[1]).mean(axis=0))

    ohue = deltaN.rolling(year=window).mean()/deltaTOS.rolling(year=window).mean()
    #ohue = deltaN/deltaTOS
    zonal_ohue, lats = zonal_mean_irregular_xarray(ohue, ohue.nav_lat) 

    zonal_N, lats = zonal_mean_irregular_xarray(deltaN.rolling(year=window).mean(), deltaN.nav_lat)
    #zonal_N, lats = zonal_mean_irregular_xarray(deltaN, deltaN.nav_lat)

    return zonal_ohue, zonal_N, lats
    
def plot_year70_map(exps, user = None, read_again = [], cart_exp = '/ec/res4/scratch/{}/ece4/', cart_out = './output/',ref_exp_c=None, ref_exp = None, atm_only = False, atmvars = 'rsut rlut rsdt tas pr'.split(), ocevars = 'tos heatc qt_oce sos'.split(),  year_clim = None, density=False):
    
    """
    Plots timeseries of var "vname" in domain "domain" for all exps.

    Domain is one among: ['atm', 'oce', 'ice']
    """
    if cart_out is None:
        raise ValueError('cart_out not specified!')

    cart_out_nc = cart_out + '/exps_clim/'
    nx = 1
    ny = 2
    fig, axs = plt.subplots(nx, ny, figsize = (12,5), subplot_kw={'projection': ccrs.PlateCarree()}) 

    map_all = read_output_map(exps, user = user, read_again = read_again, cart_exp = cart_exp, cart_out = cart_out_nc, atm_only = atm_only, atmvars = atmvars, ocevars = ocevars, year_clim = year_clim, density=density)

    oce_dataset = map_all[f'oce_map']
    oce_dataset = {co: oce_dataset[co] for co in oce_dataset if oce_dataset[co] is not None}

    if exps is None: exps = oce_dataset.keys()
    oce_dataset = create_ds_exp(oce_dataset)

    if isinstance(oce_dataset, xr.Dataset):
        shf = oce_dataset['qt_oce']

    atm_dataset = map_all[f'atm_map']
    atm_dataset = {co: atm_dataset[co] for co in atm_dataset if atm_dataset[co] is not None}
    atm_dataset = create_ds_exp(atm_dataset)
    toa = atm_dataset['rsdt'] - atm_dataset['rlut'] - atm_dataset['rsut']
    
    clim_all = read_output(exps, user = user, read_again = read_again, cart_exp = cart_exp, cart_out = cart_out_nc, atm_only = atm_only, atmvars = atmvars, ocevars = ocevars, icevars = [], year_clim = year_clim, density=density)
    atm_mean = clim_all[f'atm_mean']
    atm_mean = {co: atm_mean[co] for co in atm_mean if atm_mean[co] is not None}
    atm_mean = create_ds_exp(atm_mean)
    tas = atm_mean['tas']

    #forcing = compute_radiative_forcing()
    ds_forcing = xr.open_mfdataset('/ec/res4/hpcperm/itcv/analysis/forcing_1pct/'+exps[0]+'_forcing.nc')
    forcing = ds_forcing.forcing

    clim_all = read_output(exps, user = user, read_again = read_again, cart_exp = cart_exp, cart_out = cart_out_nc, atm_only = atm_only, atmvars = atmvars, ocevars = ocevars, icevars = [], year_clim = year_clim, density=density)
    oce_mean = clim_all[f'oce_mean']
    oce_mean = {co: oce_mean[co] for co in oce_mean if oce_mean[co] is not None}
    oce_mean = create_ds_exp(oce_mean)
    
    tos = oce_mean['tos']
    
    year1 = 1850+70-15
    window = 30

    areas = get_areas_nemo(exps[0], 'itcv', cart_exp = cart_exp)
    mask = get_mask_nemo(exps[0], 'itcv', cart_exp = cart_exp, grid = 'T')
    tot_area = np.nansum(areas*mask)

    if ref_exp is not None:
        trend_SHF = np.zeros([148,180])
        intercept_SHF = np.zeros([148,180])

        for i in range(np.shape(shf.sel(exp=ref_exp_c))[1]):
            for j in range(np.shape(shf.sel(exp=ref_exp_c))[2]):

                trend_SHF[i,j], intercept_SHF[i,j], _, _, _ = stats.linregress(np.arange(0,np.shape(shf.sel(exp=ref_exp_c))[0]), shf.sel(exp=ref_exp_c)[:,i,j])

        trend_SST, intercept_SST, _,_,_ = stats.linregress(np.arange(0,len(tos.sel(exp=ref_exp))), tos.sel(exp=ref_exp_c))
        
        deltaSHF_ref = shf.sel(exp=ref_exp)[:150] - np.arange(0,150)[:,np.newaxis, np.newaxis]*trend_SHF - intercept_SHF  
        deltaTOS_ref = tos.sel(exp=ref_exp)[:150] - np.arange(0,150)*trend_SST - intercept_SST
        deltaT = deltaTOS_ref.sel(year=slice(year1,year1+window)).mean(axis=0)#*tot_area
        ohue_ref30 = (deltaSHF_ref).sel(year=slice(year1,year1+window)).mean(axis=0)/ deltaT

        trend_N = np.zeros([90,180])
        intercept_N = np.zeros([90,180])

        for i in range(np.shape(toa.sel(exp=ref_exp_c))[1]):
            for j in range(np.shape(toa.sel(exp=ref_exp_c))[2]):

                trend_N[i,j], intercept_N[i,j], _, _, _ = stats.linregress(np.arange(0,np.shape(toa.sel(exp=ref_exp_c))[0]), toa.sel(exp=ref_exp_c)[:,i,j])

        trend_T, intercept_T, _,_,_ = stats.linregress(np.arange(0,len(tas.sel(exp=ref_exp))), tas.sel(exp=ref_exp_c))
        deltaN_ref = toa.sel(exp=ref_exp)[:150] - np.arange(0,150)[:,np.newaxis, np.newaxis]*trend_N - intercept_N - forcing #[:,np.newaxis, np.newaxis] 
        deltaTAS_ref = tas.sel(exp=ref_exp)[:150] - np.arange(0,150)*trend_T - intercept_T
        
        weights = np.cos(np.deg2rad(deltaN_ref.lat))
        tot_weights = weights.sum()*deltaN_ref.lon.size

        deltaT = deltaTAS_ref.sel(year=slice(year1,year1+40)).mean(axis=0)#*tot_weights
        lambda_ref30 = (deltaN_ref).sel(year=slice(year1,year1+40)).mean(axis=0)/ deltaT

    clevels = np.arange(-0.001, 0.0011, 0.0001)
    clevels = np.arange(-6,11,1)
    divnorm = mcolors.TwoSlopeNorm(vcenter=0, vmin=-10, vmax=10)
    c = axs[0].pcolormesh(deltaSHF_ref.nav_lon,deltaSHF_ref.nav_lat, ohue_ref30, cmap='RdBu_r',transform=ccrs.PlateCarree(),norm=divnorm)
    cb = plt.colorbar(c, ax=axs[0], extend='both', shrink=0.7)
    gl = axs[0].gridlines(draw_labels={"bottom": "x", "left": "y"}, color='gray', alpha=0.5)
    gl.xlabel_style = {'size': 10}
    gl.ylabel_style = {'size':10}
    #cb.set_ticks([0, 1e-7,1e-6,1e-5, 1e-4, 1e-3])
    #cb.set_ticklabels(['0','1e-7','1e-6','1e-5','1e-4','1e-3'], fontsize=12)
    cb.ax.set_ylabel(r'(W/m2/K)', fontsize=12)
    axs[0].set_title(ref_exp)
    axs[0].coastlines()

    divnorm = mcolors.TwoSlopeNorm(vcenter=0, vmin=-6, vmax=3)
    c = axs[1].pcolormesh(deltaN_ref.lon,deltaN_ref.lat, lambda_ref30, cmap='RdBu_r',transform=ccrs.PlateCarree(),norm=divnorm)
    cb = plt.colorbar(c, ax=axs[1], extend='both', shrink=0.7)
    #cb.set_ticks([0, 1e-7,1e-6,1e-5, 1e-4, 1e-3])
    #cb.set_ticklabels(['0','1e-7','1e-6','1e-5','1e-4','1e-3'], fontsize=12)
    gl = axs[1].gridlines(draw_labels={"bottom": "x", "left": "y"}, color='gray', alpha=0.5)
    gl.xlabel_style = {'size': 10}
    gl.ylabel_style = {'size':10}
    cb.ax.set_ylabel(r'(W/m2/K)', fontsize=12)
    axs[1].set_title(ref_exp)
    axs[1].coastlines()

    plt.show()

def compute_year70_map(exps, user = None, read_again = [], cart_exp = '/ec/res4/scratch/{}/ece4/', cart_out = './output/',ref_exp_c=None, ref_exp = None, atm_only = False, atmvars = 'rsut rlut rsdt tas pr'.split(), ocevars = 'tos heatc qt_oce sos'.split(),  year_clim = None, density=False):
    
    """
    Plots timeseries of var "vname" in domain "domain" for all exps.

    Domain is one among: ['atm', 'oce', 'ice']
    """
    if cart_out is None:
        raise ValueError('cart_out not specified!')

    cart_out_nc = cart_out + '/exps_clim/'
    nx = 1
    ny = 2
    fig, axs = plt.subplots(nx, ny, figsize = (12,5), subplot_kw={'projection': ccrs.PlateCarree()}) 

    map_all = read_output_map(exps, user = user, read_again = read_again, cart_exp = cart_exp, cart_out = cart_out_nc, atm_only = atm_only, atmvars = atmvars, ocevars = ocevars, year_clim = year_clim, density=density)

    oce_dataset = map_all[f'oce_map']
    oce_dataset = {co: oce_dataset[co] for co in oce_dataset if oce_dataset[co] is not None}

    if exps is None: exps = oce_dataset.keys()
    oce_dataset = create_ds_exp(oce_dataset)

    if isinstance(oce_dataset, xr.Dataset):
        shf = oce_dataset['qt_oce']

    atm_dataset = map_all[f'atm_map']
    atm_dataset = {co: atm_dataset[co] for co in atm_dataset if atm_dataset[co] is not None}
    atm_dataset = create_ds_exp(atm_dataset)
    toa = atm_dataset['rsdt'] - atm_dataset['rlut'] - atm_dataset['rsut']

    toa_cs  = atm_dataset['rlntcs'] + atm_dataset['rsntcs']
    toa_net = atm_dataset['rlnt'] + atm_dataset['rsnt']
    toa_cloud = toa_net - toa_cs

    toa_cs_lw = atm_dataset['rlntcs']
    toa_cs_sw = atm_dataset['rsntcs']

    toa_cloud_lw = atm_dataset['rlnt'] - atm_dataset['rlntcs']
    toa_cloud_sw = atm_dataset['rsnt'] - atm_dataset['rsntcs']

    atmos = [toa_net, toa_cs, toa_cs_lw, toa_cs_sw, toa_cloud, toa_cloud_lw, toa_cloud_sw]
    
    clim_all = read_output(exps, user = user, read_again = read_again, cart_exp = cart_exp, cart_out = cart_out_nc, atm_only = atm_only, atmvars = atmvars, ocevars = ocevars, icevars = [], year_clim = year_clim, density=density)
    atm_mean = clim_all[f'atm_mean']
    atm_mean = {co: atm_mean[co] for co in atm_mean if atm_mean[co] is not None}
    atm_mean = create_ds_exp(atm_mean)
    tas = atm_mean['tas']

    #forcing = compute_radiative_forcing()
    ds_forcing = xr.open_mfdataset('/ec/res4/hpcperm/itcv/analysis/forcing_1pct/'+exps[0]+'_forcing.nc')
    forcing = ds_forcing.forcing

    forcings = [ds_forcing.forcing_net, ds_forcing.forcing_cs, ds_forcing.forcing_cs_lw, ds_forcing.forcing_cs_sw, ds_forcing.forcing_cloud, ds_forcing.forcing_cloud_lw, ds_forcing.forcing_cloud_sw]

    clim_all = read_output(exps, user = user, read_again = read_again, cart_exp = cart_exp, cart_out = cart_out_nc, atm_only = atm_only, atmvars = atmvars, ocevars = ocevars, icevars = [], year_clim = year_clim, density=density)
    oce_mean = clim_all[f'oce_mean']
    oce_mean = {co: oce_mean[co] for co in oce_mean if oce_mean[co] is not None}
    oce_mean = create_ds_exp(oce_mean)
    
    tos = oce_mean['tos']
    
    year1 = 1850+70-15
    window = 30

    areas = get_areas_nemo(exps[0], 'itcv', cart_exp = cart_exp)
    mask = get_mask_nemo(exps[0], 'itcv', cart_exp = cart_exp, grid = 'T')
    tot_area = np.nansum(areas*mask)

    trend_SHF = np.zeros([148,180])
    intercept_SHF = np.zeros([148,180])

    for i in range(np.shape(shf.sel(exp=ref_exp_c))[1]):
        for j in range(np.shape(shf.sel(exp=ref_exp_c))[2]):

            trend_SHF[i,j], intercept_SHF[i,j], _, _, _ = stats.linregress(np.arange(0,np.shape(shf.sel(exp=ref_exp_c))[0]), shf.sel(exp=ref_exp_c)[:,i,j])

    trend_SST, intercept_SST, _,_,_ = stats.linregress(np.arange(0,len(tos.sel(exp=ref_exp))), tos.sel(exp=ref_exp_c))
    
    deltaSHF_ref = shf.sel(exp=ref_exp)[:150] - np.arange(0,150)[:,np.newaxis, np.newaxis]*trend_SHF - intercept_SHF  
    deltaTOS_ref = tos.sel(exp=ref_exp)[:150] - np.arange(0,150)*trend_SST - intercept_SST
    deltaT = deltaTOS_ref.sel(year=slice(year1,year1+window)).mean(axis=0)#*tot_area
    ohue_ref30 = (deltaSHF_ref).sel(year=slice(year1,year1+window)).mean(axis=0)/ deltaT

    ohue_ref30.to_netcdf(cart_out_nc + f'/KL/ohue_ref70_{ref_exp}.nc')

    trend_N = np.zeros([90,180])
    intercept_N = np.zeros([90,180])

    for i in range(np.shape(toa.sel(exp=ref_exp_c))[1]):
        for j in range(np.shape(toa.sel(exp=ref_exp_c))[2]):

            trend_N[i,j], intercept_N[i,j], _, _, _ = stats.linregress(np.arange(0,np.shape(toa.sel(exp=ref_exp_c))[0]), toa.sel(exp=ref_exp_c)[:,i,j])

    trend_T, intercept_T, _,_,_ = stats.linregress(np.arange(0,len(tas.sel(exp=ref_exp))), tas.sel(exp=ref_exp_c))
    deltaN_ref = toa.sel(exp=ref_exp)[:150] - np.arange(0,150)[:,np.newaxis, np.newaxis]*trend_N - intercept_N - forcing #[:,np.newaxis, np.newaxis] 
    deltaTAS_ref = tas.sel(exp=ref_exp)[:150] - np.arange(0,150)*trend_T - intercept_T
    
    weights = np.cos(np.deg2rad(deltaN_ref.lat))
    tot_weights = weights.sum()*deltaN_ref.lon.size

    deltaT = deltaTAS_ref.sel(year=slice(year1,year1+window)).mean(axis=0)#*tot_weights
    lambda_ref30 = (deltaN_ref).sel(year=slice(year1,year1+window)).mean(axis=0)/ deltaT

    lambda_ref30.to_netcdf(cart_out_nc + f'/KL/lambda_ref70_{ref_exp}.nc')

    for k, (toa_comp, feedback) in enumerate(zip(atmos, forcings)):

        trend_c = np.zeros([90,180])
        intercept_c = np.zeros([90,180])

        for i in range(np.shape(toa_comp.sel(exp=ref_exp_c))[1]):
            for j in range(np.shape(toa_comp.sel(exp=ref_exp_c))[2]):

                trend_c[i,j], intercept_c[i,j], _, _, _ = stats.linregress(np.arange(0,np.shape(toa_comp.sel(exp=ref_exp_c))[0]), toa_comp.sel(exp=ref_exp_c)[:,i,j])
        
        deltaN = toa_comp.sel(exp=ref_exp)[:150] - np.arange(0,150)[:,np.newaxis, np.newaxis]*trend_c - intercept_c - feedback #[:,np.newaxis, np.newaxis] 
        lambda_comp = (deltaN).sel(year=slice(year1,year1+window)).mean(axis=0)/ deltaT
        lambda_comp.to_netcdf(cart_out_nc+f'/KL/lambda_{feedback.long_name}70_{ref_exp}.nc')

    clevels = np.arange(-0.001, 0.0011, 0.0001)
    clevels = np.arange(-6,11,1)
    divnorm = mcolors.TwoSlopeNorm(vcenter=0, vmin=-10, vmax=10)
    c = axs[0].pcolormesh(deltaSHF_ref.nav_lon,deltaSHF_ref.nav_lat, ohue_ref30, cmap='RdBu_r',transform=ccrs.PlateCarree(),norm=divnorm)
    cb = plt.colorbar(c, ax=axs[0], extend='both', shrink=0.7)
    gl = axs[0].gridlines(draw_labels={"bottom": "x", "left": "y"}, color='gray', alpha=0.5)
    gl.xlabel_style = {'size': 10}
    gl.ylabel_style = {'size':10}
    #cb.set_ticks([0, 1e-7,1e-6,1e-5, 1e-4, 1e-3])
    #cb.set_ticklabels(['0','1e-7','1e-6','1e-5','1e-4','1e-3'], fontsize=12)
    cb.ax.set_ylabel(r'(W/m2/K)', fontsize=12)
    axs[0].set_title(ref_exp)
    axs[0].coastlines()

    divnorm = mcolors.TwoSlopeNorm(vcenter=0, vmin=-6, vmax=3)
    c = axs[1].pcolormesh(deltaN_ref.lon,deltaN_ref.lat, lambda_ref30, cmap='RdBu_r',transform=ccrs.PlateCarree(),norm=divnorm)
    cb = plt.colorbar(c, ax=axs[1], extend='both', shrink=0.7)
    #cb.set_ticks([0, 1e-7,1e-6,1e-5, 1e-4, 1e-3])
    #cb.set_ticklabels(['0','1e-7','1e-6','1e-5','1e-4','1e-3'], fontsize=12)
    gl = axs[1].gridlines(draw_labels={"bottom": "x", "left": "y"}, color='gray', alpha=0.5)
    gl.xlabel_style = {'size': 10}
    gl.ylabel_style = {'size':10}
    cb.ax.set_ylabel(r'(W/m2/K)', fontsize=12)
    axs[1].set_title(ref_exp)
    axs[1].coastlines()

    plt.show()


def plot_year70_corr_map(exps, axs=None, user = None, read_again = [], cart_exp = '/ec/res4/scratch/{}/ece4/', cart_out = './output/',ref_exp_c=None, ref_exp = None, atm_only = False, atmvars = 'rsut rlut rsdt tas pr'.split(), ocevars = 'tos heatc qt_oce sos'.split(),  year_clim = None, density=False):
    
    """
    Plots timeseries of var "vname" in domain "domain" for all exps.

    Domain is one among: ['atm', 'oce', 'ice']
    """
    if cart_out is None:
        raise ValueError('cart_out not specified!')

    cart_out_nc = cart_out + '/exps_clim/'
    nx = 2
    ny = 3
    #fig, axs = plt.subplots(nx, ny, figsize = (18,5), subplot_kw={'projection': ccrs.PlateCarree()}) 

    kappas_r = np.zeros([7,90,180])
    kappas = np.zeros([7,148,180])

    lambdas = np.zeros([7,90,180])

    kappa_global = np.zeros(7)
    lambda_global = np.zeros(7)

    for i, exp in enumerate(exps):

        lambda_ds = xr.open_mfdataset(cart_out_nc + f'/KL/lambda_ref70_{exp}.nc')
        kappa_ds_r = xr.open_mfdataset(cart_out_nc + f'/KL/ohue_ref70_{exp}_regrid.nc')
        kappa_ds =  xr.open_mfdataset(cart_out_nc + f'/KL/ohue_ref70_{exp}.nc')

        lambda_exp = lambda_ds.__xarray_dataarray_variable__
        kappa_exp_r = kappa_ds_r.__xarray_dataarray_variable__
        kappa_exp = kappa_ds.__xarray_dataarray_variable__

        if i == 0:
            lats = lambda_ds.lat
            lons = lambda_ds.lon

            lats_oce = kappa_ds.nav_lat
            lons_oce = kappa_ds.nav_lon

        weights = np.cos(np.deg2rad(lats))
        tot_weights = weights.sum()*lons.size

        areas = get_areas_nemo(exps[0], 'itcv', cart_exp = cart_exp)
        mask = get_mask_nemo(exps[0], 'itcv', cart_exp = cart_exp, grid = 'T')
        tot_area = np.nansum(areas*mask)

        areas_ds = kappa_exp.copy()
        areas_ds.values = areas*mask

        kappas[i]  = kappa_exp
        kappas_r[i]  = kappa_exp_r
        lambdas[i] = lambda_exp

        #kappa_global = (kappa_exp*weights/tot_weights).sum()
        kappa_global[i] = (kappa_exp*areas_ds/tot_area).sum().values
        lambda_global[i] = (lambda_exp*weights/tot_weights).sum()

        #print(np.round(kappa_global[i],2), np.round(lambda_global[i],2))

    # mca = MCA(kappas_r, lambdas)
    # mca.solve(complexify=True)
    # mca.rotate(3)

    # svals = mca.singular_values()
    # expvar = mca.explained_variance()

    # hom_patterns = mca.homogeneous_patterns()
    # het_patterns  = mca.heterogeneous_patterns()

    # mca.set_field_names('kappa', 'lambda')
    
    # for i in range(1,7):
    #     mca.plot(mode=i, threshold=0.25) #, **pkwargs)

    kl_corr = np.zeros([90,180])
    kl_p_values = np.zeros([90,180])
    k_corr = np.zeros([90,180])
    k_p_values = np.zeros([90,180])
    l_corr = np.zeros([90,180])
    l_p_values = np.zeros([90,180])
    k_std_r = np.zeros([90,180])
    k_std = np.zeros([148,180])
    l_std = np.zeros([90,180])
    kl_cov = np.zeros([90,180])

    for i in range(90):
            for j in range(180):
                _,_,kl_corr[i,j],kl_p_values[i,j],_ = stats.linregress(kappas_r[:,i,j], -lambdas[:,i,j])
                _,_,k_corr[i,j], k_p_values[i,j],_ = stats.linregress(kappas_r[:,i,j], -lambda_global)
                _,_,l_corr[i,j], l_p_values[i,j],_ = stats.linregress(kappa_global, -lambdas[:,i,j])

                k_std_r[i,j] = kappas_r[:,i,j].std()
                l_std[i,j] = lambdas[:,i,j].std()
                kl_cov[i,j] = np.cov(kappas_r[:,i,j], -lambdas[:,i,j])[0,1]
    
    for i in range(148):
        for j in range(180):
                k_std[i,j] = kappas[:,i,j].std()

    divnorm = mcolors.TwoSlopeNorm(vcenter=0, vmin=-1, vmax=1)
    cmap = 'RdBu_r'
    #np.ma.masked_where(kl_p_values > 0.1, kl_corr)
    comps = [kl_corr, np.ma.masked_where(kl_p_values > 0.05, k_corr), np.ma.masked_where(l_p_values > 0.05, l_corr)]
    lat = [lats, lats, lats]
    lon = [lons, lons, lons]
    titles = ['Local-Local', 'K Local - L global', 'K global - L local']
    colorbar_label = 'r'

    for i, (map,title, latitude, longitude) in enumerate(zip(comps, titles, lat, lon)):

        c = axs[0,i].pcolormesh(longitude, latitude, map, cmap=cmap,transform=ccrs.PlateCarree(),norm=divnorm)
        cb = plt.colorbar(c, ax=axs[0,i], extend='both', shrink=0.7)
        gl = axs[0,i].gridlines(draw_labels={"bottom": "x", "left": "y"}, color='gray', alpha=0.5)
        gl.xlabel_style = {'size': 10}
        gl.ylabel_style = {'size':10}
        #cb.set_ticks([0, 1e-7,1e-6,1e-5, 1e-4, 1e-3])
        #cb.set_ticklabels(['0','1e-7','1e-6','1e-5','1e-4','1e-3'], fontsize=12)
        cb.ax.set_ylabel(colorbar_label, fontsize=12)
        axs[0,i].set_title(title)
        axs[0,i].coastlines()

    clevels = np.arange(-3,3.5,0.5)
    divnorm = mcolors.BoundaryNorm(clevels,ncolors=plt.colormaps['RdBu_r'].N,clip=True) 
    cmap = 'RdBu_r'
    comps = [kl_cov, k_std_r, l_std]
    lat = [lats, lats, lats]
    lon = [lons, lons, lons]
    titles = ['Covariance', 'K std', 'L std']
    colorbar_label = 'W/m2/K'

    for i, (map,title, latitude, longitude) in enumerate(zip(comps, titles, lat, lon)): 

        c = axs[1,i].pcolormesh(longitude, latitude, map, cmap=cmap,transform=ccrs.PlateCarree(),norm=divnorm)
        cb = plt.colorbar(c, ax=axs[1,i], extend='both', shrink=0.7)
        gl = axs[1,i].gridlines(draw_labels={"bottom": "x", "left": "y"}, color='gray', alpha=0.5)
        gl.xlabel_style = {'size': 10}
        gl.ylabel_style = {'size':10}
        #cb.set_ticks([0, 1e-7,1e-6,1e-5, 1e-4, 1e-3])
        #cb.set_ticklabels(['0','1e-7','1e-6','1e-5','1e-4','1e-3'], fontsize=12)
        cb.ax.set_ylabel(colorbar_label, fontsize=12)
        axs[1,i].set_title(title)
        axs[1,i].coastlines()
    
    plt.show()

def plot_ohue_map(exps, user = None, read_again = [], cart_exp = '/ec/res4/scratch/{}/ece4/', cart_out = './output/',ref_exp_c=None, ref_exp = None, atm_only = False, atmvars = 'rsut rlut rsdt tas pr'.split(), ocevars = 'tos heatc qt_oce sos'.split(),  year_clim = None, density=False):
    
    """
    Plots timeseries of var "vname" in domain "domain" for all exps.

    Domain is one among: ['atm', 'oce', 'ice']
    """
    if cart_out is None:
        raise ValueError('cart_out not specified!')

    cart_out_nc = cart_out + '/exps_clim/'
    nx = 1
    ny = 3
    fig, axs = plt.subplots(nx, ny, figsize = (18,5), subplot_kw={'projection': ccrs.PlateCarree()}) 

    map_all = read_output_map(exps, user = user, read_again = read_again, cart_exp = cart_exp, cart_out = cart_out_nc, atm_only = atm_only, atmvars = atmvars, ocevars = ocevars, year_clim = year_clim, density=density)
    #map_con = read_output(controls, user = user, read_again = read_again, cart_exp = cart_exp, cart_out = cart_out_nc, atm_only = atm_only, atmvars = atmvars, ocevars = ocevars, year_clim = year_clim, density=density)

    oce_dataset = map_all[f'oce_map']
    oce_dataset = {co: oce_dataset[co] for co in oce_dataset if oce_dataset[co] is not None}

    if exps is None: exps = oce_dataset.keys()
    oce_dataset = create_ds_exp(oce_dataset)

    if isinstance(oce_dataset, xr.Dataset):
        shf = oce_dataset['qt_oce']
        tos = oce_dataset['tos']

    clim_all = read_output(exps, user = user, read_again = read_again, cart_exp = cart_exp, cart_out = cart_out_nc, atm_only = atm_only, atmvars = atmvars, ocevars = ocevars, icevars = [], year_clim = year_clim, density=density)
    oce_mean = clim_all[f'oce_mean']
    oce_mean = {co: oce_mean[co] for co in oce_mean if oce_mean[co] is not None}
    oce_mean = create_ds_exp(oce_mean)
    
    #tos = oce_mean['tos']

    year1 = 1870
    year2 = 1960
    window = 30

    areas = get_areas_nemo(exps[0], 'itcv', cart_exp = cart_exp)
    mask = get_mask_nemo(exps[0], 'itcv', cart_exp = cart_exp, grid = 'T')
    tot_area = np.nansum(areas*mask)

    if ref_exp is not None:
        trend_N = np.zeros([148,180])
        intercept_N = np.zeros([148,180])

        trend_T = np.zeros([148,180])
        intercept_T = np.zeros([148,180])


        for i in range(np.shape(shf.sel(exp=ref_exp_c))[1]):
            for j in range(np.shape(shf.sel(exp=ref_exp_c))[2]):

                trend_N[i,j], intercept_N[i,j], _, _, _ = stats.linregress(np.arange(0,np.shape(shf.sel(exp=ref_exp_c))[0]), shf.sel(exp=ref_exp_c)[:,i,j])
                trend_T[i,j], intercept_T[i,j], _, _, _ = stats.linregress(np.arange(0,np.shape(tos.sel(exp=ref_exp_c))[0]), tos.sel(exp=ref_exp_c)[:,i,j])

        #trend_SST, intercept_SST, _,_,_ = stats.linregress(np.arange(0,len(tos.sel(exp=ref_exp))), tos.sel(exp=ref_exp_c))
        
        deltaN_ref = shf.sel(exp=ref_exp)[:150] - np.arange(0,150)[:,np.newaxis, np.newaxis]*trend_N - intercept_N  
        deltaTOS_ref = tos.sel(exp=ref_exp)[:150] - np.arange(0,150)[:,np.newaxis, np.newaxis]*trend_T - intercept_T

        deltaT = deltaTOS_ref.rolling(year=30).mean()
        #deltaT = deltaTOS_ref.sel(year=slice(year1,year1+window)).mean(axis=0)#*tot_area
        #deltaT2 = deltaTOS_ref.sel(year=slice(year2,year2+window)).mean(axis=0)#*tot_area

        ohue = deltaN_ref.rolling(year=30).mean()/deltaT 
        print(ohue)
        ohue.to_netcdf(f'/ec/res4/hpcperm/itcv/analysis/exps_clim/KL/ohue_ts_{ref_exp}_.nc')

        ohue_ref30 = (deltaN_ref).sel(year=slice(year1,year1+window)).mean(axis=0)/ deltaT[85]
        ohue_ref70 = (deltaN_ref).sel(year=slice(year2,year2+window)).mean(axis=0)/ deltaT[149]

    # for exp, exp_c, ax in zip(exps, controls, axs.flatten()):

    #     if exp != ref_exp:

    #         trend_N = np.zeros([90,180])
    #         intercept_N = np.zeros([90,180])

    #         for i in range(np.shape(shf.sel(exp=ref_exp_c))[1]):
    #             for j in range(np.shape(shf.sel(exp=ref_exp_c))[2]):

    #                 trend_N[i,j], intercept_N[i,j], _, _, _ = stats.linregress(np.arange(0,np.shape(shf.sel(exp=ref_exp_c))[0]), shf.sel(exp=ref_exp_c)[:,i,j])

    #         trend_SST, intercept_SST, _,_,_ = stats.linregress(np.arange(0,len(tos.sel(exp=exp))), tos.sel(exp=ref_exp_c))
    #         deltaN_ref = shf.sel(exp=ref_exp)[:150] - np.arange(0,150)[:,np.newaxis, np.newaxis]*trend_N - intercept_N  
    #         deltaTOS_ref = tos.sel(exp=ref_exp)[:150] - np.arange(0,150)*trend_SST - intercept_SST

    #         deltaTOS = (tos.sel(exp=exp)-tos.sel(exp=exp_c).mean(axis=0))
    #         deltaN = (shf.sel(exp=exp)-shf.sel(exp=exp_c).mean(axis=0))

    #         ohue = deltaN.rolling(year=40).mean()/deltaTOS.rolling(year=40).mean()

    #         #print(ts_dataset.sel(exp = exp).nav_lat)
    #         if ref_exp is not None: ohue = ohue# -ohue_ref #deltaN = deltaN -deltaN_ref
            
    #         d = ax.pcolormesh(deltaN.nav_lon, deltaN.nav_lat, ohue[70], cmap='RdBu_r', transform=ccrs.PlateCarree(), norm=divnorm)
    #         gl = ax.gridlines(draw_labels={"bottom": "x", "left": "y"}, color='gray', alpha=0.5)
    #         gl.xlabel_style = {'size': 10}
    #         gl.ylabel_style = {'size':10}    
    #         ax.set_title(exp)
    #         ax.coastlines()

    #         cb2 = plt.colorbar(d, ax=ax, extend='both', shrink=0.7)
    #         # cb2.set_ticks([0, 1e-7,1e-6,1e-5, 1e-4, 1e-3])
    #         # cb2.set_ticklabels(['0','1e-7','1e-6','1e-5','1e-4','1e-3'], fontsize=12)
    #         cb2.ax.set_ylabel(r'(W/m2/K)', fontsize=12)

    clevels = np.arange(-0.001, 0.0011, 0.0001)
    clevels = np.arange(-10,11,1)
    divnorm = mcolors.BoundaryNorm(clevels,ncolors=plt.colormaps['RdBu_r'].N,clip=True)
    c = axs[0].pcolormesh(deltaN_ref.nav_lon,deltaN_ref.nav_lat, ohue_ref30, cmap='RdBu_r',transform=ccrs.PlateCarree(),norm=divnorm)
    cb = plt.colorbar(c, ax=axs[0], extend='both', shrink=0.7)
    gl = axs[0].gridlines(draw_labels={"bottom": "x", "left": "y"}, color='gray', alpha=0.5)
    gl.xlabel_style = {'size': 10}
    gl.ylabel_style = {'size':10}
    #cb.set_ticks([0, 1e-7,1e-6,1e-5, 1e-4, 1e-3])
    #cb.set_ticklabels(['0','1e-7','1e-6','1e-5','1e-4','1e-3'], fontsize=12)
    cb.ax.set_ylabel(r'(W/m2/K)', fontsize=12)
    axs[0].set_title(ref_exp)
    axs[0].coastlines()

    c = axs[1].pcolormesh(deltaN_ref.nav_lon,deltaN_ref.nav_lat, ohue_ref70, cmap='RdBu_r',transform=ccrs.PlateCarree(),norm=divnorm)
    cb = plt.colorbar(c, ax=axs[1], extend='both', shrink=0.7)
    #cb.set_ticks([0, 1e-7,1e-6,1e-5, 1e-4, 1e-3])
    #cb.set_ticklabels(['0','1e-7','1e-6','1e-5','1e-4','1e-3'], fontsize=12)
    gl = axs[1].gridlines(draw_labels={"bottom": "x", "left": "y"}, color='gray', alpha=0.5)
    gl.xlabel_style = {'size': 10}
    gl.ylabel_style = {'size':10}
    cb.ax.set_ylabel(r'(W/m2/K)', fontsize=12)
    axs[1].set_title(ref_exp)
    axs[1].coastlines()

    clevels = np.arange(-0.001, 0.0011, 0.0001)
    clevels = np.arange(-5, 5.5,0.5)
    divnorm = mcolors.BoundaryNorm(clevels, ncolors=plt.colormaps['RdBu_r'].N, clip=True)

    c = axs[2].pcolormesh(deltaN_ref.nav_lon,deltaN_ref.nav_lat, ohue_ref70 - ohue_ref30, cmap='RdBu_r',transform=ccrs.PlateCarree(),norm=divnorm)
    cb = plt.colorbar(c, ax=axs[2], extend='both', shrink=0.7)
    gl = axs[2].gridlines(draw_labels={"bottom": "x", "left": "y"}, color='gray', alpha=0.5)
    gl.xlabel_style = {'size': 10}
    gl.ylabel_style = {'size':10}
    #cb.set_ticks([0, 1e-7,1e-6,1e-5, 1e-4, 1e-3])
    #cb.set_ticklabels(['0','1e-7','1e-6','1e-5','1e-4','1e-3'], fontsize=12)
    cb.ax.set_ylabel(r'(W/m2/K)', fontsize=12)
    axs[2].set_title(ref_exp)
    axs[2].coastlines()

    plt.show()

def compute_ohue_regions(exps, user = None, read_again = [], cart_exp = '/ec/res4/scratch/{}/ece4/', cart_out = './output/', imbalance = 0., ref_exp = None, atm_only = False, atmvars = ''.split(), ocevars = 'tos heatc qt_oce sos'.split(),  year_clim = None, plot_diffref=False, plot_param=False, param_map={}, skip_first_year=False, exp_type = 'PD', density=False):
    """
    Plots timeseries of var "vname" in domain "domain" for all exps.

    Domain is one among: ['atm', 'oce', 'ice']
    """
    if cart_out is None:
        raise ValueError('cart_out not specified!')

    cart_out_nc = cart_out + '/exps_clim/'
    window = 30 
    map_all = read_output_map(exps, user = user, read_again = read_again, cart_exp = cart_exp, cart_out = cart_out_nc, atm_only = atm_only, atmvars = atmvars, ocevars = ocevars, year_clim = year_clim, density=density)

    oce_dataset = map_all[f'oce_map']
    oce_dataset = {co: oce_dataset[co] for co in oce_dataset if oce_dataset[co] is not None}

    if exps is None: exps = oce_dataset.keys()
    oce_dataset = create_ds_exp(oce_dataset)

    if isinstance(oce_dataset, xr.Dataset):

        shf = oce_dataset['qt_oce']

    clim_all = read_output(exps, user = user, read_again = read_again, cart_exp = cart_exp, cart_out = cart_out_nc, atm_only = atm_only, atmvars = atmvars, ocevars = ocevars, icevars = [], year_clim = year_clim, density=density)
    
    oce_mean = clim_all[f'oce_mean']
    oce_mean = {co: oce_mean[co] for co in oce_mean if oce_mean[co] is not None}

    oce_mean = create_ds_exp(oce_mean)

    if isinstance(oce_mean, xr.Dataset):

        tos = oce_mean['tos']
        #test_shf = oce_mean['qt_oce']

    areas = get_areas_nemo(exps[0], 'itcv', cart_exp = cart_exp)
    mask = get_mask_nemo(exps[0], 'itcv', cart_exp = cart_exp, grid = 'T')
    tot_area = np.nansum(areas*mask)

    areas_ds = shf.sel(exp=exps[0])[0].copy()
    areas_ds.values = areas*mask
    
    trend_SHF = np.zeros(np.shape(areas))
    intercept_SHF = np.zeros(np.shape(areas))

    for i in range(np.shape(areas)[0]):
        for j in range(np.shape(areas)[1]):

            trend_SHF[i,j], intercept_SHF[i,j], _, _, _ = stats.linregress(np.arange(0,np.shape(shf.sel(exp=exps[1]))[0]), shf.sel(exp=exps[1])[:,i,j])

    deltaN = (shf.sel(exp=exps[0]) - (np.arange(0,300))[:, np.newaxis, np.newaxis]*trend_SHF-intercept_SHF).rolling(year=window).mean()
    #deltaN = (shf.sel(exp=exps[0])-shf.sel(exp=exps[1]).mean(axis=0))#.rolling(year=40).mean()

    trend_TOS, intercept_TOS, _, _, _ = stats.linregress(np.arange(0,np.shape(tos.sel(exp=exps[1]))[0]), tos.sel(exp=exps[1]))
    deltaTOS = (tos.sel(exp=exps[0]) - trend_TOS*np.arange(0,len(tos.sel(exp=exps[0])))-intercept_TOS)

    #deltaTOS = (tos.sel(exp=exps[0])-tos.sel(exp=exps[1]).mean(axis=0))
    weight = (tot_area*deltaTOS).rolling(year=window).mean()

    #OHUE all 
    area_reg = areas_ds
    ohue_all = (deltaN*area_reg).sum(axis=(1,2))/weight

    #plt.plot(aa.rolling(year=40).mean()- (deltaN*area_reg).sum(axis=(1,2))/tot_area)

    #OHUE Smidlatitudes     
    area_reg = areas_ds.where(((shf.nav_lat < -30) & (shf.nav_lat > -60)), np.nan)
    ohue_smid = (deltaN*area_reg).sum(axis=(1,2))/weight

    #OHUE Nmidlatitudes 
    area_reg = areas_ds.where(((shf.nav_lat < 60) & (shf.nav_lat > 30)), np.nan)
    ohue_nmid = (deltaN*area_reg).sum(axis=(1,2))/weight

    #OHUE lowlatitudes
    area_reg = areas_ds.where(((shf.nav_lat < 30) & (shf.nav_lat > -30)), np.nan)
    ohue_low = (deltaN*area_reg).sum(axis=(1,2))/weight

    #OHUE Shighlatitudes
    area_reg = areas_ds.where(((shf.nav_lat < -60)), np.nan)
    ohue_shigh = (deltaN*area_reg).sum(axis=(1,2))/weight

    #OHUE Shighlatitudes
    area_reg = areas_ds.where(((shf.nav_lat > 60))) #, np.nan)
    ohue_nhigh = (deltaN*area_reg).sum(axis=(1,2))/weight

    return ohue_smid, ohue_nmid, ohue_low, ohue_shigh, ohue_nhigh, ohue_all

def compute_ohue_zonal(exps, user = None, read_again = [], cart_exp = '/ec/res4/scratch/{}/ece4/', cart_out = './output/', imbalance = 0., ref_exp = None, atm_only = False, atmvars = 'rsut rlut rsdt tas pr'.split(), ocevars = 'tos heatc qt_oce sos'.split(),  year_clim = None, plot_diffref=False, plot_param=False, param_map={}, skip_first_year=False, exp_type = 'PD', density=False):
    """
    Plots timeseries of var "vname" in domain "domain" for all exps.

    Domain is one among: ['atm', 'oce', 'ice']
    """
    if cart_out is None:
        raise ValueError('cart_out not specified!')
    window = 30
    cart_out_nc = cart_out + '/exps_clim/'

    map_all = read_output_map(exps, user = user, read_again = read_again, cart_exp = cart_exp, cart_out = cart_out_nc, atm_only = atm_only, atmvars = atmvars, ocevars = ocevars, year_clim = year_clim, density=density)

    oce_dataset = map_all[f'oce_map']
    oce_dataset = {co: oce_dataset[co] for co in oce_dataset if oce_dataset[co] is not None}

    if exps is None: exps = oce_dataset.keys()
    oce_dataset = create_ds_exp(oce_dataset)

    if isinstance(oce_dataset, xr.Dataset):

        shf = oce_dataset['qt_oce']

    clim_all = read_output(exps, user = user, read_again = read_again, cart_exp = cart_exp, cart_out = cart_out_nc, atm_only = atm_only, atmvars = atmvars, ocevars = ocevars, icevars = [], year_clim = year_clim, density=density)
    
    oce_mean = clim_all[f'oce_mean']
    oce_mean = {co: oce_mean[co] for co in oce_mean if oce_mean[co] is not None}

    oce_mean = create_ds_exp(oce_mean)

    if isinstance(oce_mean, xr.Dataset):

        tos = oce_mean['tos']
        #test_shf = oce_mean['qt_oce']

    areas = get_areas_nemo(exps[0], 'itcv', cart_exp = cart_exp)
    mask = get_mask_nemo(exps[0], 'itcv', cart_exp = cart_exp, grid = 'T')
    tot_area = np.nansum(areas*mask)

    areas_ds = shf.sel(exp=exps[0])[0].copy()
    areas_ds.values = areas*mask
    
    trend_SHF = np.zeros(np.shape(areas))
    intercept_SHF = np.zeros(np.shape(areas))

    for i in range(np.shape(areas)[0]):
        for j in range(np.shape(areas)[1]):

            trend_SHF[i,j], intercept_SHF[i,j], _, _, _ = stats.linregress(np.arange(0,np.shape(shf.sel(exp=exps[1]))[0]), shf.sel(exp=exps[1])[:,i,j])

    deltaN = (shf.sel(exp=exps[0]) - (np.arange(0,np.shape(shf.sel(exp=exps[1]))[0]))[:, np.newaxis, np.newaxis]*trend_SHF-intercept_SHF).rolling(year=window).mean()

    trend_TOS, intercept_TOS, _, _, _ = stats.linregress(np.arange(0,np.shape(tos.sel(exp=exps[1]))[0]), tos.sel(exp=exps[1]))
    deltaTOS = (tos.sel(exp=exps[0]) - trend_TOS*np.arange(0,len(tos.sel(exp=exps[0])))-intercept_TOS)

    weight = (tot_area*deltaTOS).rolling(year=window).mean()

    # zonal mean
    ohue_zonal, lats = zonal_mean_irregular_xarray(deltaN, deltaN.nav_lat, True)
    ohue_zonal_xr = xr.DataArray(ohue_zonal, dims=("year", "lat"), coords={"year": deltaTOS.year, 'lat':lats})

    deltaTOS_smoothed = deltaTOS.rolling(year=window).mean()#.to_numpy()
    ohue = ohue_zonal_xr / deltaTOS_smoothed#[:,None]

    # zonal decomposition - need to fix for irregular grid!
    
    nlat=90
    lats = np.linspace(-89,89,nlat)
    step = (lats[1]-lats[0])/2 + 0.16

    ohue_zonal2 = np.zeros([deltaN.shape[0], nlat])

    for i, lat in enumerate(lats):
        temp = deltaN
        area_reg = areas_ds.where(((deltaN.nav_lat < lat+step) & (deltaN.nav_lat > lat-step)), np.nan)

        ohue_zonal2[:,i] = (temp*area_reg).sum(axis=(1,2))/weight
    
    ohue_zonal2_xr = xr.DataArray(ohue_zonal2, dims=("year", "lat"), coords={"year": deltaTOS.year, 'lat':lats})

    return ohue, ohue_zonal2_xr

def plot_ohue_ts(exps, user = None, read_again = [], cart_exp = '/ec/res4/scratch/{}/ece4/', cart_out = './output/', atm_only = False, atmvars = 'rsut rlut rsdt tas pr'.split(), ocevars = 'tos heatc qt_oce sos'.split(), icevars = 'siconc sivolu sithic'.split(), year_clim = None, rolling = 30, density=False, ax = None, color=None):

    """
    Plots timeseries of var "vname" in domain "domain" for all exps.

    Domain is one among: ['atm', 'oce', 'ice']
    """
    if cart_out is None:
        raise ValueError('cart_out not specified!')
    
    cart_out_nc = cart_out + '/exps_clim/'

    clim_all = read_output(exps, user = user, read_again = read_again, cart_exp = cart_exp, cart_out = cart_out_nc, atm_only = atm_only, atmvars = atmvars, ocevars = ocevars, icevars = icevars, year_clim = year_clim, density=density)

    toa_dataset = clim_all[f'atm_mean']
    oce_dataset = clim_all[f'oce_mean']

    toa_dataset = {co: toa_dataset[co] for co in toa_dataset if toa_dataset[co] is not None}
    oce_dataset = {co: oce_dataset[co] for co in oce_dataset if oce_dataset[co] is not None}

    if exps is None: exps = toa_dataset.keys()
    toa_dataset = create_ds_exp(toa_dataset)

    if exps is None: exps = oce_dataset.keys()
    oce_dataset = create_ds_exp(oce_dataset)

    if isinstance(toa_dataset, xr.Dataset):
        toa = toa_dataset['rsdt'] - toa_dataset[ 'rsut']- toa_dataset[ 'rlut']
        tas = toa_dataset['tas']

        tos = oce_dataset['tos']
        shf = oce_dataset['qt_oce']
 
    
    # anomalies with respect to detrended pi
    trend_N, intercept_N, _, _, _ = stats.linregress(np.arange(0,len(toa.sel(exp=exps[1]))), toa.sel(exp=exps[1]))
    trend_SST, intercept_SST, _,_,_ = stats.linregress(np.arange(0,len(toa.sel(exp=exps[1]))), tas.sel(exp=exps[1]))
    deltaN = (toa.sel(exp=exps[0]) - trend_N*np.arange(0,len(toa.sel(exp=exps[0])))-intercept_N)
    deltaSST = (tas.sel(exp=exps[0]) - trend_SST*np.arange(0,len(toa.sel(exp=exps[0])))-intercept_SST)

    #1. year to year differences
    y4 = (deltaN/deltaSST)
    ax[0,0].plot(np.arange(len(y4)), y4.rolling(year = rolling).mean(), label = exps[0], color = color)
    ax[0,0].set_title('')

    #2. 20-year mean window
    ohue_atm_ts, ohue_ts = compute_ohue(exps, 2*['itcv'],cart_exp=cart_exp, cart_out=cart_out, window=rolling)
    ax[0,1].plot(np.arange(len(ohue_atm_ts)), ohue_atm_ts, color = color)
    ax[0,1].set_title('')

    #3. rolling regression
    y6 = RollingOLS(deltaN[:150].values, deltaSST[:150].values, window=rolling)
    rres= y6.fit()

    ax[0,2].plot(rres.params,label = exps[0], color = color)

    #ocean based estimate
    #1. year to year differences
    trend_SHF, intercept_SHF, _, _, _ = stats.linregress(np.arange(0,300), shf.sel(exp=exps[1]))
    trend_TOS, intercept_TOS, _,_,_ = stats.linregress(np.arange(0,300), tos.sel(exp=exps[1]))
    deltaSHF = (shf.sel(exp=exps[0])[:150] - trend_SHF*np.arange(0,150)-intercept_SHF)
    deltaTOS = (tos.sel(exp=exps[0])[:150] - trend_TOS*np.arange(0,150)-intercept_TOS)

    y7 = (deltaSHF/deltaTOS)
    ax[1,0].plot(np.arange(len(y7)), y7.rolling(year = rolling).mean(), label = exps[0], color = color)
    ax[1,0].set_title('')

    #2. 20-year mean window
    ax[1,1].plot(np.arange(len(ohue_ts)), ohue_ts, color = color)
    ax[1,1].set_title('')

    #3. rolling regression
    y9 = RollingOLS(deltaSHF.values, deltaTOS.values, window=rolling)
    rres= y9.fit()

    ax[1,2].plot(rres.params,label = exps[0], color = color)


def compute_radiative_forcing():

    alpha = 5.35
    C0 = 284.32

    C = np.zeros(150)
    C[0]= C0
    
    for i in range(1,150):
        C[i]= C[i-1]*1.01

    forcing = alpha*np.log(C/C0)

    return forcing

def compute_lambda(exps, user = None, window = 30,read_again = [], cart_exp = '/ec/res4/scratch/{}/ece4/', cart_out = './output/', imbalance = 0., ref_exp = None, atm_only = False, atmvars = 'rsut rlut rsdt tas pr'.split(), ocevars = 'tos heatc qt_oce sos'.split(), icevars = 'siconc sivolu sithic'.split(), year_clim = None, plot_diffref=False, plot_param=False, density=False):

    if cart_out is None:
        raise ValueError('cart_out not specified!')

    cart_out_nc = cart_out + '/exps_clim/'

    clim_all = read_output(exps, user = user, read_again = read_again, cart_exp = cart_exp, cart_out = cart_out_nc, atm_only = atm_only, atmvars = atmvars, ocevars = ocevars, icevars = icevars, year_clim = year_clim, density=density)

    toa_dataset = clim_all[f'atm_mean']

    toa_dataset = {co: toa_dataset[co] for co in toa_dataset if toa_dataset[co] is not None}

    if exps is None: exps = toa_dataset.keys()
    toa_dataset = create_ds_exp(toa_dataset)

    if isinstance(toa_dataset, xr.Dataset):
        toa = toa_dataset['rsdt'] - toa_dataset[ 'rsut']- toa_dataset[ 'rlut']
        tas = toa_dataset['tas']
 
    trend_N, intercept_N, _, _, _ = stats.linregress(np.arange(0,300), toa.sel(exp=exps[1]))
    trend_SST, intercept_SST, _,_,_ = stats.linregress(np.arange(0,300), tas.sel(exp=exps[1]))
    deltaN = (toa.sel(exp=exps[0])[:150] - trend_N*np.arange(0,150)-intercept_N)
    deltaSST = (tas.sel(exp=exps[0])[:150] - trend_SST*np.arange(0,150)-intercept_SST)

    try: 
        ds_forcing = xr.open_mfdataset('/ec/res4/hpcperm/itcv/analysis/forcing_1pct/'+exps[0]+'_forcing.nc')
        forcing = global_mean(ds_forcing.forcing)
    except:
        forcing = compute_radiative_forcing()

    feedback  = (deltaN-forcing).sel(year =slice(1850+55, 1850+85)).mean(axis=0)/deltaSST.sel(year =slice(1850+55, 1850+85)).mean(axis=0)
    feedback_ts  = (deltaN-forcing).rolling(year =window).mean(axis=0)/deltaSST.rolling(year =window).mean(axis=0)
    
    return feedback_ts
    #return feedback, (deltaN-forcing).sel(year =slice(1850+55, 1850+85)).mean(axis=0), deltaSST.sel(year =slice(1850+55, 1850+85)).mean(axis=0)
    #return deltaN, deltaSST, forcing

def compute_alpha_abrupt(exps, user = None, read_again = [], cart_exp = '/ec/res4/scratch/{}/ece4/', cart_out = './output/', imbalance = 0., ref_exp = None, atm_only = False, atmvars = 'rsut rlut rsdt tas pr'.split(), ocevars = 'tos heatc qt_oce sos'.split(), icevars = 'siconc sivolu sithic'.split(), year_clim = None, plot_diffref=False, plot_param=False, rolling=40, density=False):

    if cart_out is None:
        raise ValueError('cart_out not specified!')

    cart_out_nc = cart_out + '/exps_clim/'

    clim_all = read_output(exps, user = user, read_again = read_again, cart_exp = cart_exp, cart_out = cart_out_nc, atm_only = atm_only, atmvars = atmvars, ocevars = ocevars, icevars = icevars, year_clim = year_clim, density=density)

    toa_dataset = clim_all[f'atm_mean']

    toa_dataset = {co: toa_dataset[co] for co in toa_dataset if toa_dataset[co] is not None}

    if exps is None: exps = toa_dataset.keys()
    toa_dataset = create_ds_exp(toa_dataset)

    if isinstance(toa_dataset, xr.Dataset):
        toa = toa_dataset['rsdt'] - toa_dataset[ 'rsut']- toa_dataset[ 'rlut']
        tas = toa_dataset['tas']
 
    trend_N, intercept_N, _, _, _ = stats.linregress(np.arange(0,300), toa.sel(exp=exps[1]))
    trend_SST, intercept_SST, _,_,_ = stats.linregress(np.arange(0,300), tas.sel(exp=exps[1]))
    deltaN = (toa.sel(exp=exps[0])[:150] - trend_N*np.arange(0,150)-intercept_N)
    deltaSST = (tas.sel(exp=exps[0])[:150] - trend_SST*np.arange(0,150)-intercept_SST)

    feedback  = stats.linregress(deltaSST.sel(year =slice(1850,1870)),(deltaN).sel(year =slice(1850,1870)))[0]

    return feedback, deltaN, deltaSST

    
def compute_rad_components(exps, user = None, read_again = [], cart_exp = '/ec/res4/scratch/{}/ece4/', cart_out = './output/', imbalance = 0., ref_exp = None, atm_only = True, atmvars = 'rsut rlut rsdt'.split(), ocevars = 'tos heatc qt_oce sos'.split(), icevars = 'siconc sivolu sithic'.split(), year_clim = None, plot_diffref=False, plot_param=False, rolling=40, density=False):

    if cart_out is None:
        raise ValueError('cart_out not specified!')

    cart_out_nc = cart_out + '/exps_clim/'

    # read global mean values for netTOA radiation
    clim_all = read_output(exps, user = user, read_again = read_again, cart_exp = cart_exp, cart_out = cart_out_nc, atm_only = atm_only, atmvars = atmvars, ocevars = ocevars, icevars = icevars, year_clim = year_clim, density=density)
    toa_dataset = clim_all[f'atm_mean']
    toa_dataset = {co: toa_dataset[co] for co in toa_dataset if toa_dataset[co] is not None}

    if exps is None: exps = toa_dataset.keys()
    toa_dataset = create_ds_exp(toa_dataset)

    if isinstance(toa_dataset, xr.Dataset):
        toa = toa_dataset['rsdt'] - toa_dataset[ 'rsut']- toa_dataset[ 'rlut']

    bias_pi = toa.sel(exp=exps[1])
    trend_N, intercept_N, _, _, _ = stats.linregress(np.arange(0,150), bias_pi)
    deltaN = toa.sel(exp=exps[0]) - np.arange(0,150)*trend_N-intercept_N

    forcing = compute_radiative_forcing()

    return deltaN, forcing, bias_pi

def plot_lambda_ts(exps, user = None, read_again = [], cart_exp = '/ec/res4/scratch/{}/ece4/', cart_out = './output/', atm_only = False, atmvars = 'rsut rlut rsdt tas pr'.split(), ocevars = 'tos heatc qt_oce sos'.split(), icevars = 'siconc sivolu sithic'.split(), year_clim = None, rolling = 30, density=False, ax = None, color=None):

    """
    Plots timeseries of var "vname" in domain "domain" for all exps.

    Domain is one among: ['atm', 'oce', 'ice']
    """
    if cart_out is None:
        raise ValueError('cart_out not specified!')
    
    cart_out_nc = cart_out + '/exps_clim/'

    clim_all = read_output(exps, user = user, read_again = read_again, cart_exp = cart_exp, cart_out = cart_out_nc, atm_only = atm_only, atmvars = atmvars, ocevars = ocevars, icevars = icevars, year_clim = year_clim, density=density)

    toa_dataset = clim_all[f'atm_mean']

    toa_dataset = {co: toa_dataset[co] for co in toa_dataset if toa_dataset[co] is not None}

    if exps is None: exps = toa_dataset.keys()
    toa_dataset = create_ds_exp(toa_dataset)

    if isinstance(toa_dataset, xr.Dataset):
        toa = toa_dataset['rsdt'] - toa_dataset[ 'rsut']- toa_dataset[ 'rlut']
        tas = toa_dataset['tas']
 
    #anomalies with respect to pi mean
    trend_N, intercept_N, _, _, _ = stats.linregress(np.arange(0,300), toa.sel(exp=exps[1]))
    trend_SST, intercept_SST, _,_,_ = stats.linregress(np.arange(0,300), tas.sel(exp=exps[1]))
    deltaN = (toa.sel(exp=exps[0])[:150] - trend_N*np.arange(0,150)-intercept_N)
    deltaSST = (tas.sel(exp=exps[0])[:150] - trend_SST*np.arange(0,150)-intercept_SST)

    try:
        ds_forcing = xr.open_mfdataset('/ec/res4/hpcperm/itcv/analysis/forcing_1pct/'+exps[0]+'_forcing.nc')
        forcing = global_mean(ds_forcing.forcing)
    except:
        forcing = compute_radiative_forcing()
    
    #1. year to year differences
    feedback  = (deltaN-forcing)/deltaSST
    y1 = feedback
    ax[0].plot(y1.rolling(year = rolling).mean(), label = exps[0], color = color)
    ax[0].set_title(r'Year-to-year $\lambda$')

    #2. 20-year mean window
    lambda_ts = compute_lambda(exps, 2*['itcv'],cart_exp=cart_exp, cart_out=cart_out) 
    ax[1].plot(lambda_ts, label = exps[0], color = color)
    ax[1].set_title(str(rolling)+r'-yr mean $\lambda$')

    #3. rolling regression
    y3 = RollingOLS((deltaN-forcing).values, deltaSST.values, window=rolling)
    rres= y3.fit()

    ax[2].plot(rres.params,label = exps[0], color = color)
    ax[2].set_title('Rolling regression ('+str(rolling)+'-yr window)')


def compute_lambda_regions(exps, user = None, read_again = [], cart_exp = '/ec/res4/scratch/{}/ece4/', cart_out = './output/', imbalance = 0., ref_exp = None, atm_only = False, atmvars = 'rsut rlut rsdt tas'.split(), ocevars = 'tos heatc qt_oce sos'.split(),  year_clim = None, plot_diffref=False, plot_param=False, param_map={}, skip_first_year=False, exp_type = 'PD', density=False):
    """
    Plots timeseries of var "vname" in domain "domain" for all exps.

    Domain is one among: ['atm', 'oce', 'ice']
    """
    if cart_out is None:
        raise ValueError('cart_out not specified!')

    cart_out_nc = cart_out + '/exps_clim/'
    window = 30
    map_all = read_output_map(exps, user = user, read_again = read_again, cart_exp = cart_exp, cart_out = cart_out_nc, atm_only = atm_only, atmvars = atmvars, ocevars = ocevars, year_clim = year_clim, density=density)

    toa_dataset = map_all[f'atm_map']
    toa_dataset = {co: toa_dataset[co] for co in toa_dataset if toa_dataset[co] is not None}

    if exps is None: exps = toa_dataset.keys()
    toa_dataset = create_ds_exp(toa_dataset)

    if isinstance(toa_dataset, xr.Dataset):
        toa = toa_dataset['rsdt'] - toa_dataset[ 'rsut']- toa_dataset[ 'rlut']

    clim_all = read_output(exps, user = user, read_again = read_again, cart_exp = cart_exp, cart_out = cart_out_nc, atm_only = atm_only, atmvars = atmvars, ocevars = ocevars, icevars = [], year_clim = year_clim, density=density)
    
    tas_mean = clim_all[f'atm_mean']
    tas_mean = {co: tas_mean[co] for co in tas_mean if tas_mean[co] is not None}
    tas_mean = create_ds_exp(tas_mean)
    tas = tas_mean['tas']
    netTOA = tas_mean['rsdt'] - tas_mean['rsut'] - tas_mean['rlut']
    
    #forcing = compute_radiative_forcing()
    ds_forcing = xr.open_mfdataset('/ec/res4/hpcperm/itcv/analysis/forcing_1pct/'+exps[0]+'_forcing.nc')
    forcing = ds_forcing.forcing

    log_forcing = compute_radiative_forcing()

    trend_Ng, intercept_Ng, _, _, _ = stats.linregress(np.arange(0,300), netTOA.sel(exp=exps[1]))

    trend_N = np.zeros([90,180])
    intercept_N = np.zeros([90,180])

    for i in range(np.shape(toa.sel(exp=exps[1]))[1]):
        for j in range(np.shape(toa.sel(exp=exps[1]))[2]):

            trend_N[i,j], intercept_N[i,j], _, _, _ = stats.linregress(np.arange(0,np.shape(toa.sel(exp=exps[1]))[0]), toa.sel(exp=exps[1])[:,i,j])

    trend_SST, intercept_SST, _,_,_ = stats.linregress(np.arange(0,len(tas.sel(exp=exps[1]))), tas.sel(exp=exps[1]))
    deltaR = toa.sel(exp=exps[0])[:150] - np.arange(0,150)[:,np.newaxis, np.newaxis]*trend_N - intercept_N - forcing #[:,np.newaxis, np.newaxis] 
    deltaT = tas.sel(exp=exps[0])[:150] - np.arange(0,150)*trend_SST - intercept_SST

    weights = np.cos(np.deg2rad(deltaR.lat))
    tot_weights = weights.sum()*deltaR.lon.size

    # Test for consistency with previous estimates
    netTOA_s = netTOA.sel(exp=exps[0])[:150] - np.arange(0,150)*trend_Ng - intercept_Ng - global_mean(forcing)
    netTOA_s2 = netTOA.sel(exp=exps[0])[:150] - np.arange(0,150)*trend_Ng - intercept_Ng - log_forcing

    lamdba_true = netTOA_s.rolling(year=window).mean()/deltaT.rolling(year=window).mean()
    lamdba_true2 = netTOA_s2.rolling(year=window).mean()/deltaT.rolling(year=window).mean()

    #OHUE all 
    lambda_all = (deltaR*weights).rolling(year=window).mean().sum(axis=(1,2))/(deltaT.rolling(year=window).mean()*tot_weights)
    
    #OHUE Smidlatitudes     
    deltaN_reg = deltaR.where(((deltaR.lat < -30) & (deltaR.lat > -60)), np.nan)
    weights_reg = weights.where(((deltaR.lat < -30) & (deltaR.lat > -60)), 0)
    lambda_smid = (deltaN_reg*weights_reg).rolling(year=window).mean().sum(axis=(1,2))/(deltaT.rolling(year=window).mean()*tot_weights)

    # #OHUE Nmidlatitudes 
    deltaN_reg = deltaR.where(((deltaR.lat < 60) & (deltaR.lat > 30)), np.nan)
    weights_reg = weights.where(((deltaR.lat < 60) & (deltaR.lat > 30)), 0)
    lambda_nmid = (deltaN_reg*weights_reg).rolling(year=window).mean().sum(axis=(1,2))/(deltaT.rolling(year=window).mean()*tot_weights)

    # #OHUE lowlatitudes
    deltaN_reg = deltaR.where(((deltaR.lat < 30) & (deltaR.lat > -30)), np.nan)
    weights_reg = weights.where(((deltaR.lat < 30) & (deltaR.lat > -30)), 0)
    lambda_low = (deltaN_reg*weights_reg).rolling(year=window).mean().sum(axis=(1,2))/(deltaT.rolling(year=window).mean()*tot_weights)

    # #OHUE Shighlatitudes
    deltaN_reg = deltaR.where(((deltaR.lat < -60)), np.nan)
    weights_reg = weights.where(((deltaR.lat < -60)), 0)
    lambda_shigh = (deltaN_reg*weights_reg).rolling(year=window).mean().sum(axis=(1,2))/(deltaT.rolling(year=window).mean()*tot_weights)

    # #OHUE Shighlatitudes
    deltaN_reg = deltaR.where(((deltaR.lat > 60)), np.nan)
    weights_reg = weights.where(((deltaR.lat > 60)),0)
    lambda_nhigh = (deltaN_reg*weights_reg).rolling(year=window).mean().sum(axis=(1,2))/(deltaT.rolling(year=window).mean()*tot_weights)

    return lambda_smid, lambda_nmid, lambda_low, lambda_shigh, lambda_nhigh, lambda_all , lamdba_true, lamdba_true2

def compute_lambda_zonal(exps, user = None, read_again = [], cart_exp = '/ec/res4/scratch/{}/ece4/', cart_out = './output/', imbalance = 0., ref_exp = None, atm_only = False, atmvars = 'rsut rlut rsdt tas'.split(), ocevars = 'tos heatc qt_oce sos'.split(),  year_clim = None, plot_diffref=False, plot_param=False, param_map={}, skip_first_year=False, exp_type = 'PD', density=False):
    """
    Plots timeseries of var "vname" in domain "domain" for all exps.

    Domain is one among: ['atm', 'oce', 'ice']
    """
    if cart_out is None:
        raise ValueError('cart_out not specified!')

    cart_out_nc = cart_out + '/exps_clim/'
    window = 30
    map_all = read_output_map(exps, user = user, read_again = read_again, cart_exp = cart_exp, cart_out = cart_out_nc, atm_only = atm_only, atmvars = atmvars, ocevars = ocevars, year_clim = year_clim, density=density)

    toa_dataset = map_all[f'atm_map']
    toa_dataset = {co: toa_dataset[co] for co in toa_dataset if toa_dataset[co] is not None}

    if exps is None: exps = toa_dataset.keys()
    toa_dataset = create_ds_exp(toa_dataset)

    if isinstance(toa_dataset, xr.Dataset):
        toa = toa_dataset['rsdt'] - toa_dataset[ 'rsut']- toa_dataset[ 'rlut']

    clim_all = read_output(exps, user = user, read_again = read_again, cart_exp = cart_exp, cart_out = cart_out_nc, atm_only = atm_only, atmvars = atmvars, ocevars = ocevars, icevars = [], year_clim = year_clim, density=density)
    
    tas_mean = clim_all[f'atm_mean']
    tas_mean = {co: tas_mean[co] for co in tas_mean if tas_mean[co] is not None}
    tas_mean = create_ds_exp(tas_mean)
    tas = tas_mean['tas']
    netTOA = tas_mean['rsdt'] - tas_mean['rsut'] - tas_mean['rlut']
    
    #forcing = compute_radiative_forcing()
    ds_forcing = xr.open_mfdataset('/ec/res4/hpcperm/itcv/analysis/forcing_1pct/'+exps[0]+'_forcing.nc')
    forcing = ds_forcing.forcing
    
    trend_N = np.zeros([90,180])
    intercept_N = np.zeros([90,180])

    for i in range(np.shape(toa.sel(exp=exps[1]))[1]):
        for j in range(np.shape(toa.sel(exp=exps[1]))[2]):

            trend_N[i,j], intercept_N[i,j], _, _, _ = stats.linregress(np.arange(0,np.shape(toa.sel(exp=exps[1]))[0]), toa.sel(exp=exps[1])[:,i,j])

    trend_SST, intercept_SST, _,_,_ = stats.linregress(np.arange(0,len(tas.sel(exp=exps[1]))), tas.sel(exp=exps[1]))
    deltaR = toa.sel(exp=exps[0])[:150] - np.arange(0,150)[:,np.newaxis, np.newaxis]*trend_N - intercept_N - forcing #[:,np.newaxis, np.newaxis] 
    deltaT = tas.sel(exp=exps[0])[:150] - np.arange(0,150)*trend_SST - intercept_SST

    #deltaR = toa.sel(exp=exps[0])[:150] -toa.sel(exp=exps[1]).mean(axis=0) - forcing #[:,np.newaxis, np.newaxis] 
    #deltaT = tas.sel(exp=exps[0])[:150] -tas.sel(exp=exps[1]).mean(axis=0)
    
    weights = np.cos(np.deg2rad(deltaR.lat))
    tot_weights = weights.sum()*deltaR.lon.size

    # zonal mean
    zonalR = deltaR.mean(dim='lon')
    zonal_lambda = zonalR.rolling(year=window).mean()/(deltaT.rolling(year=window).mean())

    # zonal decomposition
    zonalR2 = deltaR*weights
    zonal_lambda2 = zonalR2.rolling(year=window).mean().sum(axis=(2))/(deltaT.rolling(year=window).mean()*tot_weights)
    return zonal_lambda, zonal_lambda2

def plot_lambda_map(exps, user = None, read_again = [], cart_exp = '/ec/res4/scratch/{}/ece4/', cart_out = './output/',ref_exp_c=None, ref_exp = None, atm_only = False, atmvars = 'rsut rlut rsdt tas rlnt rlntcs rsnt rsntcs'.split(), ocevars = 'tos heatc qt_oce sos'.split(),  year_clim = None, density=False):
    
    """
    Plots timeseries of var "vname" in domain "domain" for all exps.

    Domain is one among: ['atm', 'oce', 'ice']
    """
    if cart_out is None:
        raise ValueError('cart_out not specified!')

    cart_out_nc = cart_out + '/exps_clim/'
    nx = int(1)
    ny = int(3)
    fig, axes = plt.subplots(nx, ny, figsize = (18, 5), subplot_kw={'projection': ccrs.PlateCarree()}) 

    map_all = read_output_map(exps, user = user, read_again = read_again, cart_exp = cart_exp, cart_out = cart_out_nc, atm_only = atm_only, atmvars = atmvars, ocevars = ocevars, year_clim = year_clim, density=density)
    #map_con = read_output(controls, user = user, read_again = read_again, cart_exp = cart_exp, cart_out = cart_out_nc, atm_only = atm_only, atmvars = atmvars, ocevars = ocevars, year_clim = year_clim, density=density)

    atm_dataset = map_all[f'atm_map']
    atm_dataset = {co: atm_dataset[co] for co in atm_dataset if atm_dataset[co] is not None}
    atm_dataset = create_ds_exp(atm_dataset)
    toa = atm_dataset['rsdt'] - atm_dataset['rlut'] - atm_dataset['rsut']
    #tas = atm_dataset['tas']
    
    clim_all = read_output(exps, user = user, read_again = read_again, cart_exp = cart_exp, cart_out = cart_out_nc, atm_only = atm_only, atmvars = atmvars, ocevars = ocevars, icevars = [], year_clim = year_clim, density=density)
    atm_mean = clim_all[f'atm_mean']
    atm_mean = {co: atm_mean[co] for co in atm_mean if atm_mean[co] is not None}
    atm_mean = create_ds_exp(atm_mean)
    tas = atm_mean['tas']

    #forcing = compute_radiative_forcing()
    ds_forcing = xr.open_mfdataset('/ec/res4/hpcperm/itcv/analysis/forcing_1pct/'+exps[0]+'_forcing.nc')
    forcing = ds_forcing.forcing

    year1 = 1870
    year2 = 1960

    if ref_exp is not None:
        trend_N = np.zeros([90,180])
        intercept_N = np.zeros([90,180])

        trend_T = np.zeros([90,180])
        intercept_T = np.zeros([90,180])

        for i in range(np.shape(toa.sel(exp=ref_exp_c))[1]):
            for j in range(np.shape(toa.sel(exp=ref_exp_c))[2]):

                trend_N[i,j], intercept_N[i,j], _, _, _ = stats.linregress(np.arange(0,np.shape(toa.sel(exp=ref_exp_c))[0]), toa.sel(exp=ref_exp_c)[:,i,j])
                #trend_T[i,j], intercept_T[i,j], _, _, _ = stats.linregress(np.arange(0,np.shape(tas.sel(exp=ref_exp_c))[0]), tas.sel(exp=ref_exp_c)[:,i,j])

        trend_T, intercept_T, _,_,_ = stats.linregress(np.arange(0,len(tas.sel(exp=ref_exp))), tas.sel(exp=ref_exp_c))
        deltaN_ref = toa.sel(exp=ref_exp)[:150] - np.arange(0,150)[:,np.newaxis, np.newaxis]*trend_N - intercept_N - forcing #[:,np.newaxis, np.newaxis] 
        deltaTAS_ref = tas.sel(exp=ref_exp)[:150] - np.arange(0,150)*trend_T - intercept_T
        
        weights = np.cos(np.deg2rad(deltaN_ref.lat))
        tot_weights = weights.sum()*deltaN_ref.lon.size
        
        lambda_ref30 = np.zeros([90,180])
        lambda_ref70 = np.zeros([90,180])

        deltaT = deltaTAS_ref.sel(year=slice(year1,year1+40)).mean(axis=0)#*tot_weights
        deltaT2 = deltaTAS_ref.sel(year=slice(year2,year2+40)).mean(axis=0)#*tot_weights 

        # for j in range(90):
        #         for k in range(180):
        #             lambda_ref30[j,k] = stats.linregress(deltaTAS_ref.sel(year=slice(year1,year1+40)), deltaN_ref.sel(year=slice(year1,year1+40))[:,j,k])[0]
        #             lambda_ref70[j,k] = stats.linregress(deltaTAS_ref.sel(year=slice(year2,year2+40)), deltaN_ref.sel(year=slice(year2,year2+40))[:,j,k])[0]
        
        lambda_ref30 = (deltaN_ref).sel(year=slice(year1,year1+40)).mean(axis=0)/ deltaT
        lambda_ref70 = (deltaN_ref).sel(year=slice(year2,year2+40)).mean(axis=0)/ deltaT2

        lambda_ts = deltaN_ref.rolling(year=30).mean()/deltaTAS_ref.rolling(year=30).mean()
        lambda_ts.to_netcdf(f'/ec/res4/hpcperm/itcv/analysis/exps_clim/KL/lambda_ts_{ref_exp}_.nc')

    clevels = np.arange(-0.001, 0.0011,0.0001)
    clevels = np.arange(-5, 5.5,0.5)
    divnorm = mcolors.BoundaryNorm(clevels, ncolors=plt.colormaps['RdBu_r'].N, clip=True)
    
    ax = axes[0]
    d = ax.pcolormesh(deltaN_ref.lon, deltaN_ref.lat, lambda_ref30, cmap='RdBu_r', transform=ccrs.PlateCarree(), norm=divnorm)
    gl = ax.gridlines(draw_labels={"bottom": "x", "left": "y"}, color='gray', alpha=0.5)
    gl.xlabel_style = {'size': 10}
    gl.ylabel_style = {'size':10}    
    ax.set_title(ref_exp+' years 20-60')
    ax.coastlines()

    cb2 = plt.colorbar(d, ax=ax, extend='both', shrink=0.7)
    cb2.ax.set_ylabel(r'(W/m2/K)', fontsize=12)
    
    ax=axes[1]

    d = ax.pcolormesh(deltaN_ref.lon, deltaN_ref.lat, lambda_ref70, cmap='RdBu_r', transform=ccrs.PlateCarree(), norm=divnorm)
    gl = ax.gridlines(draw_labels={"bottom": "x", "left": "y"}, color='gray', alpha=0.5)
    gl.xlabel_style = {'size': 10}
    gl.ylabel_style = {'size':10}    
    ax.set_title(ref_exp+' years 110-150')
    ax.coastlines()

    cb2 = plt.colorbar(d, ax=ax, extend='both', shrink=0.7)
    cb2.ax.set_ylabel(r'(W/m2/K)', fontsize=12)

    clevels = np.arange(-3, 3.3,0.3)
    #clevels = np.arange(-0.0005, 0.00055,0.00005)
    divnorm = mcolors.BoundaryNorm(clevels, ncolors=plt.colormaps['RdBu_r'].N, clip=True)

    ax=axes[2]
    d = ax.pcolormesh(deltaN_ref.lon, deltaN_ref.lat, lambda_ref70- lambda_ref30, cmap='RdBu_r', transform=ccrs.PlateCarree(), norm=divnorm)
    gl = ax.gridlines(draw_labels={"bottom": "x", "left": "y"}, color='gray', alpha=0.5)
    gl.xlabel_style = {'size': 10}
    gl.ylabel_style = {'size':10}    
    ax.set_title(ref_exp)
    ax.coastlines()

    cb2 = plt.colorbar(d, ax=ax, extend='both', shrink=0.7)
    cb2.ax.set_ylabel(r'(W/m2/K)', fontsize=12)

    plt.show()

def plot_lambdaKappa_map(ax=None, cart_out = './output/',ref_exp_c=None, ref_exp = None, atm_only = False, atmvars = 'rsut rlut rsdt tas rlnt rlntcs rsnt rsntcs'.split(), ocevars = 'tos heatc qt_oce sos'.split(),  year_clim = None, density=False):
    
    """
    Plots timeseries of var "vname" in domain "domain" for all exps.

    Domain is one among: ['atm', 'oce', 'ice']
    """
    if cart_out is None:
        raise ValueError('cart_out not specified!')

    cart_out_nc = cart_out + '/exps_clim/'
    nx = int(1)
    ny = int(3)
    #fig, axes = plt.subplots(nx, ny, figsize = (18, 5), subplot_kw={'projection': ccrs.PlateCarree()}) 

    lambda_map = xr.open_mfdataset(f'/ec/res4/hpcperm/itcv/analysis/exps_clim/KL/lambda_ts_{ref_exp}_.nc')
    lambdas = lambda_map.__xarray_dataarray_variable__

    kappa_map = xr.open_mfdataset(f'/ec/res4/hpcperm/itcv/analysis/exps_clim/KL/ohue_ts_{ref_exp}_regrid.nc')
    kappas = kappa_map.__xarray_dataarray_variable__

    llevels = np.arange(-0.1, 0.11,0.01)
    clevels = np.arange(-15, 16,1)
    divnorm = mcolors.BoundaryNorm(clevels, ncolors=plt.colormaps['RdBu_r'].N, clip=True)
    
    d = ax.pcolormesh(kappa_map.lon, kappa_map.lat, kappas[149]- kappas[30], cmap='RdBu_r', transform=ccrs.PlateCarree(), norm=divnorm)
    ax.contourf(lambda_map.lon, lambda_map.lat, lambdas[149]-lambdas[30], colors='k', negative_linestyles='dashed', levels=llevels, hatches='..')
    gl = ax.gridlines(draw_labels={"bottom": "x", "left": "y"}, color='gray', alpha=0.5)
    gl.xlabel_style = {'size': 10}
    gl.ylabel_style = {'size':10}    
    ax.set_title(ref_exp)
    ax.coastlines()

    cb2 = plt.colorbar(d, ax=ax, extend='both', shrink=0.7)
    cb2.ax.set_ylabel(r'(W/m2/K)', fontsize=12)

    clevels = np.arange(-3, 3.3,0.3)
    #clevels = np.arange(-0.0005, 0.00055,0.00005)
    divnorm = mcolors.BoundaryNorm(clevels, ncolors=plt.colormaps['RdBu_r'].N, clip=True)
    
    #kappas = kappas.fillna(0)
    kappas_np = np.array(kappas)
    lambdas_np = np.array(lambdas)

    mca = MCA(kappas_np[30:], lambdas_np[30:])
    mca.solve()
    mca.rotate(3)

    # svals = mca.singular_values()
    # expvar = mca.explained_variance()

    # hom_patterns = mca.homogeneous_patterns()
    # het_patterns  = mca.heterogeneous_patterns()
    mca.set_field_names('kappa', 'lambda')
    
    for i in range(1,5):
        mca.plot(mode=i) #, **pkwargs)

    # ax=axes[2]
    # d = ax.pcolormesh(deltaN_ref.lon, deltaN_ref.lat, lambda_ref70- lambda_ref30, cmap='RdBu_r', transform=ccrs.PlateCarree(), norm=divnorm)
    # gl = ax.gridlines(draw_labels={"bottom": "x", "left": "y"}, color='gray', alpha=0.5)
    # gl.xlabel_style = {'size': 10}
    # gl.ylabel_style = {'size':10}    
    # ax.set_title(ref_exp)
    # ax.coastlines()

    # cb2 = plt.colorbar(d, ax=ax, extend='both', shrink=0.7)
    # cb2.ax.set_ylabel(r'(W/m2/K)', fontsize=12)

    #plt.show()

def plot_lambdaKappa_mean(exps=None, ax=None, bx = None, cart_out = './output/', atm_only = False, atmvars = 'rsut rlut rsdt tas rlnt rlntcs rsnt rsntcs'.split(), ocevars = 'tos heatc qt_oce sos'.split(),  year_clim = None, density=False):
    
    """
    Plots timeseries of var "vname" in domain "domain" for all exps.

    Domain is one amo: ['atm', 'oce', 'ice']
    """
    if cart_out is None:
        raise ValueError('cart_out not specified!')

    cart_out_nc = cart_out + '/exps_clim/'
    nx = int(2)
    ny = int(2)
    #fig, axes = plt.subplots(nx, ny, figsize = (18, 5), subplot_kw={'projection': ccrs.PlateCarree()}) 
    lambdas = np.zeros([len(exps),150, 90,180])
    kappas = np.zeros([len(exps),150, 148,180])
    
    for i in range(len(exps)):
        lambda_map = xr.open_mfdataset(f'/ec/res4/hpcperm/itcv/analysis/exps_clim/KL/lambda_ts_{exps[i]}_.nc')
        lambda_exp = lambda_map.__xarray_dataarray_variable__
        lambdas[i] = lambda_exp

        kappa_map = xr.open_mfdataset(f'/ec/res4/hpcperm/itcv/analysis/exps_clim/KL/ohue_ts_{exps[i]}_.nc')
        kappa_exp = kappa_map.__xarray_dataarray_variable__
        kappas[i] = kappa_exp

    clevels = np.arange(-16, 18,2)
    divnorm = mcolors.BoundaryNorm(clevels, ncolors=plt.colormaps['RdBu_r'].N, clip=True)
    
    d = ax[0,0].pcolormesh(kappa_map.nav_lon, kappa_map.nav_lat, kappas[:,85].mean(axis=0), cmap='RdBu_r', transform=ccrs.PlateCarree(), norm=divnorm)
    gl = ax[0,0].gridlines(draw_labels={"bottom": "x", "left": "y"}, color='gray', alpha=0.5)
    gl.xlabel_style = {'size': 10}
    gl.ylabel_style = {'size':10}    
    ax[0,0].set_title('Mean K')
    ax[0,0].coastlines()

    cb2 = plt.colorbar(d, ax=ax[0,0], extend='both', shrink=0.7)
    cb2.ax.set_ylabel(r'(W/m2/K)', fontsize=12)

    clevels = np.arange(-5, 6,1)
    divnorm = mcolors.BoundaryNorm(clevels, ncolors=plt.colormaps['RdBu_r'].N, clip=True)
    
    d = ax[0,1].pcolormesh(lambda_map.lon, lambda_map.lat, lambdas[:,85].mean(axis=0), cmap='RdBu_r', transform=ccrs.PlateCarree(), norm=divnorm)
    gl = ax[0,1].gridlines(draw_labels={"bottom": "x", "left": "y"}, color='gray', alpha=0.5)
    gl.xlabel_style = {'size': 10}
    gl.ylabel_style = {'size':10}    
    ax[0,1].set_title('Mean L')
    ax[0,1].coastlines()

    cb2 = plt.colorbar(d, ax=ax[0,1], extend='both', shrink=0.7)
    cb2.ax.set_ylabel(r'(W/m2/K)', fontsize=12)

    clevels = np.arange(0, 5.5,0.5)
    divnorm = mcolors.BoundaryNorm(clevels, ncolors=plt.colormaps['RdBu_r'].N, clip=True)
    
    d = ax[1,0].pcolormesh(kappa_map.nav_lon, kappa_map.nav_lat, kappas[:,85].std(axis=0), cmap='Reds', transform=ccrs.PlateCarree(), norm=divnorm)
    gl = ax[1,0].gridlines(draw_labels={"bottom": "x", "left": "y"}, color='gray', alpha=0.5)
    gl.xlabel_style = {'size': 10}
    gl.ylabel_style = {'size':10}    
    ax[1,0].set_title('std K')
    ax[1,0].coastlines()

    cb2 = plt.colorbar(d, ax=ax[1,0], extend='both', shrink=0.7)
    cb2.ax.set_ylabel(r'(W/m2/K)', fontsize=12)

    clevels = np.arange(0,5.5,0.5)
    divnorm = mcolors.BoundaryNorm(clevels, ncolors=plt.colormaps['RdBu_r'].N, clip=True)
    
    d = ax[1,1].pcolormesh(lambda_map.lon, lambda_map.lat, lambdas[:,85].std(axis=0), cmap='Reds', transform=ccrs.PlateCarree(), norm=divnorm)
    gl = ax[1,1].gridlines(draw_labels={"bottom": "x", "left": "y"}, color='gray', alpha=0.5)
    gl.xlabel_style = {'size': 10}
    gl.ylabel_style = {'size':10}    
    ax[1,1].set_title('Mean L')
    ax[1,1].coastlines()

    cb2 = plt.colorbar(d, ax=ax[1,1], extend='both', shrink=0.7)
    cb2.ax.set_ylabel(r'(W/m2/K)', fontsize=12)
    
    clevels2 = np.arange(-5,6,1)
    divnorm2 = mcolors.BoundaryNorm(clevels2, ncolors=plt.colormaps['RdBu_r'].N, clip=True)
    clevels = np.arange(-1,1.1,0.1)
    divnorm = mcolors.BoundaryNorm(clevels, ncolors=plt.colormaps['RdBu_r'].N, clip=True)
    
    for i in range(len(exps)):

        d = bx[i,0].pcolormesh(kappa_map.nav_lon, kappa_map.nav_lat, kappas[i,85] - kappas[:,85].mean(axis=0), cmap='RdBu_r', transform=ccrs.PlateCarree(), norm=divnorm2)
        gl = bx[i,0].gridlines(draw_labels={"bottom": "x", "left": "y"}, color='gray', alpha=0.5)
        gl.xlabel_style = {'size': 10}
        gl.ylabel_style = {'size':10}    
        bx[i,0].set_title(exps[i])
        bx[i,0].coastlines()

        cb2 = plt.colorbar(d, ax=bx[i,0], extend='both', shrink=0.7)
        cb2.ax.set_ylabel(r'(W/m2/K)', fontsize=12)

        d = bx[i,1].pcolormesh(lambda_map.lon, lambda_map.lat, lambdas[i,85] - lambdas[:,85].mean(axis=0), cmap='RdBu_r', transform=ccrs.PlateCarree(), norm=divnorm)
        gl = bx[i,1].gridlines(draw_labels={"bottom": "x", "left": "y"}, color='gray', alpha=0.5)
        gl.xlabel_style = {'size': 10}
        gl.ylabel_style = {'size':10}    
        bx[i,1].set_title(exps[i])
        bx[i,1].coastlines()

        cb2 = plt.colorbar(d, ax=bx[i,1], extend='both', shrink=0.7)
        cb2.ax.set_ylabel(r'(W/m2/K)', fontsize=12)

    plt.show()

def plot_lambda_comp_zonal(exps, user = None, axes = None,color=None, read_again = [], cart_exp = '/ec/res4/scratch/{}/ece4/', cart_out = './output/',ref_exp_c=None, ref_exp = None, atm_only = False, atmvars = ' tas'.split(), ocevars = 'tos heatc qt_oce sos'.split(),  year_clim = None, density=False):
    
    """
    Plots timeseries of var "vname" in domain "domain" for all exps.

    Domain is one among: ['atm', 'oce', 'ice']
    """
    if cart_out is None:
        raise ValueError('cart_out not specified!')

    cart_out_nc = cart_out + '/exps_clim/'
    nx = int(7)
    ny = int(3)

    map_all = read_output_map(exps, user = user, read_again = read_again, cart_exp = cart_exp, cart_out = cart_out_nc, atm_only = atm_only, atmvars = 'rlnt rlntcs rsnt rsntcs'.split(), ocevars = ocevars, year_clim = year_clim, density=density)

    atm_dataset = map_all[f'atm_map']
    atm_dataset = {co: atm_dataset[co] for co in atm_dataset if atm_dataset[co] is not None}
    atm_dataset = create_ds_exp(atm_dataset)

    # possible atmospheric components! 
    toa_cs  = atm_dataset['rlntcs'] + atm_dataset['rsntcs']
    toa_net = atm_dataset['rlnt'] + atm_dataset['rsnt']
    toa_cloud = toa_net - toa_cs

    toa_cs_lw = atm_dataset['rlntcs']
    toa_cs_sw = atm_dataset['rsntcs']

    toa_cloud_lw = atm_dataset['rlnt'] - atm_dataset['rlntcs']
    toa_cloud_sw = atm_dataset['rsnt'] - atm_dataset['rsntcs']

    atmos = [toa_net, toa_cs, toa_cs_lw, toa_cs_sw, toa_cloud, toa_cloud_lw, toa_cloud_sw]

    # read global mean values for TAS
    clim_all = read_output(exps, user = user, read_again = read_again, cart_exp = cart_exp, cart_out = cart_out_nc, atm_only = atm_only, atmvars = atmvars, ocevars = ocevars, icevars = [], year_clim = year_clim, density=density)
    atm_mean = clim_all[f'atm_mean']
    atm_mean = {co: atm_mean[co] for co in atm_mean if atm_mean[co] is not None}
    atm_mean = create_ds_exp(atm_mean)
    tas = atm_mean['tas']

    #forcing = compute_radiative_forcing()
    ds_forcing = xr.open_mfdataset('/ec/res4/hpcperm/itcv/analysis/forcing_1pct/'+exps[0]+'_forcing.nc')
    forcing = ds_forcing.forcing

    forcings = [ds_forcing.forcing_net, ds_forcing.forcing_cs, ds_forcing.forcing_cs_lw, ds_forcing.forcing_cs_sw, ds_forcing.forcing_cloud, ds_forcing.forcing_cloud_lw, ds_forcing.forcing_cloud_sw]

    year1 = 1850
    year2 = 1960

    for k, (toa, forcing) in enumerate(zip(atmos, forcings)):
        trend_N = np.zeros([90,180])
        intercept_N = np.zeros([90,180])

        trend_T = np.zeros([90,180])
        intercept_T = np.zeros([90,180])

        for i in range(np.shape(toa.sel(exp=ref_exp_c))[1]):
            for j in range(np.shape(toa.sel(exp=ref_exp_c))[2]):

                trend_N[i,j], intercept_N[i,j], _, _, _ = stats.linregress(np.arange(0,np.shape(toa.sel(exp=ref_exp_c))[0]), toa.sel(exp=ref_exp_c)[:,i,j])
                #trend_T[i,j], intercept_T[i,j], _, _, _ = stats.linregress(np.arange(0,np.shape(tas.sel(exp=ref_exp_c))[0]), tas.sel(exp=ref_exp_c)[:,i,j])

        trend_T, intercept_T, _,_,_ = stats.linregress(np.arange(0,len(tas.sel(exp=ref_exp))), tas.sel(exp=ref_exp_c))
        deltaN_ref = toa.sel(exp=ref_exp)[:150] - np.arange(0,150)[:,np.newaxis, np.newaxis]*trend_N - intercept_N - forcing #[:,np.newaxis, np.newaxis] 
        deltaTAS_ref = tas.sel(exp=ref_exp)[:150] - np.arange(0,150)*trend_T - intercept_T
        
        weights = np.cos(np.deg2rad(deltaN_ref.lat))
        tot_weights = weights.sum()*deltaN_ref.lon.size
        
        # lambda_ref30 = np.zeros([90,180])
        # lambda_ref70 = np.zeros([90,180])

        deltaT = deltaTAS_ref.sel(year=slice(year1,year1+40)).mean(axis=0)#*tot_weights
        deltaT2 = deltaTAS_ref.sel(year=slice(year2,year2+40)).mean(axis=0)#*tot_weights 

        # for j in range(90):
        #         for k in range(180):
        #             lambda_ref30[j,k] = stats.linregress(deltaTAS_ref.sel(year=slice(year1,year1+40)), deltaN_ref.sel(year=slice(year1,year1+40))[:,j,k])[0]
        #             lambda_ref70[j,k] = stats.linregress(deltaTAS_ref.sel(year=slice(year2,year2+40)), deltaN_ref.sel(year=slice(year2,year2+40))[:,j,k])[0]
        
        lambda_ref30 = (deltaN_ref).sel(year=slice(year1,year1+40)).mean(axis=0)/ deltaT
        lambda_ref70 = (deltaN_ref).sel(year=slice(year2,year2+40)).mean(axis=0)/ deltaT2

        year = 1910
        lambda70 = (deltaN_ref).sel(year=slice(year,year+20)).mean(axis=0)/ deltaTAS_ref.sel(year=slice(year,year+20)).mean(axis=0)
        #axes[k].plot(deltaN_ref.lat,lambda_ref30.mean(dim='lon'),  label = '20-60', color = color)
        #axes[k].plot(deltaN_ref.lat,lambda_ref70.mean(dim='lon'),  label = '60-100', color = color) 
        axes[k].plot(deltaN_ref.lat,lambda_ref70.mean(dim='lon')- lambda_ref30.mean(dim='lon'),  label = exps[0], color = color)
        axes[k].plot(deltaN_ref.lat,lambda_ref70.mean(dim='lon')- lambda_ref30.mean(dim='lon'),  label = exps[0], color = color)

        axes[k].set_title(forcing.long_name, fontsize=12)
        #axes[k].set_xlabel('Latitude', fontsize=12)
        axes[k].hlines(0, -90,90, color='gray', linestyle='--')
        axes[k].set_ylim(-5.5,5.5)

        for x in [-60, -30, 30, 60]:
            axes[k].vlines(x, -5.5, 5.5, color='gray')


    #plt.show()

def compute_lambda_comp_regions(exps, user = None, axes = None,color=None, read_again = [], cart_exp = '/ec/res4/scratch/{}/ece4/', cart_out = './output/',ref_exp_c=None, ref_exp = None, atm_only = False, atmvars = ' tas'.split(), ocevars = 'tos heatc qt_oce sos'.split(),  year_clim = None, density=False):
    
    """
    Plots timeseries of var "vname" in domain "domain" for all exps.

    Domain is one among: ['atm', 'oce', 'ice']
    """
    if cart_out is None:
        raise ValueError('cart_out not specified!')

    cart_out_nc = cart_out + '/exps_clim/'
    nx = int(7)
    ny = int(3)

    window = 30
    map_all = read_output_map(exps, user = user, read_again = read_again, cart_exp = cart_exp, cart_out = cart_out_nc, atm_only = atm_only, atmvars = 'rlnt rlntcs rsnt rsntcs'.split(), ocevars = ocevars, year_clim = year_clim, density=density)

    atm_dataset = map_all[f'atm_map']
    atm_dataset = {co: atm_dataset[co] for co in atm_dataset if atm_dataset[co] is not None}
    atm_dataset = create_ds_exp(atm_dataset)

    # possible atmospheric components! 
    toa_cs  = atm_dataset['rlntcs'] + atm_dataset['rsntcs']
    toa_net = atm_dataset['rlnt'] + atm_dataset['rsnt']
    toa_cloud = toa_net - toa_cs

    toa_cs_lw = atm_dataset['rlntcs']
    toa_cs_sw = atm_dataset['rsntcs']

    toa_cloud_lw = atm_dataset['rlnt'] - atm_dataset['rlntcs']
    toa_cloud_sw = atm_dataset['rsnt'] - atm_dataset['rsntcs']

    atmos = [toa_net, toa_cs, toa_cs_lw, toa_cs_sw, toa_cloud, toa_cloud_lw, toa_cloud_sw]

    # read global mean values for TAS
    clim_all = read_output(exps, user = user, read_again = read_again, cart_exp = cart_exp, cart_out = cart_out_nc, atm_only = atm_only, atmvars = atmvars, ocevars = ocevars, icevars = [], year_clim = year_clim, density=density)
    atm_mean = clim_all[f'atm_mean']
    atm_mean = {co: atm_mean[co] for co in atm_mean if atm_mean[co] is not None}
    atm_mean = create_ds_exp(atm_mean)
    tas = atm_mean['tas']

    #forcing = compute_radiative_forcing()
    ds_forcing = xr.open_mfdataset('/ec/res4/hpcperm/itcv/analysis/forcing_1pct/'+exps[0]+'_forcing.nc')
    forcing = ds_forcing.forcing

    forcings = [ds_forcing.forcing_net, ds_forcing.forcing_cs, ds_forcing.forcing_cs_lw, ds_forcing.forcing_cs_sw, ds_forcing.forcing_cloud, ds_forcing.forcing_cloud_lw, ds_forcing.forcing_cloud_sw]

    lambdas = np.zeros([7,7,150])

    for k, (toa, forcing) in enumerate(zip(atmos, forcings)):
        trend_N = np.zeros([90,180])
        intercept_N = np.zeros([90,180])

        trend_T = np.zeros([90,180])
        intercept_T = np.zeros([90,180])

        for i in range(np.shape(toa.sel(exp=exps[1]))[1]):
            for j in range(np.shape(toa.sel(exp=exps[1]))[2]):

                trend_N[i,j], intercept_N[i,j], _, _, _ = stats.linregress(np.arange(0,np.shape(toa.sel(exp=exps[1]))[0]), toa.sel(exp=exps[1])[:,i,j])

        trend_T, intercept_T, _,_,_ = stats.linregress(np.arange(0,len(tas.sel(exp=exps[1]))), tas.sel(exp=exps[1]))
        deltaR = toa.sel(exp=exps[0])[:150] - np.arange(0,150)[:,np.newaxis, np.newaxis]*trend_N - intercept_N - forcing #[:,np.newaxis, np.newaxis] 
        deltaT = tas.sel(exp=exps[0])[:150] - np.arange(0,150)*trend_T - intercept_T
        
        weights = np.cos(np.deg2rad(deltaR.lat))
        tot_weights = weights.sum()*deltaR.lon.size

        #OHUE all 
        lambda_all = (deltaR*weights).rolling(year=window).mean().sum(axis=(1,2))/(deltaT.rolling(year=window).mean()*tot_weights)
        
        #OHUE Smidlatitudes     
        deltaN_reg = deltaR.where(((deltaR.lat < -30) & (deltaR.lat > -60)), np.nan)
        weights_reg = weights.where(((deltaR.lat < -30) & (deltaR.lat > -60)), 0)
        lambda_smid = (deltaN_reg*weights_reg).rolling(year=window).mean().sum(axis=(1,2))/(deltaT.rolling(year=window).mean()*tot_weights)

        # #OHUE Nmidlatitudes 
        deltaN_reg = deltaR.where(((deltaR.lat < 60) & (deltaR.lat > 30)), np.nan)
        weights_reg = weights.where(((deltaR.lat < 60) & (deltaR.lat > 30)), 0)
        lambda_nmid = (deltaN_reg*weights_reg).rolling(year=window).mean().sum(axis=(1,2))/(deltaT.rolling(year=window).mean()*tot_weights)

        # #OHUE lowlatitudes
        deltaN_reg = deltaR.where(((deltaR.lat < 30) & (deltaR.lat > -30)), np.nan)
        weights_reg = weights.where(((deltaR.lat < 30) & (deltaR.lat > -30)), 0)
        lambda_low = (deltaN_reg*weights_reg).rolling(year=window).mean().sum(axis=(1,2))/(deltaT.rolling(year=window).mean()*tot_weights)

        # #OHUE Shighlatitudes
        deltaN_reg = deltaR.where(((deltaR.lat < -60)), np.nan)
        weights_reg = weights.where(((deltaR.lat < -60)), 0)
        lambda_shigh = (deltaN_reg*weights_reg).rolling(year=window).mean().sum(axis=(1,2))/(deltaT.rolling(year=window).mean()*tot_weights)

        # #OHUE Shighlatitudes
        deltaN_reg = deltaR.where(((deltaR.lat > 60)), np.nan)
        weights_reg = weights.where(((deltaR.lat > 60)),0)
        lambda_nhigh = (deltaN_reg*weights_reg).rolling(year=window).mean().sum(axis=(1,2))/(deltaT.rolling(year=window).mean()*tot_weights)

        lambdas[k,0,:] = lambda_all
        lambdas[k,1,:] = lambda_shigh
        lambdas[k,2,:] = lambda_smid
        lambdas[k,3,:] = lambda_low
        lambdas[k,4,:] = lambda_nmid
        lambdas[k,5,:] = lambda_nhigh
        lambdas[k,6,:] = lambda_nmid + lambda_smid

    return lambdas

def plot_lambda_comp_map(exps, user = None, read_again = [], cart_exp = '/ec/res4/scratch/{}/ece4/', cart_out = './output/',ref_exp_c=None, ref_exp = None, atm_only = False, atmvars = ' tas'.split(), ocevars = 'tos heatc qt_oce sos'.split(),  year_clim = None, density=False):
    
    """
    Plots timeseries of var "vname" in domain "domain" for all exps.

    Domain is one among: ['atm', 'oce', 'ice']
    """
    if cart_out is None:
        raise ValueError('cart_out not specified!')

    cart_out_nc = cart_out + '/exps_clim/'
    nx = int(7)
    ny = int(3)
    fig, axes = plt.subplots(nx, ny, figsize = (18, 25), subplot_kw={'projection': ccrs.PlateCarree()}) 

    map_all = read_output_map(exps, user = user, read_again = read_again, cart_exp = cart_exp, cart_out = cart_out_nc, atm_only = atm_only, atmvars = 'rlnt rlntcs rsnt rsntcs'.split(), ocevars = ocevars, year_clim = year_clim, density=density)

    atm_dataset = map_all[f'atm_map']
    atm_dataset = {co: atm_dataset[co] for co in atm_dataset if atm_dataset[co] is not None}
    atm_dataset = create_ds_exp(atm_dataset)

    # possible atmospheric components! 
    toa_cs  = atm_dataset['rlntcs'] + atm_dataset['rsntcs']
    toa_net = atm_dataset['rlnt'] + atm_dataset['rsnt']
    toa_cloud = toa_net - toa_cs

    toa_cs_lw = atm_dataset['rlntcs']
    toa_cs_sw = atm_dataset['rsntcs']

    toa_cloud_lw = atm_dataset['rlnt'] - atm_dataset['rlntcs']
    toa_cloud_sw = atm_dataset['rsnt'] - atm_dataset['rsntcs']

    atmos = [toa_net, toa_cs, toa_cs_lw, toa_cs_sw, toa_cloud, toa_cloud_lw, toa_cloud_sw]

    # read global mean values for TAS
    clim_all = read_output(exps, user = user, read_again = read_again, cart_exp = cart_exp, cart_out = cart_out_nc, atm_only = atm_only, atmvars = atmvars, ocevars = ocevars, icevars = [], year_clim = year_clim, density=density)
    atm_mean = clim_all[f'atm_mean']
    atm_mean = {co: atm_mean[co] for co in atm_mean if atm_mean[co] is not None}
    atm_mean = create_ds_exp(atm_mean)
    tas = atm_mean['tas']

    #forcing = compute_radiative_forcing()
    ds_forcing = xr.open_mfdataset('/ec/res4/hpcperm/itcv/analysis/forcing_1pct/'+exps[0]+'_forcing.nc')
    forcing = ds_forcing.forcing

    forcings = [ds_forcing.forcing_net, ds_forcing.forcing_cs, ds_forcing.forcing_cs_lw, ds_forcing.forcing_cs_sw, ds_forcing.forcing_cloud, ds_forcing.forcing_cloud_lw, ds_forcing.forcing_cloud_sw]

    year1 = 1850
    year2 = 1960

    for k, (toa, forcing) in enumerate(zip(atmos, forcings)):
        axes[k,1].text(-60,120,forcing.long_name, fontsize=14)
        trend_N = np.zeros([90,180])
        intercept_N = np.zeros([90,180])

        trend_T = np.zeros([90,180])
        intercept_T = np.zeros([90,180])

        for i in range(np.shape(toa.sel(exp=ref_exp_c))[1]):
            for j in range(np.shape(toa.sel(exp=ref_exp_c))[2]):

                trend_N[i,j], intercept_N[i,j], _, _, _ = stats.linregress(np.arange(0,np.shape(toa.sel(exp=ref_exp_c))[0]), toa.sel(exp=ref_exp_c)[:,i,j])
                #trend_T[i,j], intercept_T[i,j], _, _, _ = stats.linregress(np.arange(0,np.shape(tas.sel(exp=ref_exp_c))[0]), tas.sel(exp=ref_exp_c)[:,i,j])

        trend_T, intercept_T, _,_,_ = stats.linregress(np.arange(0,len(tas.sel(exp=ref_exp))), tas.sel(exp=ref_exp_c))
        deltaN_ref = toa.sel(exp=ref_exp)[:150] - np.arange(0,150)[:,np.newaxis, np.newaxis]*trend_N - intercept_N - forcing #[:,np.newaxis, np.newaxis] 
        deltaTAS_ref = tas.sel(exp=ref_exp)[:150] - np.arange(0,150)*trend_T - intercept_T
        
        weights = np.cos(np.deg2rad(deltaN_ref.lat))
        tot_weights = weights.sum()*deltaN_ref.lon.size
        
        lambda_ref30 = np.zeros([90,180])
        lambda_ref70 = np.zeros([90,180])

        deltaT = deltaTAS_ref.sel(year=slice(year1,year1+40)).mean(axis=0)#*tot_weights
        deltaT2 = deltaTAS_ref.sel(year=slice(year2,year2+40)).mean(axis=0)#*tot_weights 

        for i in range(90):
                for j in range(180):
                    lambda_ref30[i,j] = stats.linregress(deltaTAS_ref.sel(year=slice(year1,year1+40)), deltaN_ref.sel(year=slice(year1,year1+40))[:,i,j])[0]
                    lambda_ref70[i,j] = stats.linregress(deltaTAS_ref.sel(year=slice(year2,year2+40)), deltaN_ref.sel(year=slice(year2,year2+40))[:,i,j])[0]
        
        # lambda_ref30 = (deltaN_ref).sel(year=slice(year1,year1+40)).mean(axis=0)/ deltaT
        # lambda_ref70 = (deltaN_ref).sel(year=slice(year2,year2+40)).mean(axis=0)/ deltaT2
        
        # lambda_ref30 = forcing.sel(year=slice(year1,year1+40)).mean(axis=0)
        # lambda_ref70 = (forcing).sel(year=slice(year2,year2+40)).mean(axis=0)

        clevels = np.arange(-0.001, 0.0011,0.0001) # for partial contributions
        clevels = np.arange(-3, 3.3,0.3) # for feedback
        divnorm = mcolors.BoundaryNorm(clevels, ncolors=plt.colormaps['RdBu_r'].N, clip=True)
        
        ax = axes[k,0]
        d = ax.pcolormesh(deltaN_ref.lon, deltaN_ref.lat, lambda_ref30, cmap='RdBu_r', transform=ccrs.PlateCarree(), norm=divnorm)
        gl = ax.gridlines(draw_labels={"bottom": "x", "left": "y"}, color='gray', alpha=0.5)
        gl.xlabel_style = {'size': 10}
        gl.ylabel_style = {'size':10}    
        ax.set_title(ref_exp+' years 20-60')
        ax.coastlines()

        cb2 = plt.colorbar(d, ax=ax, extend='both', shrink=0.7)
        cb2.ax.set_ylabel(r'(W/m2/K)', fontsize=12)
        
        ax=axes[k,1]
        #clevels = np.arange(-10, 11,1) # for feedback
        divnorm = mcolors.BoundaryNorm(clevels, ncolors=plt.colormaps['RdBu_r'].N, clip=True)
        d = ax.pcolormesh(deltaN_ref.lon, deltaN_ref.lat, lambda_ref70, cmap='RdBu_r', transform=ccrs.PlateCarree(), norm=divnorm)
        gl = ax.gridlines(draw_labels={"bottom": "x", "left": "y"}, color='gray', alpha=0.5)
        gl.xlabel_style = {'size': 10}
        gl.ylabel_style = {'size':10}    
        ax.set_title(ref_exp+' years 110-150')
        ax.coastlines()

        cb2 = plt.colorbar(d, ax=ax, extend='both', shrink=0.7)
        cb2.ax.set_ylabel(r'(W/m2/K)', fontsize=12)

        #clevels = np.arange(-3, 3.3,0.3)
        #clevels = np.arange(-0.0005, 0.00055,0.00005)
        divnorm = mcolors.BoundaryNorm(clevels, ncolors=plt.colormaps['RdBu_r'].N, clip=True)

        ax=axes[k,2]
        d = ax.pcolormesh(deltaN_ref.lon, deltaN_ref.lat, lambda_ref70- lambda_ref30, cmap='RdBu_r', transform=ccrs.PlateCarree(), norm=divnorm)
        gl = ax.gridlines(draw_labels={"bottom": "x", "left": "y"}, color='gray', alpha=0.5)
        gl.xlabel_style = {'size': 10}
        gl.ylabel_style = {'size':10}    
        ax.set_title(ref_exp)
        ax.coastlines()

        cb2 = plt.colorbar(d, ax=ax, extend='both', shrink=0.7)
        cb2.ax.set_ylabel(r'(W/m2/K)', fontsize=12)

    plt.show()

def plot_clouds_map(exps, user = None, read_again = [], cart_exp = '/ec/res4/scratch/{}/ece4/', cart_out = './output/',ref_exp_c=None, ref_exp = None, atm_only = False, atmvars = 'clh clm cll'.split(), ocevars = 'tos heatc qt_oce sos'.split(),  year_clim = None, density=False):
    
    """
    Plots timeseries of var "vname" in domain "domain" for all exps.

    Domain is one among: ['atm', 'oce', 'ice']
    """
    if cart_out is None:
        raise ValueError('cart_out not specified!')

    cart_out_nc = cart_out + '/exps_clim/'
    nx = int(3)
    ny = int(3)
    fig, axes = plt.subplots(nx, ny, figsize = (18, 10), subplot_kw={'projection': ccrs.PlateCarree()}) 

    map_all = read_output_map(exps, user = user, read_again = read_again, cart_exp = cart_exp, cart_out = cart_out_nc, atm_only = atm_only, atmvars = atmvars, ocevars = ocevars, year_clim = year_clim, density=density)

    atm_dataset = map_all[f'atm_map']
    atm_dataset = {co: atm_dataset[co] for co in atm_dataset if atm_dataset[co] is not None}
    atm_dataset = create_ds_exp(atm_dataset)

    # possible atmospheric components! 

    atmos = [atm_dataset['clh'], atm_dataset['clm'], atm_dataset['cll']]

    year1 = 1850
    year2 = 1960

    for k, clouds in enumerate(atmos):

        axes[k,1].text(-60,120,clouds.long_name, fontsize=14)
        trend_N = np.zeros([90,180])
        intercept_N = np.zeros([90,180])

        for i in range(np.shape(clouds.sel(exp=ref_exp_c))[1]):
            for j in range(np.shape(clouds.sel(exp=ref_exp_c))[2]):

                trend_N[i,j], intercept_N[i,j], _, _, _ = stats.linregress(np.arange(0,np.shape(clouds.sel(exp=ref_exp_c))[0]), clouds.sel(exp=ref_exp_c)[:,i,j])
                #trend_T[i,j], intercept_T[i,j], _, _, _ = stats.linregress(np.arange(0,np.shape(tas.sel(exp=ref_exp_c))[0]), tas.sel(exp=ref_exp_c)[:,i,j])

        deltaN_ref = clouds.sel(exp=ref_exp)[:150] # - np.arange(0,150)[:,np.newaxis, np.newaxis]*trend_N - intercept_N  #[:,np.newaxis, np.newaxis] 
        
        weights = np.cos(np.deg2rad(deltaN_ref.lat))
        tot_weights = weights.sum()*deltaN_ref.lon.size

        # for j in range(90):
        #         for k in range(180):
        #             lambda_ref30[j,k] = stats.linregress(deltaTAS_ref.sel(year=slice(year1,year1+40)), deltaN_ref.sel(year=slice(year1,year1+40))[:,j,k])[0]
        #             lambda_ref70[j,k] = stats.linregress(deltaTAS_ref.sel(year=slice(year2,year2+40)), deltaN_ref.sel(year=slice(year2,year2+40))[:,j,k])[0]
        
        lambda_ref30 = (deltaN_ref).sel(year=slice(year1,year1+40)).mean(axis=0)
        lambda_ref70 = (deltaN_ref).sel(year=slice(year2,year2+40)).mean(axis=0)

        clevels = np.arange(0, 110,10)
        divnorm = mcolors.BoundaryNorm(clevels, ncolors=plt.colormaps['RdBu_r'].N, clip=True)
        
        ax = axes[k,0]
        d = ax.pcolormesh(deltaN_ref.lon, deltaN_ref.lat, lambda_ref30, cmap='Purples', transform=ccrs.PlateCarree(), norm=divnorm)
        gl = ax.gridlines(draw_labels={"bottom": "x", "left": "y"}, color='gray', alpha=0.5)
        gl.xlabel_style = {'size': 10}
        gl.ylabel_style = {'size':10}    
        ax.set_title(ref_exp+' years 20-60')
        ax.coastlines()

        cb2 = plt.colorbar(d, ax=ax, extend='both', shrink=0.7)
        cb2.ax.set_ylabel(r'()', fontsize=12)
        
        ax=axes[k,1]

        d = ax.pcolormesh(deltaN_ref.lon, deltaN_ref.lat, lambda_ref70, cmap='Purples', transform=ccrs.PlateCarree(), norm=divnorm)
        gl = ax.gridlines(draw_labels={"bottom": "x", "left": "y"}, color='gray', alpha=0.5)
        gl.xlabel_style = {'size': 10}
        gl.ylabel_style = {'size':10}    
        ax.set_title(ref_exp+' years 110-150')
        ax.coastlines()

        cb2 = plt.colorbar(d, ax=ax, extend='both', shrink=0.7)
        cb2.ax.set_ylabel(r'()', fontsize=12)

        clevels = np.arange(-20, 24,4)
        divnorm = mcolors.BoundaryNorm(clevels, ncolors=plt.colormaps['RdBu_r'].N, clip=True)

        ax=axes[k,2]
        d = ax.pcolormesh(deltaN_ref.lon, deltaN_ref.lat, lambda_ref70- lambda_ref30, cmap='RdBu_r', transform=ccrs.PlateCarree(), norm=divnorm)
        gl = ax.gridlines(draw_labels={"bottom": "x", "left": "y"}, color='gray', alpha=0.5)
        gl.xlabel_style = {'size': 10}
        gl.ylabel_style = {'size':10}    
        ax.set_title(ref_exp)
        ax.coastlines()

        cb2 = plt.colorbar(d, ax=ax, extend='both', shrink=0.7)
        cb2.ax.set_ylabel(r'()', fontsize=12)

    plt.show()

def plot_clouds_zonal(exps,user=None,axes=None,color=None,read_again=[],cart_exp='/ec/res4/scratch/{}/ece4/',cart_out='./output/',ref_exp_c=None,
    ref_exp=None,atm_only=False,atmvars='clh clm cll'.split(),ocevars='tos heatc qt_oce sos'.split(),year_clim=None,density=False):
    """
    Zonal comparison of cloud components (early vs late period).
    """

    cart_out_nc = cart_out + '/exps_clim/'

    # --- READ MAP DATA ---
    map_all = read_output_map(exps,user=user,read_again=read_again,cart_exp=cart_exp,cart_out=cart_out_nc,atm_only=atm_only, atmvars=atmvars,ocevars=ocevars,year_clim=year_clim,density=density)

    atm_dataset = map_all['atm_map']
    atm_dataset = {co: atm_dataset[co] for co in atm_dataset if atm_dataset[co] is not None}
    atm_dataset = create_ds_exp(atm_dataset)

    # Cloud components
    atmos = [atm_dataset['clh'],atm_dataset['clm'],atm_dataset['cll']]

    year1 = 1850
    year2 = 1960

    tot = np.zeros(90)

    for k, clouds in enumerate(atmos):

        # --- TREND REMOVAL using control experiment ---
        trend = np.zeros([90, 180])
        intercept = np.zeros([90, 180])

        control = clouds.sel(exp=ref_exp_c)

        for i in range(control.shape[1]):
            for j in range(control.shape[2]):
                trend[i, j], intercept[i, j], _, _, _ = stats.linregress(np.arange(control.shape[0]),control[:, i, j])

        # --- anomaly for reference experiment ---
        delta = clouds.sel(exp=ref_exp)[:150] - np.arange(0, 150)[:, np.newaxis, np.newaxis] * trend - intercept

        # --- EARLY and LATE period means ---
        early = delta.sel(year=slice(year1, year1+40)).mean(dim='year')
        late  = delta.sel(year=slice(year2, year2+40)).mean(dim='year')

        # --- ZONAL MEAN ---
        early_zonal = early.mean(dim='lon')
        late_zonal  = late.mean(dim='lon')

        # --- PLOT difference (late - early) ---
        axes[k].plot(delta.lat,late_zonal - early_zonal,label=exps[0],color=color)

        axes[k].set_title(clouds.long_name, fontsize=12)
        axes[k].hlines(0, -90, 90, color='gray', linestyle='--')
        axes[k].set_xlim(-90, 90)

        for x in [-60, -30, 30, 60]:
            axes[k].vlines(x, (late_zonal - early_zonal).min(),
                           (late_zonal - early_zonal).max(),
                           color='gray')

        axes[k].set_xlabel('Latitude')
        axes[k].set_ylabel('Cloud fraction change')

        tot += late_zonal - early_zonal

    axes[3].plot(delta.lat, tot, label=exps[0], color=color)
    axes[3].set_title('Total cloud fraction change', fontsize=12)
    axes[3].hlines(0, -90, 90, color='gray', linestyle='--')
    axes[3].set_xlim(-90, 90)
    for x in [-60, -30, 30, 60]:
        axes[3].vlines(x, tot.min(), tot.max(), color='gray')

def compute_base_anom_state(exps, user = None, read_again = [], ax=None, cart_exp = '/ec/res4/scratch/{}/ece4/', cart_out = './output/', ref_exp = None, atm_only = False, atmvars = 'rsut rlut rsdt tas pr'.split(), ocevars = 'tos heatc qt_oce sos'.split(), icevars = 'siconc sivolu sithic'.split(), year_clim = None, density=False, color= None):
    """
    Plots timeseries of var "vname" in domain "domain" for all exps.

    Domain is one among: ['atm', 'oce', 'ice']
    """
    if cart_out is None:
        raise ValueError('cart_out not specified!')
    
    if not os.path.exists(cart_out): os.mkdir(cart_out)

    cart_out_nc = cart_out + '/exps_clim/'
    cart_out_figs = cart_out + f'/check_{'-'.join(exps)}/'

    clim_all = read_output(exps, user = user, read_again = read_again, cart_exp = cart_exp, cart_out = cart_out_nc, atm_only = atm_only, atmvars = atmvars, ocevars = ocevars, icevars = icevars, year_clim = year_clim, density=density)

    toa_dataset = clim_all[f'atm_mean']
    toa_dataset = {co: toa_dataset[co] for co in toa_dataset if toa_dataset[co] is not None}
    toa_dataset = create_ds_exp(toa_dataset)
    tas = toa_dataset['tas']
    netTOA = toa_dataset['rsdt'] - toa_dataset['rlut'] - toa_dataset['rsut']

    ice_dataset = clim_all[f'ice_mean']
    ice_dataset = {co: ice_dataset[co] for co in ice_dataset if ice_dataset[co] is not None}
    ice_dataset = create_ds_exp(ice_dataset)
    siconcN = ice_dataset['siconc_N']

    oce_dataset = clim_all[f'oce_mean']
    oce_dataset = {co: oce_dataset[co] for co in oce_dataset if oce_dataset[co] is not None}
    oce_dataset = create_ds_exp(oce_dataset)
    shf = oce_dataset['qt_oce']

    ds_forcing = xr.open_mfdataset('/ec/res4/hpcperm/itcv/analysis/forcing_1pct/'+exps[0]+'_forcing.nc')
    forcing = ds_forcing.forcing

    forcing_g = global_mean(forcing)

    amoc_dataset = clim_all['amoc_ts']
    amoc_dataset = {co: amoc_dataset[co] for co in amoc_dataset if amoc_dataset[co] is not None}
    amoc_dataset = create_ds_exp(amoc_dataset)
    amoc = amoc_dataset.sel(x=0)
    #amoc = amoc_dataset['msftyz']

    rho_dataset = clim_all[f'rho_clim']
    rho_dataset = {co: rho_dataset[co] for co in rho_dataset if rho_dataset[co] is not None}
    rho_dataset = create_ds_exp(rho_dataset)
    N2 = rho_dataset['Nsquared']

    vars = [tas, siconcN, shf, amoc]
    rolling = 30

    for var in vars:
        var_ref = var.sel(exp=ref_exp)
        var_exp = var.sel(exp=exps[0])
        
        trend, intercept,_,_,_ = stats.linregress(np.arange(0,len(var_ref)), var_ref)
        x = var_exp - (trend*np.arange(0,len(var_ref))+intercept)
        x = x.rolling(year=rolling).mean()[:150]

        print(var.name)
        print('Pi:', var_ref.mean(axis=0).values)
        print('Year 70:', x[85].values)
        print('Year 150:', x[-1].values)

    var3d = [N2]

    for var in var3d:
        var_ref = var.sel(exp=ref_exp)

        var_mean = global_mean_oce_3d(rho_dataset,ref_exp, 'itcv', 'Nsquared', depth_mean=True)['Nsquared']
        print(var.name)
        print('Pi:', var_mean)


    # compute internal variability in forced run
    # trend_30, intercept30,_,_,_ = stats.linregress(np.arange(0,30), z.sel(year=slice(1970,1999)))
    # z_int = z[120:] - (trend_30*np.arange(0,180)+intercept30)
    #print(z_int.mean(axis=0).values, z_int.std(axis=0).values)
    #print(y_ref.mean(axis=0).values, y_ref.sel(year=slice(1970,1999)).std(axis=0).values, z.rolling(year = rolling).mean()[149].values)

    #fig.savefig(cart_out + f'check_ts_anom_{'-'.join([exp for exp in exps])}.pdf')
    

def compute_cre(exps, user = None, read_again = [], cart_exp = '/ec/res4/scratch/{}/ece4/', cart_out = './output/', imbalance = 0., ref_exp = None, atm_only = False, atmvars = 'rsut rlut rsdt tas pr'.split(), ocevars = 'tos heatc qt_oce sos'.split(), icevars = 'siconc sivolu sithic'.split(), year_clim = None, plot_diffref=False, plot_param=False, param_map={}, skip_first_year=False, exp_type = 'PD', density=False):
    """
    Plots timeseries of var "vname" in domain "domain" for all exps.

    Domain is one among: ['atm', 'oce', 'ice']
    """
    if cart_out is None:
        raise ValueError('cart_out not specified!')
    
    if not os.path.exists(cart_out): os.mkdir(cart_out)

    cart_out_nc = cart_out + '/exps_clim/'
    cart_out_figs = cart_out + f'/check_{'-'.join(exps)}/'

    if not os.path.exists(cart_out_nc): os.mkdir(cart_out_nc)
    if not os.path.exists(cart_out_figs): os.mkdir(cart_out_figs)
    clim_all = read_output(exps, user = user, read_again = read_again, cart_exp = cart_exp, cart_out = cart_out_nc, atm_only = atm_only, atmvars = atmvars, ocevars = ocevars, icevars = icevars, year_clim = year_clim, density=density)

    toa_dataset = clim_all[f'cre_clim']

    toa_dataset = {co: toa_dataset[co] for co in toa_dataset if toa_dataset[co] is not None}

    if exps is None: exps = toa_dataset.keys()
    toa_dataset = create_ds_exp(toa_dataset)

    if isinstance(toa_dataset, xr.Dataset):
        cre = toa_dataset['rlntcs'] + toa_dataset[ 'rsnt']- toa_dataset[ 'rsntcs']
        #oce_dataset = oce_dataset['tos']

    #anomalies
    deltaCRE = global_mean((cre.sel(exp=exps[0]) - cre.sel(exp=exps[1])))#.sel(year =slice(1911, 1930))

    return deltaCRE

# ============================================================
################################################ MAIN FUNCTION ###########################

def compare_multi_exps_map(exps, user = None, read_again = [], cart_exp = '/ec/res4/scratch/{}/ece4/', cart_out = './output/', imbalance = 0., ref_exp = None, atm_only = False, atmvars = ''.split(), ocevars = ''.split(), icevars = 'siconc'.split(), year_clim = None, plot_diffref=False, plot_param=False, param_map={}, skip_first_year=False, exp_type = 'PD', density=False, density_only=False,colors=None):
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
    cart_out_figs = cart_out + f'/check_{'-'.join(exps)}/'

    ### read outputs for all exps
    clim_all = read_output_map(exps, user = user, read_again = read_again, cart_exp = cart_exp, cart_out = cart_out_nc, atm_only = atm_only, atmvars = atmvars, ocevars = ocevars, icevars = icevars, year_clim = year_clim, density=density, density_only=density_only)

    fig_siconc_map  = plot_var_map(clim_all, 'ice', 'siconc', ref_exp=ref_exp,cart_out = cart_out_figs, clevels=np.arange(-0.5,0.6,0.1))

    print(f'Done! Check results in {cart_out_figs}')


def compare_multi_exps(exps, user = None, read_again = [], cart_exp = '/ec/res4/scratch/{}/ece4/', cart_out = './output/', imbalance = 0., ref_exp = None, atm_only = False, atmvars = 'rsut rlut rsdt tas pr'.split(), ocevars = 'tos heatc qt_oce sos'.split(), icevars = 'siconc sivolu sithic'.split(), year_clim = None, plot_diffref=False, plot_param=False, param_map={}, skip_first_year=False, exp_type = 'PD', density=False, density_only=False,colors=None):
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
    cart_out_figs = cart_out + f'/check_{'-'.join(exps)}/'

    if not os.path.exists(cart_out_nc): os.mkdir(cart_out_nc)
    if not os.path.exists(cart_out_figs): os.mkdir(cart_out_figs)

    ### read outputs for all exps
    clim_all = read_output(exps, user = user, read_again = read_again, cart_exp = cart_exp, cart_out = cart_out_nc, atm_only = atm_only, atmvars = atmvars, ocevars = ocevars, icevars = icevars, year_clim = year_clim, density=density, density_only=density_only)

    coupled = False
    if 'amoc_ts' in clim_all: coupled = True

    allfigs = []
    ### Gregory and amoc gregory
    #fig_greg = plot_greg(clim_all['atm_mean'], exps, imbalance = imbalance, ylim = None, cart_out = cart_out_figs, exp_type = exp_type, colors=colors)
    # allfigs += [fig_greg]

    # if coupled:
    #      fig_amoc_greg = plot_amoc_vs_gtas(clim_all, exps, lw = 0.25, cart_out = cart_out_figs, exp_type = exp_type, colors=colors)
    #     allfigs.append(fig_amoc_greg)

    #     # fig_amoc_all = plot_amoc_2d_all(clim_all['amoc_mean'], exps, cart_out = cart_out_figs)
    #     # allfigs.append(fig_amoc_all)

    #fig_amoc_ts = plot_var_ts(clim_all, 'amoc', 'amoc', cart_out = cart_out_figs, rolling=1, colors=colors)

    # # Atm fluxes and zonal tas
    # figs_rad = plot_zonal_fluxes_vs_ceres(clim_all['atm_clim'], exps = exps, cart_out = cart_out_figs, colors=colors)
    # allfigs += figs_rad

    # fig_tas = plot_zonal_tas_vs_ref(clim_all['atm_clim'], exps = exps, ref_exp = ref_exp, cart_out = cart_out_figs, colors=colors)
    # allfigs.append(fig_tas)

    # if coupled:
    #     fig_tas = plot_zonal_tas_vs_ref(clim_all['atm_clim'], exps = exps, ref_exp = ref_exp, cart_out = cart_out_figs, colors=colors)
    #     allfigs.append(fig_tas)
    rolling=1
    #fig_tas2 = plot_var_ts(clim_all, 'atm', 'tas', cart_out = cart_out_figs, rolling=rolling, colors=colors)
    #fig_tas_map  = plot_var_map(clim_all, 'atm', 'tas', ref_exp=ref_exp,cart_out = cart_out_figs, clevels=np.arange(-5,6,1))
    #fig_toa_map  = plot_toa_map(clim_all, 'atm', 'tas', ref_exp=ref_exp,cart_out = cart_out_figs, clevels=np.arange(-5,6,1))

    #fig_ice_map  = plot_var_map(clim_all, 'ice', 'siconc', ref_exp=ref_exp,cart_out = cart_out_figs, clevels=np.arange(-5,6,1))

    #fig_tos = plot_var_ts(clim_all, 'oce', 'tos', cart_out = cart_out_figs, rolling=rolling, colors=colors)
    ##### CAN ADD NEW DIAGS HERE
    if coupled:
        rolling =  20
        fig_n2 = plot_var_ts_3d(clim_all, 'rho', 'Nsquared', cart_out = cart_out_figs, rolling=rolling, colors=colors)

        #fig_tas2 = plot_var_ts(clim_all, 'atm', 'tas', cart_out = cart_out_figs, rolling=rolling, colors=colors)
        #fig_tos = plot_var_ts(clim_all, 'oce', 'tos', cart_out = cart_out_figs, rolling=rolling, colors=colors)
        # fig_heatc = plot_var_ts(clim_all, 'oce', 'heatc', cart_out = cart_out_figs, rolling=rolling, colors=colors)
        # fig_qtoce = plot_var_ts(clim_all, 'oce', 'qt_oce', cart_out = cart_out_figs, rolling=rolling, colors=colors)
        # fig_enebal = plot_var_ts(clim_all, 'oce', 'enebal', cart_out = cart_out_figs, rolling=rolling, colors=colors)
        # fig_siv =plot_var_ts(clim_all, 'ice', 'sivolu_N', cart_out = cart_out_figs, rolling=rolling, colors=colors)
        # fig_sic = plot_var_ts(clim_all, 'ice', 'siconc_N', cart_out = cart_out_figs, rolling=rolling, colors=colors)
        # fig_siv2 = plot_var_ts(clim_all, 'ice', 'sivolu_S', cart_out = cart_out_figs, rolling=rolling, colors=colors)
        # fig_sic2 = plot_var_ts(clim_all, 'ice', 'siconc_S', cart_out = cart_out_figs, rolling=rolling, colors=colors)
        # allfigs += [fig_tos, fig_heatc, fig_qtoce, fig_enebal, fig_siv, fig_sic, fig_siv2, fig_sic2]
        if density:
            #fig_rho = plot_var_ts_3d(clim_all, 'rho', 'density', cart_out = cart_out_figs, rolling=rolling)
            #fig_den = plot_var_profile(clim_all, 'rho', 'density',  cart_out = cart_out_figs, colors=colors)
            #fig_n2 = plot_var_profile(clim_all, 'rho', 'Nsquared', ref_exp=ref_exp, vcoord='depth_mid', cart_out = cart_out_figs, colors=colors)
            fig_n2so = plot_var_region(clim_all, 'rho', 'Nsquared',[30,60],ref_exp=ref_exp, vcoord='depth_mid', cart_out = cart_out_figs, colors=colors,cart_exp = cart_exp)
            #fig_n2zonal  = plot_zonal_profile(clim_all, 'rho', 'Nsquared', ref_exp=ref_exp, vcoord='depth_mid', cart_out = cart_out_figs, colors=colors)
            #fig_siconc_map  = plot_var_map(clim_all, 'ice', 'siconc', ref_exp=ref_exp,cart_out = cart_out_figs, clevels=np.arange(-0.5,0.6,0.1))
            #fig_tos_map  = plot_var_map(clim_all, 'atm', 'tas', ref_exp=ref_exp,cart_out = cart_out_figs, clevels=np.arange(-1,1.1,0.1))
            #fig_cre = plot_cre_zonal_map(clim_all, exps, ref_exp=ref_exp, cart_out = cart_out_figs)
            #fig_n2_k = plot_zonal_ohue_correlation(clim_all, 'rho', 'Nsquared',vcoord='depth_mid', cart_out = cart_out_figs)
            
            #allfigs += [fig_n2]
            print('ciao')

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
    
    """
    if coupled:
        if density:
            allfigs = [fig_greg, fig_amoc_greg] + figs_rad + [fig_tas, fig_tas2] + [fig_tos, fig_heatc, fig_qtoce, fig_enebal, fig_siv, fig_sic, fig_siv2, fig_sic2] + [fig_rho, fig_den, fig_n2]
        else:
            allfigs = [fig_greg, fig_amoc_greg] + figs_rad + [fig_tas, fig_tas2] + [fig_tos, fig_heatc, fig_qtoce, fig_enebal, fig_siv, fig_sic, fig_siv2, fig_sic2] 

    else:
        allfigs = [fig_greg] + figs_rad
    """
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