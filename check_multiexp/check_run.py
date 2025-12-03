import xarray as xr
from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd

import matplotlib.cm as cm
from matplotlib.patches import Patch
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

import glob

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
        ds_mean = ds.groupby('time.year').mean().groupby('lat').mean().weighted(weights).mean('lat')
    else:
        ds_mean = ds.groupby('lat').mean().weighted(weights).mean('lat')
    
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


def global_mean_oce_3d(ds, exp, user, cart_exp = cart_exp, compute = True, grid = 'T'):

    ## Account for variable area at depth
    return


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


def file_list(exp, user, cart_exp = '/ec/res4/scratch/{}/ece4/', remove_last_year = False, coupled = True):
    cart = f'{cart_exp.format(user)}/{exp}/output/oifs/'
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

    return filz_exp, filz_nemo, filz_amoc, filz_ice


def read_output(exps, user = None, read_again = [], cart_exp = cart_exp, cart_out = cart_out, atmvars = 'rsut rlut rsdt tas pr'.split(), ocevars = 'tos heatc qt_oce sos'.split(), icevars = 'siconc sivolu sithic'.split(), atm_only = False, year_clim = None):
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
    filz_nemo = dict()
    filz_ice = dict()

    for exp, us in zip(exps, user):
        filz_exp[exp], filz_nemo[exp], filz_amoc[exp], filz_ice[exp] = file_list(exp, us, cart_exp = cart_exp)

    atmmean_exp = dict()
    atmclim_exp = dict()
    oceclim_exp = dict()
    ocemean_exp = dict()
    amoc_mean_exp = dict()
    amoc_ts_exp = dict()
    iceclim_exp = dict()
    icemean_exp = dict()

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
        
        elif os.path.exists(cart_out + f'clim_tuning_{exp}.nc') and exp in read_again:
            print('Updating existing diagnostics with new data...')
            
            # Load existing data
            atmclim_old = xr.load_dataset(cart_out + f'clim_tuning_{exp}.nc')
            atmmean_old = xr.load_dataset(cart_out + f'mean_tuning_{exp}.nc')
            
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
            else:
                print(f'Found {len(ds_new.time_counter)} new time steps')
                # Compute diagnostics for new data only
                atmclim_new, atmmean_new = compute_atm_clim(ds_new, exp, cart_out = None, atmvars = atmvars, year_clim = year_clim)
                
                # Concatenate old and new data
                atmclim_exp[exp] = atmclim_new
                atmmean_exp[exp] = xr.concat([atmmean_old, atmmean_new], dim='year')
                
                # Save updated data
                atmclim_exp[exp].to_netcdf(cart_out + f'clim_tuning_{exp}.nc')
                atmmean_exp[exp].to_netcdf(cart_out + f'mean_tuning_{exp}.nc')
                    
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

            if coupled:
                # OCE CLIM
                ds = xr.open_mfdataset(filz_nemo[exp], use_cftime=True, chunks = {'time_counter': 240})
                oceclim_exp[exp], ocemean_exp[exp] = compute_oce_clim(ds, exp, us, cart_exp = cart_exp, cart_out = cart_out, ocevars = ocevars, year_clim = year_clim)

                ds = xr.open_mfdataset(filz_ice[exp], use_cftime=True, chunks = {'time_counter': 240})
                iceclim_exp[exp], icemean_exp[exp] = compute_ice_clim(ds, exp, us, cart_exp = cart_exp, cart_out = cart_out, icevars = icevars, year_clim = year_clim)

                ds = xr.open_mfdataset(filz_amoc[exp], use_cftime=True, chunks = {'time_counter': 240})
                amoc_mean_exp[exp], amoc_ts_exp[exp] = compute_amoc_clim(ds, exp, cart_out = cart_out, year_clim = year_clim)

    clim_all = dict()
    clim_all['atm_clim'] = atmclim_exp
    clim_all['atm_mean'] = atmmean_exp
    if coupled:
        clim_all['oce_clim'] = oceclim_exp
        clim_all['oce_mean'] = ocemean_exp
        clim_all['ice_clim'] = iceclim_exp
        clim_all['ice_mean'] = icemean_exp
        clim_all['amoc_mean'] = amoc_mean_exp

        # if 'time_counter' in clim_all['amoc_ts']['pal3'].dims:
        #     amoc_ts_exp = amoc_ts_exp.groupby('time_counter.year').mean()
        clim_all['amoc_ts'] = amoc_ts_exp

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
        ax.scatter(x, y, s = 1000, color = col, marker = 'o', alpha = 0.5)
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
        ax.scatter(x, y, s = 1000, color = col, marker = 'o', edgecolors = col, alpha = 0.5)
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
        ax.scatter(x, y, s = 1000, color = col, marker = 'o', edgecolors = col, alpha = 0.5)
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


def plot_zonal_tas_vs_ref(atmclim, exps, ref_exp = None, cart_out = cart_out):
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

def plot_var_ts(clim_all, domain, vname, exps = None, ref_exp = None, rolling = None, norm_factor = 1., cart_out = cart_out):
    """
    Plots timeseries of var "vname" in domain "domain" for all exps.

    Domain is one among: ['atm', 'oce', 'ice']
    """

    if domain not in ['atm', 'oce', 'ice', 'amoc']:
        raise ValueError('domain should be one among: atm, oce, ice, amoc')
    
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

# ============================================================
################################################ MAIN FUNCTION ###########################

def compare_multi_exps(exps, user = None, read_again = [], cart_exp = '/ec/res4/scratch/{}/ece4/', cart_out = './output/', imbalance = 0., ref_exp = None, atm_only = False, atmvars = 'rsut rlut rsdt tas pr'.split(), ocevars = 'tos heatc qt_oce sos'.split(), icevars = 'siconc sivolu sithic'.split(), year_clim = None, plot_diffref=False, plot_param=False, param_map={}, skip_first_year=False, exp_type = 'PD'):
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
    clim_all = read_output(exps, user = user, read_again = read_again, cart_exp = cart_exp, cart_out = cart_out_nc, atm_only = atm_only, atmvars = atmvars, ocevars = ocevars, icevars = icevars, year_clim = year_clim)

    coupled = False
    if 'amoc_ts' in clim_all: coupled = True

    ### Gregory and amoc gregory
    fig_greg = plot_greg(clim_all['atm_mean'], exps, imbalance = imbalance, ylim = None, cart_out = cart_out_figs, exp_type = exp_type)
    allfigs = [fig_greg]

    if coupled:
        fig_amoc_greg = plot_amoc_vs_gtas(clim_all, exps, lw = 0.25, cart_out = cart_out_figs, exp_type = exp_type)
        allfigs.append(fig_amoc_greg)

        fig_amoc_all = plot_amoc_2d_all(clim_all['amoc_mean'], exps, cart_out = cart_out_figs)
        allfigs.append(fig_amoc_all)

        fig_amoc_ts = plot_var_ts(clim_all, 'amoc', 'amoc', cart_out = cart_out_figs, rolling=None)

    # Atm fluxes and zonal tas
    figs_rad = plot_zonal_fluxes_vs_ceres(clim_all['atm_clim'], exps = exps, cart_out = cart_out_figs)
    allfigs += figs_rad

    fig_tas = plot_zonal_tas_vs_ref(clim_all['atm_clim'], exps = exps, ref_exp = ref_exp, cart_out = cart_out_figs)
    allfigs.append(fig_tas)

    if coupled:
        fig_tas = plot_zonal_tas_vs_ref(clim_all['atm_clim'], exps = exps, ref_exp = ref_exp, cart_out = cart_out_figs)
        allfigs.append(fig_tas)

    ###### CAN ADD NEW DIAGS HERE
    if coupled:
        rolling = 20 # None
        fig_tos = plot_var_ts(clim_all, 'oce', 'tos', cart_out = cart_out_figs, rolling=rolling)
        fig_heatc = plot_var_ts(clim_all, 'oce', 'heatc', cart_out = cart_out_figs, rolling=rolling)
        fig_qtoce = plot_var_ts(clim_all, 'oce', 'qt_oce', cart_out = cart_out_figs, rolling=rolling)
        fig_enebal = plot_var_ts(clim_all, 'oce', 'enebal', cart_out = cart_out_figs, rolling=rolling)
        fig_siv =plot_var_ts(clim_all, 'ice', 'sivolu_N', cart_out = cart_out_figs, rolling=rolling)
        fig_sic = plot_var_ts(clim_all, 'ice', 'siconc_N', cart_out = cart_out_figs, rolling=rolling)
        fig_siv2 = plot_var_ts(clim_all, 'ice', 'sivolu_S', cart_out = cart_out_figs, rolling=rolling)
        fig_sic2 = plot_var_ts(clim_all, 'ice', 'siconc_S', cart_out = cart_out_figs, rolling=rolling)
        allfigs += [fig_tos, fig_heatc, fig_qtoce, fig_enebal, fig_siv, fig_sic, fig_siv2, fig_sic2]

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