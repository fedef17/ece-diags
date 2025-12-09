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
    #areas = areas.rename({f'x_ORCA2-T': 'x', f'y_ORCA2-T': 'y'}) # for 4.1.0
    gname = [nam for nam in areas.data_vars if '-T' in nam]

    if len(gname) > 1:
        raise ValueError(f'Too many grid names matching: {gname}')

    ocean_area = areas[gname[0]].values

    return ocean_area

def get_mask_nemo(exp, user, cart_exp = cart_exp, grid = 'T'):
    #ocean areas
    masks = xr.load_dataset(cart_exp.format(user) + f'/{exp}/masks.nc')

    gname = [nam for nam in masks.data_vars if '-T' in nam]
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


def global_mean_oce_3d(ds, exp, user, vars, cart_exp = cart_exp, compute = True, grid = 'T', singlelevel = False, lev = 0):
    
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
    ds = ds[atmvars].groupby('time.year').mean()

    if year_clim is None:
        print('Using last 20 years for climatology')
        atmclim = ds.isel(year = slice(-20, None)).mean('year').compute()
    else:
        atmclim = ds.sel(year = slice(year_clim[0], year_clim[1])).mean('year').compute()
    atmmean = global_mean(ds, compute = True)
    atmclim.to_netcdf(cart_out + f'clim_tuning_{exp}.nc')
    atmmean.to_netcdf(cart_out + f'mean_tuning_{exp}.nc')

    return atmclim, atmmean


def compute_oce_clim(ds, exp, user, cart_exp = cart_exp, cart_out = cart_out, ocevars = 'tos heatc qt_oce sos'.split(), year_clim = None, grid = 'T'):
    ds = ds.rename({'time_counter': 'time'})
    # print(ds.data_vars)
    ds = ds[ocevars].groupby('time.year').mean()
    #ds = ds.rename({f'x_grid_{grid}_inner': 'x', f'y_grid_{grid}_inner': 'y'})
    #ds = ds.rename({f'x_grid_{grid}': 'x', f'y_grid_{grid}': 'y'}) # for 4.1.0

    if year_clim is None:
        print('Using last 20 years for climatology')
        oceclim = ds.isel(year = slice(-20, None)).mean('year').compute()
    else:
        oceclim = ds.sel(year = slice(year_clim[0], year_clim[1])).mean('year').compute()

    oceclim.to_netcdf(cart_out + f'clim_oce_tuning_{exp}.nc')

    ocemean = global_mean_oce_2d(ds, exp, user, cart_exp, compute = True)
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

    iceclim.to_netcdf(cart_out + f'clim_ice_tuning_{exp}.nc')

    icemean = global_mean_ice(ds, exp, user, cart_exp, compute = True)
    icemean.to_netcdf(cart_out + f'mean_ice_tuning_{exp}.nc')

    return iceclim, icemean


def compute_amoc_clim(ds, exp, cart_out = cart_out, year_clim = None):
    amoc_ts = calc_amoc_ts(ds, plot = False)

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
    amoc_max = amoc_yearly.max(dim=['depthw', 'y'])
    
    # Plot timeseries
    if plot:
        amoc_max.plot(ylim=ylim, label = exp_name, ax = ax)

    return amoc_max


##################################### READ OUTPUTS ################################


def file_list_(exp, user, cart_exp = '/ec/res4/scratch/{}/ece4/', remove_last_year = False, coupled = True):
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

def file_list(exp, user, cart_exp = '/ec/res4/scratch/{}/ece4/', remove_last_year = False, coupled = True, density= False):
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
        
    if density:
        filz_rho = '../density/density_fields/' + f'{exp}/{exp}_*_density.nc'

        return    filz_exp, filz_nemo, filz_amoc, filz_ice, filz_rho
    
    else:
        return filz_exp, filz_nemo, filz_amoc, filz_ice



def read_output(exps, user = None, read_again = [], cart_exp = cart_exp, cart_out = cart_out, atmvars = 'rsut rlut rsdt tas pr'.split(), ocevars = 'tos heatc qt_oce sos'.split(), icevars = 'siconc sivolu sithic'.split(), atm_only = False, year_clim = None, density=False):
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

    if density:
        for exp, us in zip(exps, user):
            filz_exp[exp], filz_nemo[exp], filz_amoc[exp], filz_ice[exp], filz_rho[exp] = file_list(exp, us, cart_exp = cart_exp, density=True)
    else:
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
    rhomean_exp = dict()
    rhoclim_exp = dict()

    for exp, us in zip(exps, user):
        print(exp)
        coupled = False
        if not atm_only: 
            if len(glob.glob(filz_nemo[exp])) > 0:
                print('coupled')
                coupled = True
            else:
                print(f'NO files matching pattern: {filz_nemo[exp]}. Assuming atm-only')

        if os.path.exists(cart_out + f'clim_tuning_{exp}.nc') and exp not in read_again:
            print('Already computed, reading clim..')
            atmclim_exp[exp] = xr.open_dataset(cart_out + f'clim_tuning_{exp}.nc')
            atmmean_exp[exp] = xr.open_dataset(cart_out + f'mean_tuning_{exp}.nc')

            if coupled:
                amoc_ts_exp[exp] = xr.open_dataset(cart_out + f'amoc_ts_tuning_{exp}.nc')
                amoc_mean_exp[exp] = xr.open_dataset(cart_out + f'amoc_2d_tuning_{exp}.nc')
                if os.path.exists(cart_out + f'clim_oce_tuning_{exp}.nc'):
                    oceclim_exp[exp] = xr.open_dataset(cart_out + f'clim_oce_tuning_{exp}.nc')
                    ocemean_exp[exp] = xr.open_dataset(cart_out + f'mean_oce_tuning_{exp}.nc')
                    iceclim_exp[exp] = xr.open_dataset(cart_out + f'clim_ice_tuning_{exp}.nc')
                    icemean_exp[exp] = xr.open_dataset(cart_out + f'mean_ice_tuning_{exp}.nc')
                else: # legacy for old exps
                    oceclim_exp[exp] = xr.open_dataset(cart_out + f'oce_tuning_{exp}.nc')
                    ocemean_exp[exp] = None
                
                if os.path.exists(cart_out + f'clim_rho_tuning_{exp}.nc'):
                    rhoclim_exp[exp] = xr.open_dataset(cart_out + f'clim_rho_tuning_{exp}.nc')
                    rhomean_exp[exp] = xr.open_dataset(cart_out + f'mean_rho_tuning_{exp}.nc')
                    density=True
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
    if coupled:
        clim_all['oce_clim'] = oceclim_exp
        clim_all['oce_mean'] = ocemean_exp
        clim_all['ice_clim'] = iceclim_exp
        clim_all['ice_mean'] = icemean_exp
        clim_all['amoc_mean'] = amoc_mean_exp
        clim_all['amoc_ts'] = amoc_ts_exp
    if density:
        clim_all['rho_mean'] = rhomean_exp

    return clim_all


def create_ds_exp(exp_dict):
    x_ds = xr.concat(exp_dict.values(), dim=pd.Index(exp_dict.keys(), name='exp'))
    return x_ds


####################################### PLOTS #######################################


def plot_amoc_2d(amoc_mean, exp = None, ax = None):
    if ax is None:
        fig, ax = plt.subplots(figsize = (12,8))

    if isinstance(amoc_mean, xr.Dataset):
        amoc_mean = amoc_mean['msftyz']

    amoc_mean.sel(basin = 2).plot.contourf(x = 'nav_lat', y = 'depthw', ylim = (3000, 0), levels = np.arange(-15, 15, 2), ax = ax)
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


def plot_greg(atmmean_exp, exps, cart_out = cart_out, tas_clim = 287.29, net_toa_clim = 0.6, n_end = 20, imbalance = -0.9, ylim = None, colors = None):
    """
    gregory plot
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    ax.fill_betweenx(np.arange(0., 3.5, 0.1), tas_clim - 0.3, tas_clim + 0.3, color = 'grey', alpha = 0.2, edgecolor = None)
    ax.fill_between(np.arange(286.5, 288, 0.1), net_toa_clim - imbalance - 0.3, net_toa_clim - imbalance + 0.3, color = 'grey', alpha = 0.2, edgecolor = None)

    if colors is None:
        colors = get_colors(exps)

    for exp, col in zip(exps, colors):
        ax.plot(atmmean_exp[exp].tas, atmmean_exp[exp].toa_net, label = exp, lw = 0.2, color = col)
        #ax.scatter(atmmean_exp[exp].tas.sel(year = slice(1990, 2000)).mean(), atmmean_exp[exp].toa_net.sel(year = slice(1990, 2000)).mean(), s = 1000, color = 'red', marker = 'o')
        x, y = atmmean_exp[exp].tas.isel(year = slice(-n_end, None)).mean(), atmmean_exp[exp].toa_net.isel(year = slice(-n_end, None)).mean()
        ax.scatter(x, y, s = 1000, color = col, marker = 'o', alpha = 0.5)
        ax.text(x+0.1, y+0.1, exp, fontsize=12, ha='right', color = col)

    ax.set_xlabel('GTAS (K)')
    ax.set_ylabel('net TOA (W/m$^2$)')
    plt.legend()

    if ylim is not None:
        ax.set_ylim(ylim)

    fig.savefig(cart_out + f'check_tuning_{'-'.join(exps)}.pdf')
    plt.show()

    return fig



def plot_amoc_vs_gtas(clim_all, exps = None, cart_out = cart_out, tas_clim = 287.29, n_end = 20, colors = None, labels = None, colors_legend = None, lw = 0.3, alpha = 0.5, background_color = None):
    fig, ax = plt.subplots(figsize=(12, 8))

    ax.fill_betweenx(np.arange(4, 21, 0.1), tas_clim - 0.3, tas_clim + 0.3, color = 'grey', alpha = 0.2, edgecolor = None)
    ax.fill_between(np.arange(286, 289, 0.1), 15, 20, color = 'grey', alpha = 0.2, edgecolor = None)

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
        y = y.groupby('time_counter.year').mean().squeeze()

        ax.plot(x, y, label = exp, lw = lw, color = col)
        
        x, y = x.isel(year = slice(-n_end, None)).mean(), y.isel(year = slice(-n_end, None)).mean()
        ax.scatter(x, y, s = 1000, color = col, marker = 'o', edgecolors = col, alpha = 0.5)
        ax.text(x+0.1, y+0.1, exp, fontsize=12, ha='right', color = col)

    ax.set_xlabel('GTAS (K)')
    ax.set_ylabel('AMOC max (Sv)')

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


def plot_map_ocean(oce_clim, exps, var, ref_exp = None, vmin = None, vmax = None, xlabel = None, ylabel = None):
    """
    oce_clim is clim_all['oce_clim'] produced by read_output
    exps: list of experiments
    var: var name to plot
    ref_exp: if specified, plot differences to ref_exp

    TO BE IMPROVED: regrid and plot with cartopy

    """
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
    else:
        ax.set_ylabel('zonal temp (K)')

    fig.savefig(cart_out + f'check_zonal_tas_{'-'.join([exp for exp in exps if exp != ref_exp])}_vs_{ref_exp}.pdf')

    return fig


def plot_var_ts(clim_all, domain, vname, exps = None, ref_exp = None, rolling = None, norm_factor = 1., cart_out = cart_out):
    """
    Plots timeseries of var "vname" in domain "domain" for all exps.

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

    colors = get_colors(exps)

    for exp, col in zip(exps, colors):
        y = norm_factor*ts_dataset.sel(exp = exp)[vname]
        
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

def plot_var_ts_3d(clim_all, domain, vname, exps = None, ref_exp = None, rolling = None, norm_factor = 1., cart_out = cart_out):
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

def plot_var_profile(clim_all, domain, vname, vcoord='deptht', exps = None, ref_exp = None, norm_factor = 1., cart_out = cart_out):
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
    y_ref = None
    if ref_exp is not None:
        y_ref = norm_factor*ts_dataset.sel(exp = ref_exp)[vname]

    colors = get_colors(exps)

    for exp, col in zip(exps, colors):
        y = norm_factor*ts_dataset.sel(exp = exp)[vname]
    
        if (vcoord == 'depth_mid'):
            v_levels = ts_dataset.sel(exp = exp)['density']['deptht']
            levels = (v_levels[1:].values + v_levels[:-1].values)/2
        else:
            levels = ts_dataset.sel(exp = exp)[vname][vcoord]

        if y_ref is not None: y = y - y_ref

        ax.plot(y.mean(axis=0), -levels, label=exp, color= col)
        # ax.text(y.year[-1] + 5, np.nanmean(y.values[-30:]), exp, fontsize=12, ha='right', color = col) # not working for some evil reason
    
    ax.set_title('')
    ax.legend()
    ax.set_ylabel('Depth (m)')

    power = 1/2  # o 1/1.5
    fwd = lambda y: np.sign(y) * (abs(y) ** power)
    inv = lambda y: np.sign(y) * (abs(y) ** (1/power))
    ax.set_yscale('function', functions=(fwd, inv))

    fig.savefig(cart_out + f'check_profile_{domain}_{vname}_{'-'.join([exp for exp in exps])}.pdf')
    
    return fig

def check_energy_balance_ocean(clim_all, remove_ice_formation = False):
    fact = 334*1000*1000/(3.1e7*4*3.14*6e6**2) # to convert sea ice formation in W/m2

    # (clim_all['oce_mean'][exp]['enebal']+clim_all['ice_mean'][exp]['sivolu_N'].diff('year')*fact).rolling(year = 20).mean().plot(label = exp, color = col, ls = ':')
    return

################################################ MAIN FUNCTION ###########################


def compare_multi_exps(exps, user = None, read_again = [], cart_exp = '/ec/res4/scratch/{}/ece4/', cart_out = './output/', imbalance = 0., ref_exp = None, atm_only = False, atmvars = 'rsut rlut rsdt tas pr'.split(), ocevars = 'tos heatc qt_oce sos'.split(), icevars = 'siconc sivolu sithic'.split(), year_clim = None, density=False):
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
    clim_all = read_output(exps, user = user, read_again = read_again, cart_exp = cart_exp, cart_out = cart_out_nc, atm_only = atm_only, atmvars = atmvars, ocevars = ocevars, icevars = icevars, year_clim = year_clim, density=density)

    coupled = False
    if 'amoc_ts' in clim_all: coupled = True

    ## Gregory and amoc gregory
    fig_greg = plot_greg(clim_all['atm_mean'], exps, imbalance = imbalance, ylim = None, cart_out = cart_out_figs)
    if coupled:
        fig_amoc_greg = plot_amoc_vs_gtas(clim_all, exps, lw = 0.25, cart_out = cart_out_figs)

    # Atm fluxes and zonal tas
    figs_rad = plot_zonal_fluxes_vs_ceres(clim_all['atm_clim'], exps = exps, cart_out = cart_out_figs)

    if coupled:
        fig_tas = plot_zonal_tas_vs_ref(clim_all['atm_clim'], exps = exps, ref_exp = ref_exp, cart_out = cart_out_figs)

    ##### CAN ADD NEW DIAGS HERE
    if coupled:
        rolling =  20
        fig_tas2 = plot_var_ts(clim_all, 'atm', 'tas', cart_out = cart_out_figs, rolling=rolling)
        fig_tos = plot_var_ts(clim_all, 'oce', 'tos', cart_out = cart_out_figs, rolling=rolling)
        fig_heatc = plot_var_ts(clim_all, 'oce', 'heatc', cart_out = cart_out_figs, rolling=rolling)
        fig_qtoce = plot_var_ts(clim_all, 'oce', 'qt_oce', cart_out = cart_out_figs, rolling=rolling)
        fig_enebal = plot_var_ts(clim_all, 'oce', 'enebal', cart_out = cart_out_figs, rolling=rolling)
        fig_siv =plot_var_ts(clim_all, 'ice', 'sivolu_N', cart_out = cart_out_figs, rolling=rolling)
        fig_sic = plot_var_ts(clim_all, 'ice', 'siconc_N', cart_out = cart_out_figs, rolling=rolling)
        fig_siv2 = plot_var_ts(clim_all, 'ice', 'sivolu_S', cart_out = cart_out_figs, rolling=rolling)
        fig_sic2 = plot_var_ts(clim_all, 'ice', 'siconc_S', cart_out = cart_out_figs, rolling=rolling)
        
        fig_rho = plot_var_ts_3d(clim_all, 'rho', 'density', cart_out = cart_out_figs, rolling=rolling)
        fig_den = plot_var_profile(clim_all, 'rho', 'density', cart_out = cart_out_figs)
        fig_n2 = plot_var_profile(clim_all, 'rho', 'Nsquared', vcoord='depth_mid', cart_out = cart_out_figs)
    if coupled:
        allfigs = [fig_greg, fig_amoc_greg] + figs_rad + [fig_tas, fig_tas2] + [fig_tos, fig_heatc, fig_qtoce, fig_enebal, fig_siv, fig_sic, fig_siv2, fig_sic2] + [fig_rho, fig_den, fig_n2]
    else:
        allfigs = [fig_greg] + figs_rad

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

    if user is None:
        user = os.getenv('USER')
    
    # Example: Print loaded configuration
    print(f"Experiments: {exps}")
    print(f"User: {user}")
    print(f"Read again: {read_again}")
    print(f"Cart exp: {cart_exp}")
    print(f"Cart out: {cart_out}")

    clim_all, figs = compare_multi_exps(exps, user = user, read_again = read_again, cart_exp = cart_exp, cart_out = cart_out, imbalance = imbalance, ref_exp = ref_exp)

    return clim_all, figs
    

# Main execution
if __name__ == '__main__':
    main()