import xarray as xr
from matplotlib import pyplot as plt
import numpy as np
import os
import netCDF4
import io
import pandas as pd

import matplotlib.cm as cm
from matplotlib.patches import Patch
from matplotlib.colors import Normalize
import glob
import re
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.util import add_cyclic_point
import matplotlib.ticker as mticker
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba

import yaml
import argparse
import fcntl

import logging
from smmregrid import Regridder, CdoGenerate

logger = logging.getLogger(__name__) 

########################################################################################

def _debug_check(fig_or_figs, label):
    """DEBUG AID: force-render figure(s) right away so the log pinpoints which
    plot call raises 'Image size ... too large' (remove once the cause is found)."""
    figs = fig_or_figs if isinstance(fig_or_figs, (list, tuple)) else [fig_or_figs]
    for i, fig in enumerate(figs):
        if fig is None:
            continue
        tag = label if len(figs) == 1 else f'{label}[{i}]'
        try:
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight')
            logging.info(f'[DEBUG-PLOT] ok: {tag} (figsize={fig.get_size_inches()}, dpi={fig.dpi})')
        except Exception as e:
            logging.error(f'[DEBUG-PLOT] CRASHED: {tag} -> {e}')
            raise

datadir = '../data/'
cart_out = './output/'
cart_exp = '/ec/res4/scratch/{}/ece4/'

time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)

# years in the climatologies when year_clim is None; also the window of the Gregory big dots and of the boxplots
CLIM_YEARS = 20

######################################################################################
# Long names and units used in plot labels.
# Derived variables (toa_net, srf_net, ...) inherit the attributes of their first operand
# in xarray, so their saved long_name/units are wrong: this table takes precedence over attrs.

VAR_LONG_NAMES = {
    # atmosphere
    'rsut': 'TOA outgoing shortwave radiation',
    'rlut': 'TOA outgoing longwave radiation',
    'rsdt': 'TOA incident shortwave radiation',
    'rsnt': 'TOA net shortwave radiation',
    'rlnt': 'TOA net longwave radiation',
    'rsntcs': 'TOA net shortwave radiation, clear sky',
    'rlntcs': 'TOA net longwave radiation, clear sky',
    'toa_net': 'Net TOA radiation',
    'crf': 'Cloud radiative effect (TOA)',
    'tas': 'Near-surface air temperature',
    'pr': 'Precipitation',
    'prsn': 'Snowfall',
    'snvolu': 'Snow volume on sea ice',
    'hfxerr': 'SI3 heat-flux error',
    'evspsbl': 'Evaporation',
    'E-P': 'Evaporation minus precipitation',
    'alb': 'Albedo',
    'ps': 'Surface air pressure',
    'hfss': 'Surface upward sensible heat flux',
    'hfls': 'Surface upward latent heat flux',
    'rsns': 'Surface net shortwave radiation',
    'rlns': 'Surface net longwave radiation',
    'srf_net': 'Net surface heat flux',
    'srf_net_nosn': 'Net surface heat flux (no snowfall melting)',
    'srf_net_oce': 'Net surface heat flux over ocean',
    'srf_net_oce_nosn': 'Net surface heat flux over ocean (no snowfall melting)',
    'srf_net_land': 'Net surface heat flux over land',
    'toa-srf': 'Net TOA minus net surface flux',
    'atm_imb': 'Atmospheric energy imbalance (>0: source)',
    'atm_imb_bottino': 'Atmospheric energy imbalance, no snowfall over ocean (>0: source)',
    # ocean
    'tos': 'Sea surface temperature',
    'sos': 'Sea surface salinity',
    'zos': 'Sea surface height',
    'heatc': 'Ocean heat content',
    'qt_oce': 'Net heat flux into the ocean',
    'mldr10_1': 'Mixed layer depth',
    'oce_imb': 'Ocean energy imbalance (>0: source)',
    'interface_imb': 'Atmosphere-ocean interface imbalance (>0: source)',
    'density': 'Sea water density',
    'Nsquared': 'Brunt-Vaisala frequency squared',
    # sea ice
    'siconc': 'Sea-ice concentration',
    'sivolu': 'Sea-ice volume per area',
    'sithic': 'Sea-ice thickness',
    # amoc
    'amoc': 'AMOC strength',
    'msftyz': 'AMOC strength',
}

# units of the fields as stored per grid point (maps, zonal means, global means)
VAR_UNITS = {
    **{v: 'W m$^{-2}$' for v in ['rsut', 'rlut', 'rsdt', 'rsnt', 'rlnt', 'rsntcs', 'rlntcs',
                                 'toa_net', 'crf', 'hfss', 'hfls', 'rsns', 'rlns', 'srf_net',
                                 'srf_net_nosn', 'srf_net_oce', 'srf_net_oce_nosn', 'srf_net_land',
                                 'toa-srf', 'atm_imb', 'atm_imb_bottino', 'qt_oce', 'oce_imb',
                                 'interface_imb']},
    **{v: 'kg m$^{-2}$ s$^{-1}$' for v in ['pr', 'prsn', 'evspsbl', 'E-P']},
    'tas': 'K', 'alb': '0-1', 'ps': 'Pa',
    'tos': '°C', 'sos': 'psu', 'zos': 'm', 'heatc': 'J m$^{-2}$', 'mldr10_1': 'm',
    'density': 'kg m$^{-3}$', 'Nsquared': 's$^{-2}$',
    'siconc': '0-1', 'sivolu': 'm', 'sithic': 'm', 'snvolu': 'm', 'hfxerr': 'W m$^{-2}$',
    'amoc': 'Sv', 'msftyz': 'Sv',
}

# global_mean_oce_2d / global_mean_ice integrate (not average) these variables:
# names and units of the resulting time series
VAR_LONG_NAMES_TS = {
    'oce': {'zos': 'Sea surface height (area-integrated)', 'mldr10_1': 'Mixed layer depth (area-integrated)'},
    'ice': {'siconc': 'Sea-ice area', 'sivolu': 'Sea-ice volume', 'sithic': 'Mean sea-ice thickness',
            'snvolu': 'Snow volume on sea ice', 'hfxerr': 'SI3 heat-flux error, mean over sea ice'},
}
VAR_UNITS_TS = {
    'oce': {'zos': 'm$^3$', 'heatc': 'J', 'mldr10_1': 'm$^3$'},
    'ice': {'siconc': 'm$^2$', 'sivolu': 'm$^3$', 'sithic': 'm', 'snvolu': 'm$^3$', 'hfxerr': 'W m$^{-2}$'},
}

def var_label(var, da = None, units = True, domain = None):
    """
    Plot label for variable `var`: long name, plus units in brackets if units=True.
    Looks up VAR_LONG_NAMES/VAR_UNITS first, then the attrs of `da`, then falls back to `var`.
    Hemispheric suffixes (_N, _S) become (NH)/(SH). If `domain` is given ('oce', 'ice'), the units
    of area-integrated time series are used.
    """
    base, hemi = var, ''
    if var[-2:] in ('_N', '_S'):
        base, hemi = var[:-2], {'_N': 'NH', '_S': 'SH'}[var[-2:]]

    attrs = getattr(da, 'attrs', {}) or {}
    name = (VAR_LONG_NAMES_TS.get(domain, {}).get(base) or VAR_LONG_NAMES.get(base)
            or attrs.get('long_name') or var)
    if hemi:
        name += f' ({hemi})'
    if not units:
        return name

    ts_units = VAR_UNITS_TS.get(domain, {})
    unit = ts_units[base] if base in ts_units else (VAR_UNITS.get(base) or attrs.get('units'))
    return f'{name} [{unit}]' if unit else name

######################################################################################

# Optional short descriptions shown in the legends, e.g. {'xb09': '3 ice steps (NEMO 3600 s)'}.
# Set through compare_multi_exps(..., exp_labels=...); empty = legends show the experiment ids only.
EXP_LABELS = {}

def exp_label(exp):
    """Legend label of an experiment: 'exp: description' if a description is set."""
    desc = EXP_LABELS.get(exp)
    return f'{exp}: {desc}' if desc else exp

def _exp_legend(ax, **kwargs):
    """Legend for the experiments. With descriptions it goes outside, to the right of the axes,
    in a smaller font, so that it does not cover the data."""
    if EXP_LABELS:
        return ax.legend(fontsize = 9, loc = 'upper left', bbox_to_anchor = (1.01, 1.0), frameon = False)
    return ax.legend(**kwargs)

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

#################################################################################

def _nemo_domain_cfg(exp, user, cart_exp = cart_exp):
    """
    T-grid cell areas and ocean mask from domain_cfg.nc, used when the OASIS areas.nc /
    masks.nc of the run directory are gone (they are links into the inidata, e.g. xa08).
    Same values as the OASIS files: areas e1t*e2t, ocean where top_level > 0, and the
    duplicated half of the north-fold row masked (checked on xa09 and AA02).
    """
    d = xr.open_dataset(cart_exp.format(user) + f'/{exp}/domain_cfg.nc').squeeze()
    area = (d.e1t * d.e2t).values
    ocean = d.top_level.values > 0
    ocean[-1, ocean.shape[1] // 2:] = False
    return area, ocean

def _oasis_file_ok(path):
    return os.path.exists(path)          # False also for a broken link

def get_areas_nemo(exp, user, cart_exp = cart_exp, grid = 'T'):
    #ocean areas
    path = cart_exp.format(user) + f'/{exp}/areas.nc'
    if grid == 'T' and not _oasis_file_ok(path):
        logging.info(f'{path} not available, using cell areas from domain_cfg.nc')
        return _nemo_domain_cfg(exp, user, cart_exp)[0]
    areas = xr.load_dataset(path)
    
    gname = [nam for nam in areas.data_vars if f'-{grid}' in nam]

    if len(gname) > 1:
        raise ValueError(f'Too many grid names matching: {gname}')

    ocean_area = areas[gname[0]].values

    return ocean_area

def get_areas_oifs(exp, user, cart_exp = cart_exp):
    areas = xr.load_dataset(cart_exp.format(user) + f'/{exp}/areas.nc')
    masks = xr.load_dataset(cart_exp.format(user) + f'/{exp}/masks.nc')

    oce_weights = (areas['TL63-ocean.srf']*(~masks['TL63-ocean.msk'].values.astype(bool))).values.squeeze()
    land_weights = (areas['TL63-land.srf']*(~masks['TL63-land.msk'].values.astype(bool))).values.squeeze()

    return oce_weights, land_weights

def get_mask_nemo(exp, user, cart_exp = cart_exp, grid = 'T'):
    #ocean areas
    path = cart_exp.format(user) + f'/{exp}/masks.nc'
    if grid == 'T' and not _oasis_file_ok(path):
        logging.info(f'{path} not available, using the ocean mask from domain_cfg.nc')
        return _nemo_domain_cfg(exp, user, cart_exp)[1]
    masks = xr.load_dataset(path)

    gname = [nam for nam in masks.data_vars if f'-{grid}' in nam]
    if len(gname) > 1:
        raise ValueError(f'Too many grid names matching: {gname}')
    
    ocean_mask = ~masks[gname[0]].values.astype(bool)

    return ocean_mask

def get_ghflux(exp, user, cart_exp = cart_exp):
    # 0.1 W/m2
    try:
        gout = xr.load_dataset(cart_exp.format(user) + f'/{exp}/Goutorbe_ghflux.nc') # mW/m2
        return float(global_mean(gout.drop_vars('time')).gh_flux.values[0])/1000./0.7 # only over ocean
    except Exception as err:
        logging.error("ERROR in get_ghflux:")
        logging.error(err)
        return 0.1


def global_mean(ds, compute = True):
    """
    Global mean of oifs outputs on reduced gaussian grid. Using zonal means and lat weights, a cleaner implementation should use areas as for ocean.
    """
    try:
        all_lats = ds.lat.groupby('lat').mean()
        weights = np.cos(np.deg2rad(all_lats)).compute()
    except ValueError as coso:
        logging.debug('Dask array, trying to use unique instead')
        all_lats = np.unique(ds.lat.values)
        weights = np.cos(np.deg2rad(all_lats))

    if 'time' in ds.coords:
        ds = ds.groupby('time.year').mean()
    
    if 'cell' in ds.dims:
        ds_mean = ds.groupby('lat').mean().weighted(weights).mean('lat')
    else:
        ds_mean = (ds.groupby('lat').mean().weighted(weights).mean('lat')).mean('lon')
    

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

    heat_trend = ds_time_mean['heatc'].diff('year')/year_sec/tot_area
    ds_time_mean['oce_imb'] = heat_trend - ds_time_mean.qt_oce - gh_flux # source of energy in the ocean

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
        logging.debug('Global mean for one vertical level only, lev = ', lev)
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
        logging.debug('Global mean for all vertical levels')
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
    
def global_mean_ice(ds, exp, user, cart_exp = cart_exp, compute = True, grid = 'T'):
    area = get_areas_nemo(exp, user, cart_exp = cart_exp, grid = grid)
    mask = get_mask_nemo(exp, user, cart_exp = cart_exp, grid = grid)

    ds_norm = (ds*area*mask)
    ds_time_mean = ds_norm.copy()

    # min_count=1: a variable that is missing in some years (all NaN) stays NaN instead of 0
    for var in ds.data_vars:
        ds_time_mean[var + '_N'] = ds_norm[var].where(ds.nav_lat > 0.).sum(['x', 'y'], min_count = 1)
        ds_time_mean[var + '_S'] = ds_norm[var].where(ds.nav_lat < 0.).sum(['x', 'y'], min_count = 1)

    # SI3 heat-flux error (hfx_err_dif, per cell area): integrated flux / sea-ice area,
    # i.e. the mean error over the ice-covered area in W m-2
    if 'hfxerr' in ds.data_vars and 'siconc' in ds.data_vars:
        for h in ['N', 'S']:
            ds_time_mean[f'hfxerr_{h}'] = ds_time_mean[f'hfxerr_{h}'] / ds_time_mean[f'siconc_{h}']
        
    # mean thickness over the ice-covered area, in m: ice volume / ice area (sivolu is volume per cell area).
    # From annual means this is the only consistent choice: mean(sithic*siconc) != mean(sithic)*mean(siconc)
    if 'sithic' in ds.data_vars:
        for h in ['N', 'S']:
            if f'sivolu_{h}' in ds_time_mean and f'siconc_{h}' in ds_time_mean:
                ds_time_mean[f'sithic_{h}'] = ds_time_mean[f'sivolu_{h}'] / ds_time_mean[f'siconc_{h}']
            else:
                ds_time_mean[f'sithic_{h}'] = ds_time_mean[f'sithic_{h}'] * np.nan
                logging.warning('sivolu or siconc missing: sea-ice mean thickness not computed')

    ds_time_mean = ds_time_mean[[var for var in ds_time_mean.data_vars if '_N' in var or '_S' in var]]

    if compute:
        return ds_time_mean.compute()
    else:
        return ds_time_mean


def repair_ice_mean(ds):
    """
    Ice means saved before 2026-11-24 hold sithic_N/S = (area-integrated thickness) x (ice area), a
    meaningless ~1e27 number. Mean thickness over the ice-covered area is sivolu/siconc (both area
    integrals), the same as global_mean_ice computes now: recompute it from them. Idempotent.
    """
    if ds is None:
        return ds
    for h in ('N', 'S'):
        if all(f'{v}_{h}' in ds for v in ('sithic', 'sivolu', 'siconc')):
            attrs = ds[f'sithic_{h}'].attrs
            ds[f'sithic_{h}'] = (ds[f'sivolu_{h}'] / ds[f'siconc_{h}']).assign_attrs(attrs)
    return ds

def compute_atm_balance(ds):#, exp, user, cart_exp = cart_exp):
    """
    Computes net toa, net srf, atm imbalance (toa-srf)
    """
    if np.all([va in ds for va in 'rlut rsut rsdt'.split()]):
        ds['toa_net'] = ds.rsdt - ds.rlut - ds.rsut

    if 'rlntcs' in ds and 'rsntcs' in ds:
        ds['crf'] =  ds.toa_net - ds.rlntcs - ds.rsntcs

    if np.all([va in ds for va in 'rlns rsns hfss hfls'.split()]):
        ds['srf_net_nosn'] = ds.rlns + ds.rsns - ds.hfss - ds.hfls
        if 'prsn' in ds:
            ds['srf_net'] = ds['srf_net_nosn'] - 334000.*ds.prsn
        else:
            logging.warning('prsn not available, computing srf_net without snow contribution.. (to avoid this, add prsn to atmvars)')

    if 'toa_net' in ds:
        if 'srf_net' in ds:
            # compute toa-srf
            ds['toa-srf'] = ds['toa_net'] - ds['srf_net']
        elif 'srf_net_nosn' in ds:
            ds['toa-srf'] = ds['toa_net'] - ds['srf_net_nosn']
        else:
            pass

        if 'toa-srf' in ds:
            ds['atm_imb'] = -(ds['toa-srf']) # atm imbalance: positive means energy is created

    if 'pr' in ds and 'evspsbl' in ds:
        # compute pr-evap
        ds['E-P'] = ds['evspsbl'] - ds['pr']

    return ds


def compute_interface_balance(clim_all):
    """
    Computes balance at interface. Requires atm, oce, ice
    """
    fact = 334*1000*1000/(3.1e7*4*3.14*6e6**2) # to convert sea ice formation in W/m2

    if 'qt_oce' in list(clim_all['oce_mean'].items())[0][1] and 'srf_net_oce' in list(clim_all['atm_mean'].items())[0][1] and 'sivolu_N' in list(clim_all['ice_mean'].items())[0][1]:
        for exp in clim_all['oce_clim']:
            clim_all['oce_mean'][exp]['interface_imb'] = clim_all['atm_mean'][exp]['srf_net_oce_nosn'] - (clim_all['ice_mean'][exp]['sivolu_N']+clim_all['ice_mean'][exp]['sivolu_S']).diff('year')*fact - clim_all['oce_mean'][exp]['qt_oce']

    return clim_all

    
def _annual(ds, variables = None):
    """Annual means (dim 'year') of the monthly/daily output, optionally of some variables only."""
    if variables is not None:
        ds = ds[variables]
    return ds.groupby('time_counter.year').mean()

def _rename_xy(ds, grid = 'T'):
    """NEMO horizontal dims (x_grid_T[_inner], y_grid_T[_inner]) -> x, y."""
    for suffix in ('_inner', ''):
        if f'x_grid_{grid}{suffix}' in ds.dims:
            ds = ds.rename({f'x_grid_{grid}{suffix}': 'x', f'y_grid_{grid}{suffix}': 'y'})
    return ds

def _clim_tag(year_clim):
    """Filename tag of the climatology: '' for the last CLIM_YEARS, '_y0-y1' for year_clim."""
    return '' if year_clim is None else f'_{year_clim[0]}-{year_clim[1]}'

def _clim_mean(ds, year_clim):
    """Climatology of annual means: the last CLIM_YEARS years, or year_clim=(y0, y1) inclusive."""
    if year_clim is None:
        logging.debug(f'Using last {CLIM_YEARS} years for climatology')
        return ds.isel(year = slice(-CLIM_YEARS, None)).mean('year')
    logging.debug(f'Using years {year_clim[0]}-{year_clim[1]} for climatology')
    return ds.sel(year = slice(year_clim[0], year_clim[1])).mean('year')

def compute_atm_clim(ds, exp, user, cart_exp = cart_exp, cart_out = cart_out, atmvars = 'rsut rlut rsdt tas pr'.split(), year_clim = None):
    ds = _annual(ds, atmvars).compute()
    logging.debug('Atmospheric climatology computed')

    ds = compute_atm_balance(ds)#, exp, user, cart_exp)

    if 'srf_net' in ds:
        # net srf over ocean and over land
        srf_bal = dict()
        oce_weights, land_weights = get_areas_oifs(exp, user, cart_exp=cart_exp)
        logging.debug(f"Shape of srf_net: {ds['srf_net'].dims}, Shape of oce_weights: {oce_weights.shape}")
        srf_bal['srf_net_oce_nosn'] = np.sum(oce_weights[np.newaxis, ...]*ds['srf_net_nosn'], axis = 1)/np.sum(oce_weights)
        srf_bal['srf_net_oce'] = np.sum(oce_weights[np.newaxis, ...]*ds['srf_net'], axis = 1)/np.sum(oce_weights)
        srf_bal['srf_net_land'] = np.sum(land_weights[np.newaxis, ...]*ds['srf_net'], axis = 1)/np.sum(land_weights)

    atmclim = _clim_mean(ds, year_clim)
    atmmean = global_mean(ds, compute = True)

    if 'srf_net' in ds:
        for ke in srf_bal:
            atmmean[ke] = srf_bal[ke]

        atmmean['atm_imb_bottino'] = -(atmmean['toa_net'] - atmmean['srf_net_land'] - atmmean['srf_net_oce_nosn'])

    if cart_out is not None:
        tag = _clim_tag(year_clim)
        _save_nc(atmclim, cart_out + f'clim_tuning_{exp}{tag}.nc')
        _save_nc(atmmean, cart_out + f'mean_tuning_{exp}.nc')

    return atmclim, atmmean

def compute_oce_clim(ds, exp, user, cart_exp = cart_exp, cart_out = cart_out, ocevars = 'tos heatc qt_oce sos'.split(), year_clim = None, grid = 'T'):
    # annual means computed once: climatology and global means then come from memory (one read of the output)
    ds = _rename_xy(_annual(ds, ocevars), grid).compute()
    oceclim = _clim_mean(ds, year_clim)
    ocemean = global_mean_oce_2d(ds, exp, user, cart_exp, compute = True)

    if cart_out is not None:
        tag = _clim_tag(year_clim)
        _save_nc(oceclim, cart_out + f'clim_oce_tuning_{exp}{tag}.nc')
        _save_nc(ocemean, cart_out + f'mean_oce_tuning_{exp}.nc')

    return oceclim, ocemean

def compute_ice_clim(ds, exp, user, cart_exp = cart_exp, cart_out = cart_out, icevars = 'sithic sivolu siconc'.split(), year_clim = None):
    # variables not in this experiment's output (e.g. hfxerr, only with the SI3 process
    # diagnostics) are saved as NaN, so that they are not looked for again at every update
    absent = [v for v in icevars if v not in ds]
    if absent:
        logging.info(f'{exp}: {absent} not in the ice output, saved as NaN')
        template = ds[[v for v in icevars if v in ds][0]]
        for v in absent:
            ds[v] = xr.full_like(template, np.nan)
    # annual means computed once: climatology and global means then come from memory (one read of the output)
    ds = _annual(ds, icevars).compute()
    iceclim = _clim_mean(ds, year_clim)
    icemean = global_mean_ice(ds, exp, user, cart_exp, compute = True)

    if cart_out is not None:
        tag = _clim_tag(year_clim)
        _save_nc(iceclim, cart_out + f'clim_ice_tuning_{exp}{tag}.nc')
        _save_nc(icemean, cart_out + f'mean_ice_tuning_{exp}.nc')

    return iceclim, icemean

def compute_amoc_clim(ds, exp, cart_out = cart_out, year_clim = None):
    amoc_ts = calc_amoc_ts(ds, plot = False)
    amoc_ts = amoc_ts.groupby('time_counter.year').mean() # year as time coordinate

    if isinstance(amoc_ts, xr.Dataset):
        amoc_ts = amoc_ts['msftyz']

    amoc = _annual(ds['msftyz']).compute()
    amoc_mean = _clim_mean(amoc, year_clim).squeeze().compute()
    amoc_ts = amoc_ts.compute()

    if cart_out is not None:
        tag = _clim_tag(year_clim)
        _save_nc(amoc_mean, cart_out + f'amoc_2d_tuning_{exp}{tag}.nc')
        _save_nc(amoc_ts, cart_out + f'amoc_ts_tuning_{exp}.nc')

    return amoc_mean, amoc_ts

def compute_rho_clim(ds, exp, user, cart_exp = cart_exp, cart_out = cart_out, ocevars = 'density Nsquared'.split(), year_clim = None, grid = 'T'):
    oceclim = _clim_mean(ds, year_clim).compute()
    _save_nc(oceclim, cart_out + f'clim_rho_tuning_{exp}{_clim_tag(year_clim)}.nc')

    ocemean = global_mean_oce_3d(ds, exp, user, 'density Nsquared'.split(), cart_exp, compute = True)
    _save_nc(ocemean, cart_out + f'mean_rho_tuning_{exp}.nc')

    return oceclim, ocemean 

def calc_amoc_ts(data, ax = None, exp_name = 'exp', depth_min = 500., depth_max = 2000., lat_min = 25, lat_max = 28, ylim = (5, 20), plot = False, basin = 2):

    if plot and ax is None:
        fig, ax = plt.subplots()

    if len(data.basin) < 2:
        logging.info('Basin 2 not found! fall back to basin 1 (global MOC)')
        basin = 1

    amoc = data.sel(
        depthw=slice(depth_min, depth_max), 
        basin=basin
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

_FILE_YEARS_RE = re.compile(r'_(\d{4})-(\d{4})\.nc$')

def file_years(fname):
    """(first_year, last_year) from an ECE4 output filename ending in _YYYY-YYYY.nc, else None."""
    m = _FILE_YEARS_RE.search(os.path.basename(fname))
    return (int(m.group(1)), int(m.group(2))) if m else None

def select_files(pattern_or_list, first_year = None, last_year = None):
    """
    Sorted list of files (from a glob pattern or a list) whose years, as read from the
    filename, overlap [first_year, last_year]. Nothing is opened. Files without years
    in their name are always kept.
    """
    if isinstance(pattern_or_list, str):
        files = glob.glob(pattern_or_list)
    else:
        files = list(pattern_or_list)
    files = sorted(os.path.normpath(f) for f in files)

    selected = []
    for f in files:
        yrs = file_years(f)
        if yrs is not None:
            if first_year is not None and yrs[1] < first_year:
                continue
            if last_year is not None and yrs[0] > last_year:
                continue
        selected.append(f)
    return selected

def last_file_year(pattern_or_list):
    """Last year available according to filenames (None if no file / no years in names)."""
    yrs = [file_years(f) for f in select_files(pattern_or_list)]
    yrs = [y[1] for y in yrs if y is not None]
    return max(yrs) if yrs else None

def _is_locked(path):
    """
    True if another process holds an exclusive lock on path, as XIOS does on the file it is
    still writing. Tested with a non-blocking shared flock, the same lock HDF5 asks for when it
    opens a file: this way HDF5 never tries the locked file and does not print its error stack.
    """
    try:
        with open(path, 'rb') as fh:
            try:
                fcntl.flock(fh, fcntl.LOCK_SH | fcntl.LOCK_NB)
            except BlockingIOError:
                return True
            except OSError:
                return False   # filesystem without flock support: let HDF5 decide
            fcntl.flock(fh, fcntl.LOCK_UN)
    except OSError:
        return False
    return False

def _save_nc(ds, path):
    """
    to_netcdf through a temporary file renamed at the end: an interrupted save (kernel
    restart, cell interrupted, quota) never leaves an empty or truncated file in place.
    """
    tmp = f'{path}.tmp{os.getpid()}'
    try:
        ds.to_netcdf(tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)

def _nc_readable(path):
    """True if path exists and opens as netCDF (False for empty or truncated files)."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return False
    try:
        with netCDF4.Dataset(path):
            return True
    except (OSError, RuntimeError):
        return False

def _n_time_records(path):
    """Number of time_counter records in a netCDF file; None if it cannot be read."""
    try:
        with netCDF4.Dataset(path) as nc:
            return len(nc.dimensions['time_counter']) if 'time_counter' in nc.dimensions else None
    except (OSError, RuntimeError):
        return None

def _check_time_axis(ds):
    """
    open_mfdataset preprocess: raise if the decoded time_counter does not lie in the
    years of the filename or is not monotonic. This happens when the run starts before
    the XIOS time origin (ref_year, default 1800): XIOS then writes every year as if it
    were the origin year, with the last month wrapped around. The output is unusable.
    """
    src = ds.encoding.get('source', '')
    yrs = file_years(src)
    if yrs is None or 'time_counter' not in ds.dims:
        return ds
    time = ds['time_counter']
    years = time.dt.year.values
    ok_years = np.all((years >= yrs[0]) & (years <= yrs[1]))
    ok_mono = time.size < 2 or bool(np.all(time.values[1:] > time.values[:-1]))
    if not (ok_years and ok_mono):
        raise ValueError(
            f"Wrong calendar in {src}: filename years {yrs[0]}-{yrs[1]}, but time_counter "
            f"goes from {time.values[0]} to {time.values[-1]}"
            f"{'' if ok_mono else ' and is not monotonic'} (units: {time.encoding.get('units')}). "
            "The run probably starts before the XIOS time origin (ref_year in context_nemo.xml / "
            "context_oifs.xml): set ref_year <= first year and rerun the experiment.")
    return ds

def open_output(files, chunks = {'time_counter': 240}):
    """
    Open a sorted list of ECE4 output files (one or more years each) along time_counter.
    Files are concatenated in the given order (no coordinate sorting), and variables /
    coordinates without time are taken from the first file instead of being compared
    across all files: much faster with many files.
    If the last file is still being written (HDF error -101, or no time records yet), it is
    dropped. For monthly files ('_1m_') only complete years (12 months) are kept, so annual
    means never include a year that is still being written.
    Raises if a file has a wrong calendar (run before the XIOS time origin): rerun needed.
    """
    # normalised paths: xarray reports normalised filenames in its errors (e.g. no '//')
    files = [os.path.normpath(f) for f in files]
    kwargs = dict(decode_times = time_coder, chunks = chunks,
                  combine = 'nested', concat_dim = 'time_counter',
                  data_vars = 'minimal', coords = 'minimal', compat = 'override',
                  parallel = True, preprocess = _check_time_axis)
    if not files:
        raise FileNotFoundError('open_output: empty file list, no output files found')

    # the file of the year XIOS is still writing is locked: drop it before HDF5 tries to open it
    while len(files) > 1 and _is_locked(files[-1]):
        logging.info(f'{files[-1]} is locked (still being written by the run?), dropping it')
        files = files[:-1]
    if len(files) == 1 and _is_locked(files[0]):
        raise OSError(11, f'Cannot read {files[0]}: it is locked (still being written) and it is the only '
                          'output file found. Wait for the next year to be written, or leave this experiment out for now.')

    # the last file of a running experiment can exist with 0 time records: XIOS writes the
    # header first and the data only at sync_freq. It is readable (no HDF error) but its
    # empty time axis is not decoded and would break the time-axis check: drop it.
    if len(files) > 1 and _n_time_records(files[-1]) == 0:
        logging.info(f'{files[-1]} has no time records yet (run still ongoing?), dropping it')
        files = files[:-1]

    def _unreadable(err, files_tried, after_drop = False):
        bad = os.path.normpath(getattr(err, 'filename', None) or files_tried[-1])
        reason = 'check the run output.'
        if after_drop:
            reason = ('it is still unreadable after dropping the last file (still being written): '
                      'check the run output.')
        elif len(files_tried) == 1:
            reason = ('it is the only output file found: the run has probably not completed '
                      'its first output period yet (file still being written). '
                      'Wait for the next year to be written, or leave this experiment out for now.')
        elif bad != files_tried[-1]:
            reason = ('it is not the last file, so it is probably corrupted '
                      '(e.g. a crashed or restarted leg): check or remove it.')
        return OSError(err.errno, f'Cannot read {bad} (NetCDF/HDF error {err.errno}): {reason}')

    try:
        ds = xr.open_mfdataset(files, **kwargs)
    except OSError as err:
        if err.errno != -101:
            raise
        bad = os.path.normpath(getattr(err, 'filename', None) or files[-1])
        if len(files) == 1 or bad != files[-1]:
            raise _unreadable(err, files) from err
        logging.info(f'Cannot read {files[-1]} (run still ongoing?), dropping it')
        try:
            ds = xr.open_mfdataset(files[:-1], **kwargs)
        except OSError as err2:
            if err2.errno != -101:
                raise
            raise _unreadable(err2, files[:-1], after_drop = True) from err2
    if all('_1m_' in os.path.basename(f) for f in files):
        ds = _complete_years(ds, files)
    return ds

def _complete_years(ds, files, nrec = 12):
    """
    Keep only the years with nrec monthly records in a monthly dataset. A running
    experiment writes its current year month by month, so the last file usually holds an
    incomplete year: its annual mean would be biased, and once saved it would never be
    recomputed by the incremental update.
    """
    years = ds['time_counter'].dt.year.values
    yrs, counts = np.unique(years, return_counts = True)
    incomplete = yrs[counts < nrec]
    if incomplete.size == 0:
        return ds
    if incomplete.size == yrs.size:
        raise ValueError(f'No complete year yet in {os.path.dirname(files[0])} '
                         f'({", ".join(f"{y}: {c} months" for y, c in zip(yrs, counts))}): '
                         'wait for the first year to be completed, or leave this experiment out for now.')
    for y, c in zip(yrs, counts):
        if c < nrec:
            logging.info(f'{os.path.basename(files[-1]).split("_")[0]}: year {y} has only {c} of {nrec} '
                         'months (still being written?), excluded')
    return ds.isel(time_counter = ~np.isin(years, incomplete))

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

def read_output(exps, user=None, read_again=[], cart_exp=cart_exp, cart_out=cart_out,
                atmvars='rsut rlut rsdt tas pr'.split(),
                ocevars='tos heatc qt_oce sos'.split(),
                icevars='siconc sivolu sithic snvolu hfxerr'.split(),
                atm_only=False, year_clim=None, file_lists=None, ongoing = [], density=False, do_all_from_scratch = False):
    """
    Reads outputs and computes global means.

    exps: list of experiment names to read
    user: list of users
    read_again: list of exps to read again (if run has proceeded)
    atm_only: compute only atm diags
    year_clim: set years for computing climatologies (if None, considers the last CLIM_YEARS = 20 years)
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
    def _smmregrid_area_weights(filelist, exp, domain, fallback_file=None,
                                target_grid='r180x90', method="ycon"):
        """
        Regridding weights and areas for exp, in order of preference:
          1. cached files in cart_out (computed earlier from the original output)
          2. original model output (result is cached for later use)
          3. fallback_file (e.g. the saved climatology, which keeps lat/lon but no
             cell corners): if `method` needs corners (e.g. ycon), retry with 'dis'
        """
        logging.info(f'Computing regridding weights and areas for {exp} ({domain}) to {target_grid}')
        w_cache = cart_out + f'weights_{domain}_{exp}_{target_grid}_{method}.nc'
        a_cache = cart_out + f'areas_{domain}_{exp}.nc'

        if os.path.exists(w_cache):
            logging.info(f'  → reading cached weights: {w_cache}')
            weights = xr.load_dataset(w_cache)
            areas = xr.load_dataset(a_cache) if os.path.exists(a_cache) else None
            return weights, areas

        filz = filelist[exp]
        rawfiles = sorted(glob.glob(filz)) if isinstance(filz, str) else list(filz)
        if rawfiles:
            basefile, from_raw = rawfiles[0], True
        elif fallback_file is not None and os.path.exists(fallback_file):
            basefile, from_raw = fallback_file, False
            logging.info(f'  → original output not found, using {fallback_file} as source grid')
        else:
            logging.error(f'No source grid available for {exp} ({domain}): no weights')
            return None, None
        logging.debug(f'  → base file: {basefile}, method: {method}')

        weights = None
        areas = None
        try:
            generator = CdoGenerate(source_grid=basefile, target_grid=target_grid)
        except Exception as e:
            logging.error(f'Cannot set up regridding for {exp} ({domain}): {e}')
            return None, None
        try:
            weights = generator.weights(method=method).load()
        except Exception as e:
            if from_raw:
                logging.error(f'Error while computing weights for {exp}: {e}')
            else:
                logging.info(f'  → {method} weights failed on {basefile} (no cell corners?), retrying with dis')
                try:
                    weights = generator.weights(method='dis').load()
                except Exception as e2:
                    logging.error(f'Error while computing weights for {exp}: {e2}')
        try:
            areas = generator.areas().load()
        except Exception as e:
            logging.debug(f'Could not compute areas for {exp} ({domain}): {e}')

        # cache only what comes from the original output, so it survives its deletion
        if from_raw:
            if weights is not None:
                _save_nc(weights, w_cache)
            if areas is not None:
                _save_nc(areas, a_cache)

        return weights, areas

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
        broken = [f for f in (clim_path, mean_path) if not _nc_readable(f)]
        if broken:
            logging.warning(f'{exp}: cannot read {broken} (empty or truncated, e.g. an interrupted save): '
                            'recomputing from scratch')
            return 'from_scratch', required_vars
        if exp in read_again:
            if required_vars is not None:
                try:
                    with xr.open_dataset(clim_path) as ds_saved:
                        missing = [v for v in required_vars if v not in ds_saved]
                except Exception:
                    missing = required_vars
                if missing:
                    logging.info(f'  → missing vars in {clim_path}: {missing}')
                    return 'missing_vars', missing
            return 'update', None
        return False, None

    def _open_mfdataset_safe(filz, *args, **kwargs):
        # kept for backward compatibility: file selection + ongoing-run handling now in open_output
        return open_output(select_files(filz))

    def _normalise_amoc_ts(ds):
        if 'time_counter' in ds.dims:
            ds = ds.groupby('time_counter.year').mean()
        if isinstance(ds, xr.Dataset):
            ds = ds['msftyz']
        return ds

    # ── per-domain compute (full dataset) ─────────────────────────────────────

    def _compute_atm(exp, us, coupled, vars=atmvars, save=True):
        ds = _open_mfdataset_safe(filz_atm[exp])
        return compute_atm_clim(ds, exp, us, cart_exp=cart_exp,
                                cart_out=cart_out if save else None,
                                atmvars=vars, year_clim=year_clim)

    def _compute_oce(exp, us, vars=ocevars, save=True):
        ds = _open_mfdataset_safe(filz_nemo[exp])
        return compute_oce_clim(ds, exp, us, cart_exp=cart_exp,
                                cart_out=cart_out if save else None,
                                ocevars=vars, year_clim=year_clim)

    def _compute_ice(exp, us, vars=icevars, save=True):
        ds = _open_mfdataset_safe(filz_ice[exp])
        return compute_ice_clim(ds, exp, us, cart_exp=cart_exp,
                                cart_out=cart_out if save else None,
                                icevars=vars, year_clim=year_clim)

    def _compute_amoc(exp, save=True):
        ds = _open_mfdataset_safe(filz_amoc[exp])
        return compute_amoc_clim(ds, exp,
                                cart_out=cart_out if save else None,
                                year_clim=year_clim)

    # ── per-domain update (read only new years + climatology window) ──────────

    def _update_files(domain, filz, last_saved):
        """
        Decide from the filenames alone which files an update needs.
        Returns (files, first_year, clim_changes), or (None, None, False) if no new data.
        Files cover the new years and, if the climatology changes, its window:
        the last CLIM_YEARS years (+1 spare in case the last file is still being written)
        or year_clim.
        """
        last_avail = last_file_year(filz)
        if last_avail is None or last_avail <= last_saved:
            logging.info(f'[{domain}] No new files after {last_saved}, using existing diagnostics')
            return None, None, False

        if year_clim is None:
            clim_first, clim_changes = last_avail - CLIM_YEARS, True
        elif year_clim[1] > last_saved:
            clim_first, clim_changes = year_clim[0], True
        else:  # fixed climatology window already fully in saved data
            clim_first, clim_changes = None, False

        first = last_saved + 1 if clim_first is None else min(last_saved + 1, clim_first)
        files = select_files(filz, first_year=first)
        logging.info(f'[{domain}] Saved data up to {last_saved}, files up to {last_avail}: '
                     f'reading {len(files)} files from {first}')
        return files, first, clim_changes

    def _update_domain(exp, us, domain, clim_old, mean_old, filz,
                       compute_fn, clim_path, mean_path):
        last_saved = int(mean_old.year[-1].values)
        files, first, clim_changes = _update_files(domain, filz, last_saved)
        if files is None:
            return clim_old, mean_old

        ds = open_output(files)
        ds = ds.sel(time_counter=slice(f'{first:04d}-01-01', None))  # in case of multi-year files
        clim_new, mean_new = compute_fn(ds)
        mean_new = mean_new.sel(year=slice(last_saved + 1, None))
        if mean_new.sizes['year'] == 0:
            logging.info(f'[{domain}] No new complete years, using existing diagnostics')
            return clim_old, mean_old
        logging.info(f'[{domain}] Adding years {int(mean_new.year[0])}-{int(mean_new.year[-1])}')

        clim_updated = clim_new if clim_changes else clim_old
        mean_updated = xr.concat([mean_old, mean_new], dim='year')
        _save_nc(clim_updated, clim_path)
        _save_nc(mean_updated, mean_path)
        return clim_updated, mean_updated

    def _update_amoc(exp, amoc_ts_old, amoc_mean_old, amoc_2d_path, amoc_ts_path):
        last_saved = int(amoc_ts_old.year[-1].values)
        files, first, clim_changes = _update_files('amoc', filz_amoc[exp], last_saved)
        if files is None:
            return amoc_mean_old, amoc_ts_old

        ds = open_output(files)
        ds = ds.sel(time_counter=slice(f'{first:04d}-01-01', None))
        amoc_mean_new, amoc_ts_new = compute_amoc_clim(ds, exp, cart_out=None,
                                                       year_clim=year_clim)
        amoc_ts_new = amoc_ts_new.sel(year=slice(last_saved + 1, None))
        if amoc_ts_new.sizes['year'] == 0:
            logging.info('[amoc] No new complete years, using existing diagnostics')
            return amoc_mean_old, amoc_ts_old
        logging.info(f'[amoc] Adding years {int(amoc_ts_new.year[0])}-{int(amoc_ts_new.year[-1])}')

        amoc_mean_updated = amoc_mean_new if clim_changes else amoc_mean_old
        amoc_ts_updated = xr.concat([amoc_ts_old, amoc_ts_new], dim='year')
        _save_nc(amoc_mean_updated, amoc_2d_path)
        _save_nc(amoc_ts_updated, amoc_ts_path)
        return amoc_mean_updated, amoc_ts_updated

    # ── merge missing vars into existing saved files ───────────────────────────

    def _merge_missing_vars(domain, clim_path, mean_path, missing, compute_fn_missing):
        """
        Compute only the missing variables on the full dataset and merge
        into the existing clim/mean files.
        compute_fn_missing: callable(missing_vars) -> (clim_new, mean_new)
        """
        logging.info(f'[{domain}] Computing missing vars on full dataset: {missing}')
        clim_old = xr.load_dataset(clim_path)
        mean_old = xr.load_dataset(mean_path)

        try:
            clim_new, mean_new = compute_fn_missing(missing)
        except FileNotFoundError:
            logging.warning(f'[{domain}] {missing} cannot be computed: the original output is gone. '
                            'Using the saved diagnostics without them')
            return clim_old, mean_old

        # same years as the saved series: a newer complete year must come in through the
        # regular update, otherwise the update would think it is already saved
        if 'year' in mean_new.dims and 'year' in mean_old.dims:
            mean_new = mean_new.reindex(year = mean_old.year)
        # ice time series are per hemisphere (var_N, var_S)
        clim_keep = [v for v in clim_new.data_vars if v in missing]
        mean_keep = [v for v in mean_new.data_vars if v in missing or v[:-2] in missing]
        clim_merged = xr.merge([clim_old, clim_new[clim_keep]])
        mean_merged = xr.merge([mean_old, mean_new[mean_keep]])

        _save_nc(clim_merged, clim_path)
        _save_nc(mean_merged, mean_path)
        return clim_merged, mean_merged

    # ── weights and areas containers ─────────────────────────────────────────────────────

    atmweights_exp = dict()
    atmareas_exp = dict()
    oceweights_exp = dict()
    oceareas_exp = dict()
    iceweights_exp = dict()
    iceareas_exp = dict()

    # ── output containers for climatologies ─────────────────────────────────────
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
        logging.info(f'\n{"="*60}\n{exp}')

        # ── detect coupled / ocean-only ───────────────────────────────────────
        coupled = False
        if not atm_only:
            coupled = (os.path.exists(cart_out + f'clim_oce_tuning_{exp}.nc') or
                       os.path.exists(cart_out + f'oce_tuning_{exp}.nc') or
                       _files_exist(filz_nemo[exp]))
            logging.info('coupled' if coupled else f'No ocean files found for {exp}. Assuming atm-only')

        ocean_only = (not _files_exist(filz_atm[exp]) and
                      not os.path.exists(cart_out + f'clim_tuning_{exp}.nc'))
        if ocean_only:
            logging.info(f'No atm files found for {exp}. Assuming ocean-only')

        coupled_exps.append(True if coupled else False)

        # ── file paths ────────────────────────────────────────────────────────

        yclim_tag = _clim_tag(year_clim)

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
        if not do_all_from_scratch:
            atm_action,  atm_missing  = _needs_compute(atm_clim_path, atm_mean_path, atmvars) if not ocean_only else (False, None)
            oce_action,  oce_missing  = _needs_compute(oce_clim_path, oce_mean_path, ocevars) if coupled        else (False, None)
            ice_action,  ice_missing  = _needs_compute(ice_clim_path, ice_mean_path, icevars) if coupled        else (False, None)
            amoc_action, _            = _needs_compute(amoc_2d_path,  amoc_ts_path)           if coupled        else (False, None)
            if density:
                rho_action, _            = _needs_compute(rho_clim_path, rho_mean_path)       if coupled        else (False, None)
        else:
            atm_action, atm_missing = ('from_scratch', atmvars) if not ocean_only else (False, None)
            oce_action, oce_missing = ('from_scratch', ocevars) if coupled        else (False, None)
            ice_action, ice_missing = ('from_scratch', icevars) if coupled        else (False, None)
            amoc_action = 'from_scratch' if coupled else False
            if density: rho_action = 'from_scratch' if coupled else False
            
        # ── ATM ───────────────────────────────────────────────────────────────
        if not ocean_only:
            if not atm_action:
                logging.info('[atm] Already computed, reading...')
                atmclim_exp[exp] = xr.load_dataset(atm_clim_path)
                atmmean_exp[exp] = xr.load_dataset(atm_mean_path)

            elif atm_action == 'missing_vars':
                atmclim_exp[exp], atmmean_exp[exp] = _merge_missing_vars(
                    'atm', atm_clim_path, atm_mean_path,
                    missing=atm_missing,
                    compute_fn_missing=lambda vars: compute_atm_clim(
                        _open_mfdataset_safe(filz_atm[exp]),
                        exp, us, cart_exp=cart_exp, cart_out=None, atmvars=vars, year_clim=year_clim))

            elif atm_action == 'update':
                logging.info('[atm] Updating with new data...')
                clim_old = xr.load_dataset(atm_clim_path)
                mean_old = xr.load_dataset(atm_mean_path)
                atmclim_exp[exp], atmmean_exp[exp] = _update_domain(
                    exp, us, 'atm', clim_old, mean_old,
                    filz=filz_atm[exp],
                    compute_fn=lambda ds: compute_atm_clim(
                        ds, exp, us, cart_exp=cart_exp, cart_out=None, atmvars=atmvars, year_clim=year_clim),
                    clim_path=atm_clim_path, mean_path=atm_mean_path)

            else:  # 'from_scratch'
                logging.info('[atm] Computing from scratch...')
                atmclim_exp[exp], atmmean_exp[exp] = _compute_atm(exp, us, coupled)

            # add weights and areas for smm regridding (from original file, all grids the same)
            atmweights_exp[exp], atmareas_exp[exp] = _smmregrid_area_weights(filz_atm, exp, 'atm', fallback_file=atm_clim_path)

        # ── OCE ───────────────────────────────────────────────────────────────
        if coupled:
            if not oce_action:
                logging.info('[oce] Already computed, reading...')
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
                        _open_mfdataset_safe(filz_nemo[exp]),
                        exp, us, cart_exp=cart_exp, cart_out=None,
                        ocevars=vars, year_clim=year_clim))

            elif oce_action == 'update':
                logging.info('[oce] Updating with new data...')
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
                logging.info('[oce] Computing from scratch...')
                oceclim_exp[exp], ocemean_exp[exp] = _compute_oce(exp, us)

            # add weights and areas for smm regridding (from original file, all grids the same)
            oceweights_exp[exp], oceareas_exp[exp] = _smmregrid_area_weights(filz_nemo, exp, 'oce', fallback_file=oce_clim_path, method="bil")

            # ── ICE ───────────────────────────────────────────────────────────
            if not ice_action:
                logging.info('[ice] Already computed, reading...')
                iceclim_exp[exp] = xr.load_dataset(ice_clim_path)
                icemean_exp[exp] = xr.load_dataset(ice_mean_path)

            elif ice_action == 'missing_vars':
                iceclim_exp[exp], icemean_exp[exp] = _merge_missing_vars(
                    'ice', ice_clim_path, ice_mean_path,
                    missing=ice_missing,
                    compute_fn_missing=lambda vars: compute_ice_clim(
                        _open_mfdataset_safe(filz_ice[exp]),
                        exp, us, cart_exp=cart_exp, cart_out=None,
                        # siconc is needed to express hfxerr per unit ice area
                        icevars=list(dict.fromkeys(vars + ['siconc'])), year_clim=year_clim))

            elif ice_action == 'update':
                logging.info('[ice] Updating with new data...')
                clim_old = xr.load_dataset(ice_clim_path)
                mean_old = repair_ice_mean(xr.load_dataset(ice_mean_path))
                iceclim_exp[exp], icemean_exp[exp] = _update_domain(
                    exp, us, 'ice', clim_old, mean_old,
                    filz=filz_ice[exp],
                    compute_fn=lambda ds: compute_ice_clim(
                        ds, exp, us, cart_exp=cart_exp, cart_out=None,
                        icevars=icevars, year_clim=year_clim),
                    clim_path=ice_clim_path, mean_path=ice_mean_path)

            else:  # 'from_scratch'
                logging.info('[ice] Computing from scratch...')
                iceclim_exp[exp], icemean_exp[exp] = _compute_ice(exp, us)

            # add weights and areas for smm regridding (from original file, all grids the same)
            iceweights_exp[exp], iceareas_exp[exp] = _smmregrid_area_weights(filz_ice, exp, 'ice', fallback_file=ice_clim_path, method="bil")

            ## density
            if density:
                if not rho_action:
                    logging.info('[rho] Already computed, reading...')
                    rhoclim_exp[exp] = xr.load_dataset(rho_clim_path)
                    rhomean_exp[exp] = xr.load_dataset(rho_mean_path)
                elif rho_action == 'from_scratch':
                    ds = xr.open_mfdataset(filz_rho[exp], decode_times=time_coder, chunks = {'time': 20})
                    rhoclim_exp[exp], rhomean_exp[exp] = compute_rho_clim(ds, exp, us, cart_exp = cart_exp, cart_out = cart_out, year_clim = year_clim)
                    ds.close()
                else:
                    raise ValueError('rho update not implemented')

            # ── AMOC ──────────────────────────────────────────────────────────
            if not amoc_action:
                if os.path.exists(amoc_ts_path):
                    logging.info('[amoc] Already computed, reading...')
                    amoc_ts_exp[exp]   = _normalise_amoc_ts(xr.load_dataset(amoc_ts_path))
                    amoc_mean_exp[exp] = xr.load_dataset(amoc_2d_path)

            elif amoc_action == 'update':
                logging.info('[amoc] Updating with new data...')
                amoc_ts_old   = _normalise_amoc_ts(xr.load_dataset(amoc_ts_path))
                amoc_mean_old = xr.load_dataset(amoc_2d_path)
                amoc_mean_exp[exp], amoc_ts_exp[exp] = _update_amoc(
                    exp, amoc_ts_old, amoc_mean_old, amoc_2d_path, amoc_ts_path)

            else:  # 'from_scratch'
                logging.info('[amoc] Computing from scratch...')
                try:
                    amoc_mean_exp[exp], amoc_ts_exp[exp] = _compute_amoc(exp, save=True)
                except Exception as e:
                    logging.info(e)
                    logging.info('Cannot compute AMOC! pass')

    # ── assemble output ───────────────────────────────────────────────────────
    clim_all = dict()
    clim_all['atm_clim']  = atmclim_exp
    clim_all['atm_mean']  = atmmean_exp
    if np.any(coupled_exps):
        clim_all['oce_clim']  = oceclim_exp
        clim_all['oce_mean']  = ocemean_exp
        clim_all['ice_clim']  = iceclim_exp
        clim_all['ice_mean']  = {exp: repair_ice_mean(ds) for exp, ds in icemean_exp.items()}
        clim_all['amoc_mean'] = amoc_mean_exp
        clim_all['amoc_ts'] = amoc_ts_exp
        clim_all['atm_weights'] = atmweights_exp
        clim_all['atm_areas'] = atmareas_exp
        clim_all['oce_weights'] = oceweights_exp
        clim_all['oce_areas'] = oceareas_exp
        clim_all['ice_weights'] = iceweights_exp
        clim_all['ice_areas'] = iceareas_exp
        if density:
            clim_all['rho_mean'] = rhomean_exp
            clim_all['rho_clim'] = rhoclim_exp

        clim_all = compute_interface_balance(clim_all)

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

    x_ds = xr.concat(okdict.values(), dim=pd.Index(okdict.keys(), name='exp'), join='outer')
    return x_ds


####################################### PLOTS #######################################

def plot_amoc_2d(amoc_mean, exp = None, ax = None, basin = 2):
    if ax is None:
        fig, ax = plt.subplots(figsize = (12,8))

    if isinstance(amoc_mean, xr.Dataset):
        amoc_mean = amoc_mean['msftyz']

    try:
        amoc_mean.sel(basin = basin).plot.contourf(x = 'nav_lat', y = 'depthw', ylim = (3000, 0), xlim = (-30, 70), levels = np.arange(-16, 16.1, 2), ax = ax)
    except:
        amoc_mean.plot.contourf(x = 'nav_lat', y = 'depthw', ylim = (3000, 0), xlim = (-30, 70), levels = np.arange(-16, 16.1, 2), ax = ax)
        
    ax.set_title(exp)

    return ax


def plot_greg(atmmean_exp, exps, cart_out = cart_out, exp_type = 'PI', n_end = CLIM_YEARS, 
              imbalance = 0., ylim = None, colors = None, year_clim = None, rolling = None, s_dot = 1000, 
              labels = True, lw = 0.8, alpha_min = 0.2, alpha = 0.8, s_start = 80, mark_last = True, s_last = 20):
    """
    gregory plot
    The line alpha increases with time (faint = start, opaque = end).
    A square marks the first year, a small black-edged dot marks the last year,
    the big dot is the climatological mean (last n_end years or year_clim).
    """

    fig, ax = plt.subplots(figsize=(12, 8))
    shift = 0.05

    ax.axhline(0., color = 'grey', lw = 0.5)

    if colors is None:
        colors = get_colors(exps)

    for exp, col in zip(exps, colors):
        tas = atmmean_exp[exp].tas
        toa = atmmean_exp[exp].toa_net
        if rolling:
            tas = tas.rolling(year = rolling, min_periods = 1).mean()
            toa = toa.rolling(year = rolling, min_periods = 1).mean()

        xs, ys = np.asarray(tas.values, dtype=float), np.asarray(toa.values, dtype=float)

        _plot_trajectory(ax, xs, ys, col, exp_label(exp), lw = lw, alpha_min = alpha_min, s_start = s_start, mark_last = mark_last, s_last = s_last)

        if year_clim is None:
            x, y = atmmean_exp[exp].tas.isel(year = slice(-n_end, None)).mean(), atmmean_exp[exp].toa_net.isel(year = slice(-n_end, None)).mean()
        else:
            x, y = atmmean_exp[exp].tas.sel(year = slice(year_clim[0], year_clim[1])).mean(), atmmean_exp[exp].toa_net.sel(year = slice(year_clim[0], year_clim[1])).mean()
        ax.scatter(x, y, s = s_dot, color = col, marker = 'o', edgecolors=col, alpha = alpha, zorder = 3)
        if labels:
            ax.text(x+shift, y+shift, exp, fontsize=12, ha='right', color = 'black')

    ax.autoscale_view()  # needed because add_collection doesn't rescale the axes

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
    if EXP_LABELS:
        _exp_legend(ax)

    if ylim is not None:
        ax.set_ylim(ylim)

    name = '-'.join(exps)
    fig.savefig(cart_out + f'check_tuning_{name}.pdf')
    plt.show()

    return fig



def plot_amoc_vs_gtas(clim_all, exps = None, cart_out = cart_out, exp_type = 'PI', n_end = CLIM_YEARS, colors = None, 
                      labels = None, colors_legend = None, lw = 0.8, alpha = 0.8, alpha_min = 0.3, background_color = None, 
                      year_clim = None, rolling = None, s_dot = 1000, s_start = 80, mark_last = True, s_last = 20):
    """
    AMOC vs GTAS.
    The line alpha increases with time (faint = start, opaque = end).
    A square marks the first year, a small black-edged dot marks the last year,
    the big dot is the climatological mean (last n_end years or year_clim).
    `alpha` is the transparency of the big dot.
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    if exps is None:
        exps = clim_all['atm_mean'].keys()

    if colors is None:
        colors = get_colors(exps)

    for exp, col in zip(exps, colors):
        if exp not in clim_all['amoc_ts']: 
            logging.info(f'AMOC not computed for {exp}')
            continue

        if isinstance(clim_all['amoc_ts'][exp], xr.DataArray):
            y = clim_all['amoc_ts'][exp]
        else:
            y = clim_all['amoc_ts'][exp]['msftyz']

        x = clim_all['atm_mean'][exp]['tas']
        if 'time_counter' in y.dims:
            y = y.groupby('time_counter.year').mean()
            
        # drop singleton dims (basin, x) but keep 'year', which may have length 1 for a short run
        y = y.squeeze([d for d in y.dims if d != 'year' and y.sizes[d] == 1])
        logging.debug(f'Exp: {exp}: After processing, AMOC shape: {y.shape}, GTAS shape: {x.shape}')
        if len(y.year) > len(x.year):
            logging.info('cutting excess data in amoc')
            y = y.sel(year = slice(x.year.min(), x.year.max()))

        if rolling:
            x_plot = x.rolling(year = rolling, min_periods = 1).mean()
            y_plot = y.rolling(year = rolling, min_periods = 1).mean()
        else:
            x_plot, y_plot = x, y

        xs = np.asarray(x_plot.values, dtype=float)
        ys = np.asarray(y_plot.values, dtype=float)
        n = min(len(xs), len(ys))  # safety in case lengths still differ
        xs, ys = xs[:n], ys[:n]

        _plot_trajectory(ax, xs, ys, col, exp_label(exp), lw = lw, alpha_min = alpha_min, s_start = s_start, mark_last = mark_last, s_last = s_last)

        if year_clim is None:
            xm, ym = x.isel(year = slice(-n_end, None)).mean(), y.isel(year = slice(-n_end, None)).mean()
        else:
            xm, ym = x.sel(year = slice(year_clim[0], year_clim[1])).mean(), y.sel(year = slice(year_clim[0], year_clim[1])).mean()

        ax.scatter(xm, ym, s = s_dot, color = col, marker = 'o', edgecolors = col, alpha = alpha, zorder = 3)
        ax.text(xm+0.1, ym+0.1, exp, fontsize=12, ha='right', color = 'black')

    ax.autoscale_view()  # needed because add_collection doesn't rescale the axes

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
        _exp_legend(ax)
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


def plot_imbalance(clim_all, cart_out = cart_out, rolling = None):
    atmmean = create_ds_exp(clim_all['atm_mean'])
    ocemean = create_ds_exp(clim_all['oce_mean'])

    figs = []
    fig = plot_custom_greg(atmmean['atm_imb'], ocemean['oce_imb'], (-0.1, 0.1), (-0.1, 0.1), xlabel = 'atm imbalance (>0 -> source)', ylabel = 'oce imbalance (>0 -> source)', cart_out = cart_out, symmetric_axes=True, rolling = rolling)
    figs.append(fig)

    fig = plot_custom_greg(atmmean['atm_imb_bottino'], ocemean['oce_imb'], (-0.1, 0.1), (-0.1, 0.1), xlabel = 'atm imbalance (>0 -> source, no prsn over ocean)', ylabel = 'oce imbalance (>0 -> source)', cart_out = cart_out, symmetric_axes=True, rolling = rolling)
    figs.append(fig)

    fig = plot_custom_greg(atmmean['tas'], atmmean['atm_imb'], None, (-0.1, 0.1), xlabel = 'GTAS', ylabel = 'atm imbalance (>0 -> source)', cart_out = cart_out, symmetric_axes=False, rolling = rolling)
    figs.append(fig)

    fig = plot_custom_greg(atmmean['tas'], atmmean['atm_imb_bottino'], None, (-0.1, 0.1), xlabel = 'GTAS', ylabel = 'atm imbalance (>0 -> source)', cart_out = cart_out, symmetric_axes=False, rolling = rolling)
    figs.append(fig)    

    fig = plot_custom_greg(ocemean['tos'], ocemean['oce_imb'], None, (-0.1, 0.1), xlabel = 'GTOS', ylabel = 'oce imbalance (>0 -> source)', cart_out = cart_out, symmetric_axes=False, rolling = rolling)
    figs.append(fig)

    fig = plot_custom_greg(atmmean['tas'], 2.5e6*atmmean['E-P'], None, (-0.1, 0.1), xlabel = 'GTAS', ylabel = 'E-P energy flux [W/m2] (>0 -> energy needed for sustaining the E-P imbalance)', cart_out = cart_out, symmetric_axes=False, rolling = rolling)
    figs.append(fig)

    fig = plot_custom_greg(atmmean['tas'], atmmean['E-P'], None, None, xlabel = 'GTAS', ylabel = 'E-P (>0 -> source)', cart_out = cart_out, symmetric_axes=False, rolling = rolling)
    figs.append(fig)
    
    if 'interface_imb' in ocemean:
        fig = plot_custom_greg(atmmean['tas'], ocemean['interface_imb'], None, (-0.1, 0.1), xlabel = 'GTAS', ylabel = 'interface imbalance (>0 -> source)', cart_out = cart_out, symmetric_axes=False, rolling = rolling)
        figs.append(fig)

    return figs


def _plot_trajectory(ax, xs, ys, col, label, lw = 0.8, alpha_min = 0.2, s_start = 80, mark_last = True, s_last = 20):
    """
    Draw one experiment's trajectory in Gregory style: line alpha increasing in time
    (faint = start, opaque = end), a square on the first point and a small black-edged
    dot on the last one.
    """
    points = np.column_stack([xs, ys]).reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis = 1)
    rgba = np.tile(to_rgba(col), (len(segments), 1))
    rgba[:, 3] = np.linspace(alpha_min, 1., len(segments))
    ax.add_collection(LineCollection(segments, colors = rgba, linewidths = lw))
    ax.plot([], [], color = col, label = label)  # proxy for legend
    ax.scatter(xs[0], ys[0], s = s_start, marker = 's', color = col, edgecolor = 'k', lw = 1, zorder = 4, alpha = alpha_min)
    if mark_last:
        ax.scatter(xs[-1], ys[-1], s = s_last, marker = 'o', color = col, edgecolor = 'k', lw = 0.5, zorder = 5)


def plot_custom_greg(x_ds, y_ds, x_target, y_target, color_var = None, exps = None, cart_out = cart_out, n_end = CLIM_YEARS, colors = None, labels = None, colors_legend = None, lw = 0.8, alpha = 0.8, background_color = None, cmap_name = 'viridis', xlabel = '', ylabel = '', cbar_label = '', symmetric_axes = False, x_ext = None, y_ext = None, rolling = None,
                     year_clim = None, alpha_min = 0.2, s_dot = 1000, s_start = 80, mark_last = True, s_last = 20, exp_labels = True):
    """
    Gregory-style plot of y_ds vs x_ds for all exps, with the same look as plot_greg.
    The line alpha increases with time (faint = start, opaque = end).
    A square marks the first year, a small black-edged dot marks the last year,
    the big dot is the climatological mean (last n_end years or year_clim).

    x_ds, y_ds: two datasets or dicts containing all exps
    x_target, y_target: two tuples (min, max), drawn as grey bands
    x_ext, y_ext: optional axis limits (min, max); autoscaled to the data if None
    """

    if isinstance(x_ds, dict):
        x_ds = xr.concat(x_ds.values(), dim=pd.Index(x_ds.keys(), name='exp'))
    if isinstance(y_ds, dict):
        y_ds = xr.concat(y_ds.values(), dim=pd.Index(y_ds.keys(), name='exp'))

    fig, ax = plt.subplots(figsize=(12, 8))

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

        # the multi-exp concat pads shorter runs with NaN: keep only this exp's years
        valid = (np.isfinite(x) & np.isfinite(y)).values
        if not valid.any():
            logging.info(f'{exp}: no valid data for {x_ds.name} vs {y_ds.name}')
            continue
        x, y = x.isel(year = valid), y.isel(year = valid)

        if rolling:
            x_plot = x.rolling(year = rolling, min_periods = 1).mean()
            y_plot = y.rolling(year = rolling, min_periods = 1).mean()
        else:
            x_plot, y_plot = x, y
        xs = np.asarray(x_plot.values, dtype=float)
        ys = np.asarray(y_plot.values, dtype=float)

        _plot_trajectory(ax, xs, ys, col, exp_label(exp), lw = lw, alpha_min = alpha_min, s_start = s_start, mark_last = mark_last, s_last = s_last)

        if year_clim is None:
            xm, ym = float(x.isel(year = slice(-n_end, None)).mean()), float(y.isel(year = slice(-n_end, None)).mean())
        else:
            xm, ym = float(x.sel(year = slice(year_clim[0], year_clim[1])).mean()), float(y.sel(year = slice(year_clim[0], year_clim[1])).mean())
        ax.scatter(xm, ym, s = s_dot, color = col, marker = 'o', edgecolors = col, alpha = alpha, zorder = 3)
        if exp_labels:
            ax.annotate(exp, (xm, ym), xytext = (6, 6), textcoords = 'offset points', fontsize = 12, ha = 'left', color = 'black', zorder = 6)
        logging.debug(f'scatter: {xm}, {ym}')

    ax.autoscale_view()  # needed because add_collection doesn't rescale the axes
    if x_ext is not None: ax.set_xlim(x_ext)
    if y_ext is not None: ax.set_ylim(y_ext)

    if symmetric_axes:
        max_ext = max(map(abs, ax.get_xlim() + ax.get_ylim()))
        ax.set_xlim(-max_ext, max_ext)
        ax.set_ylim(-max_ext, max_ext)
        ax.set_aspect('equal')

    # target bands and zero lines, spanning the final axes
    if x_target is not None: ax.axvspan(x_target[0], x_target[1], color = 'grey', alpha = 0.2, lw = 0, zorder = 0)
    if y_target is not None: ax.axhspan(y_target[0], y_target[1], color = 'grey', alpha = 0.2, lw = 0, zorder = 0)
    if symmetric_axes or y_target is not None: ax.axhline(0., color = 'grey', lw = 0.5, zorder = 0)
    if symmetric_axes or x_target is not None: ax.axvline(0., color = 'grey', lw = 0.5, zorder = 0)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if labels is None:
        _exp_legend(ax)
    else:
        if colors_legend is None:
            if len(colors) == len(labels):
                colors_legend = colors
            else:
                raise ValueError("specify colors for labels in legend")
        legend_elements = [Patch(facecolor=col, label=lab) for col, lab in zip(colors_legend, labels)]

        ax.legend(handles=legend_elements)

    name = '-'.join(exps)
    try:
        fig.savefig(cart_out + f'check_{x_ds.name}_vs_{y_ds.name}_{name}.pdf', dpi = 300)
    except:
        logging.error('could not output fig!!')
        
    plt.show()

    return fig

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

            ax.plot(atmclim.lat, y, label = exp_label(exp), color = col)
            ax.text(100, y.values[-1], exp, fontsize=12, ha='right', color = col)
            
        if not plot_anomalies: 
            if weighted: y_ref *= weights
            ax.plot(atmclim.lat, y_ref, label = 'CERES', color = 'black')
            ax.text(100, y_ref.values[-1], 'CERES', fontsize=12, ha='right', color = 'black')

        ax.set_xlabel('lat')
        add = ''
        if weighted: add = ' (weighted with cosine)'
        ax.set_title(var_label(var, units = False))
        if plot_anomalies:
            ax.set_ylabel(f'{var_label(var, units = False)} bias wrt CERES 2001-2011 (W/m2)'+add)
        else:
            ax.set_ylabel(f'{var_label(var, units = False)} vs CERES 2001-2011 (W/m2)'+add)

        plt.xlim(-90, 105)
        if ylim is not None: plt.ylim(ylim)
        #plt.legend()

        add = ''
        if not plot_anomalies: add = '_full'
        if weighted: add += '_weighted'

        name = '-'.join(exps)
        logging.info(f'Saving figure: {name}{add}')
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

            ax.plot(atmclim.lat, y, label=exp_label(exp), color=col)
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
        ax.set_title(var_label(var, units = False))
        if plot_anomalies:
            ax.set_ylabel(f'{var_label(var, units = False)} bias wrt {ref_exp} (W/m2)' + add)
        else:
            ax.set_ylabel(f'{var_label(var, units = False)} vs {ref_exp} (W/m2)' + add)

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

            ax.set_ylabel(var_label(var))
            ax.set_title(f"{param} — {var_label(var, units = False)}", fontsize=13)
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

def plot_map_var(clim, weights, exps, var, ref_exp = None, cart_out = cart_out, last_years = None,
                  ncols = 3, projection = ccrs.Robinson(), cmap = None, nlevels = 10,
                  vmax = None, cbar_label = ''):
    """
    Multi-panel lat-lon contourf maps with coastlines, one panel per exp.
    If ref_exp is given, plots (exp - ref_exp) for all the other exps, otherwise absolute values.
    If the data has a 'year' dim, only the last `last_years` years are averaged.
    """
    def get_field(exp):
        da = clim[exp][var]
        if 'year' in da.dims:
            if last_years is not None:
                da = da.isel(year = slice(-last_years, None))
            da = da.mean('year')
        regrid = Regridder(weights = weights[exp])
        return regrid.regrid(da)

    if ref_exp is not None and ref_exp not in exps:
        logging.info(f'WARNING: {ref_exp} not in exps! plotting absolute values')
        ref_exp = None

    plot_exps = [e for e in exps if e != ref_exp] if ref_exp is not None else list(exps)

    fields = {e: get_field(e) for e in plot_exps}
    if ref_exp is not None:
        ref = get_field(ref_exp)
        fields = {e: f - ref for e, f in fields.items()}

    # shared colour scale: symmetric for differences, min/max for absolute values
    if ref_exp is not None:
        if vmax is None:
            vmax = max(float(np.nanpercentile(np.abs(f.values), 98)) for f in fields.values())
        levels = np.linspace(-vmax, vmax, nlevels)
        cmap = cmap or 'RdBu_r'
    else:
        vmin_ = min(float(np.nanmin(f.values)) for f in fields.values())
        vmax_ = max(float(np.nanmax(f.values)) for f in fields.values())
        levels = np.linspace(vmin_, vmax_, nlevels)
        cmap = cmap or 'viridis'

    ncols = min(ncols, len(plot_exps))
    nrows = int(np.ceil(len(plot_exps) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize = (6*ncols, 3.5*nrows + 1),
                             subplot_kw = {'projection': projection},
                             squeeze = False, constrained_layout = True)

    for ax, exp in zip(axes.flat, plot_exps):
        f = fields[exp]
        data, lon = add_cyclic_point(f.values, coord = f.lon)  # avoids white stripe at the seam
        cf = ax.contourf(lon, f.lat, data, levels = levels, cmap = cmap, extend = 'both',
                         transform = ccrs.PlateCarree())
        ax.coastlines(linewidth = 0.8)
        ax.add_feature(cfeature.BORDERS, linewidth = 0.3, linestyle = ':')
        ax.set_global()
        ax.set_title(exp if ref_exp is None else f'{exp} - {ref_exp}')

        ax.gridlines(crs = ccrs.PlateCarree(), draw_labels = False,
            linewidth = 0.5, color = 'gray', alpha = 0.5, linestyle = '--',
            xlocs = mticker.FixedLocator(np.arange(-180, 181, 30)),
            ylocs = mticker.FixedLocator(np.arange(-90, 91, 30)))

    # hide unused panels
    for ax in axes.flat[len(plot_exps):]:
        ax.set_visible(False)

    da0 = clim[plot_exps[0]][var]
    fig.suptitle(var_label(var, da0, units = False), fontsize = 14)
    label = cbar_label or (var_label(var, da0) if ref_exp is None else f'Δ {var_label(var, da0)} (wrt {ref_exp})')
    fig.colorbar(cf, ax = axes, orientation = 'horizontal', shrink = 0.6, pad = 0.03, label = label)

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
            logging.info(f'AMOC not computed for {exp}')
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
        logging.info(f'WARNING: {ref_exp} not in exps! plotting absolute values')
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

        plt.plot(atmclim.lat, y, label = exp_label(exp), color = col)
        plt.text(100, y.values[-1], exp, fontsize=12, ha='right', color = col)
        
    ax.axhline(0., color = 'grey')
    plt.xlim(-90, 105)
    #plt.legend()
    
    ax.set_xlabel('lat')
    if ref_exp is not None:
        name = '-'.join([exp for exp in exps if exp != ref_exp])
        ax.set_ylabel(f'Δ zonal {var_label("tas")} (wrt {ref_exp})')
        fig.savefig(cart_out + f'check_zonal_tas_{name}_vs_{ref_exp}.pdf')
    else:    
        name = '-'.join(exps)
        ax.set_ylabel(f'zonal {var_label("tas")}')
        fig.savefig(cart_out + f'check_zonal_tas_{name}.pdf')


    return fig


def plot_zonal_var(atmclim, exps, var, ref_exp = None, cart_out = cart_out, colors = None):
    # Missing tas reference
    atmclim = create_ds_exp(atmclim)
    atmclim = atmclim.groupby('lat').mean()

    if ref_exp is not None and ref_exp not in exps:
        logging.info(f'WARNING: {ref_exp} not in exps! plotting absolute values')
        ref_exp = None

    fig, ax = plt.subplots(figsize=(12, 8))

    y_ref = None
    if ref_exp is not None:
        y_ref = atmclim.sel(exp = ref_exp)[var]

    if colors is None:
        colors = get_colors(exps)

    for exp, col in zip(exps, colors):
        y = atmclim.sel(exp = exp)[var]
        
        if y_ref is not None: y = y - y_ref

        plt.plot(atmclim.lat, y, label = exp_label(exp), color = col)
        plt.text(100, y.values[-1], exp, fontsize=12, ha='right', color = col)
        
    ax.axhline(0., color = 'grey')
    plt.xlim(-90, 105)
    #plt.legend()
    
    da0 = atmclim[var]
    ax.set_ylabel(var_label(var, da0) if ref_exp is None else f'Δ {var_label(var, da0)} (wrt {ref_exp})')
    ax.set_xlabel('lat')
    ax.set_title(var_label(var, da0, units = False))

    return fig


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
            logging.info(f"⚠️ Skipping {f}: unexpected YAML structure ({type(data)}). Error: {e}")
            continue

        params = {}
        for block in tuning.values():
            for k, v in block.items():
                if v is not None:
                    try:
                        params[k] = float(v)
                    except ValueError:
                        logging.info(f"⚠️ Non-numeric value for {k} in {f}: {v}")
        param_dict[exp_name] = params

    logging.info(f"Loaded {len(param_dict)} tuning files from {folder}")
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

def regrid_to_regular_smm_safe(ds, target_grid="r180x90", method="ycon", grid_in=None):
    import shutil
    from smmregrid import cdo_generate_weights, Regridder

    os.environ["PATH"] += ":/usr/local/apps/cdo/2.5.1/bin"
    os.environ["CDO_PTHREADS"] = "1"

    if shutil.which("cdo") is None:
        logging.info("CDO not found in PATH. Skip regridding.")
        return ds

    if 'cell' not in ds.dims:
        logging.info("Dataset already on regular grid. Skip regrid.")
        return ds

    # If not provided, take the first timestep of the dataset
    if grid_in is None:
        grid_in = ds.isel(time_counter=0)

    try:
        weights = cdo_generate_weights(grid_in, target_grid=target_grid, method=method)
        regridder = Regridder(weights=weights)
        ds_reg = regridder.regrid(ds)
        logging.info(f"Regridding completed on {target_grid}")
        return ds_reg
    except Exception as e:
        logging.info(f"Regridding failed: {e}")
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
            logging.info(f"Slope {param} has extra dimensions {extra_dims}, averaging.")
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

        logging.info(f"\n=== PARAM {param} | exps = {exps}")

        if len(exps) != 2:
            logging.info(f"Parameter {param} does not have two experiments. Skip.")
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
                logging.info(f"Regridding {exp} with CDO on {target_grid}...")
                grid_file = filz[0]
                grid_in = xr.open_dataset(grid_file).isel(time_counter=0)
                ds = regrid_to_regular_smm_safe(ds, target_grid=target_grid, method="ycon", grid_in=grid_in)
                logging.info(f"Regrid completed: dims = {list(ds.dims.keys())}")
            ds['toa_net'] = ds['rsdt'] - ds['rlut'] - ds['rsut']
            ds = ds.rename({'time_counter': 'time'}).chunk({'time': 240})
            ds = ds.groupby('time.year').mean()
            ds_dict[exp] = ds

            logging.info(f"  Loaded dataset for {exp}: dims={list(ds.dims.keys())}")

        # --- retrieve parameter values
        logging.info("  YAML keys:", sorted(param_yaml.keys()))
        logging.info(f"  Trying to match exps: minus={exp_minus}, ref={ref_exp}, plus={exp_plus}")
        try:
            key_minus = normalize_exp_key(exp_minus, param_yaml.keys())
            key_ref   = normalize_exp_key(ref_exp,   param_yaml.keys())
            key_plus  = normalize_exp_key(exp_plus,  param_yaml.keys())
        except KeyError as e:
            logging.info(f"❌ Skip {param}: {e}")
            logging.info("   Available YAML keys:", sorted(param_yaml.keys()))
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
        logging.info(f" {param}: mean R²={r2_mean:.3f}, min R²={r2_min:.3f}")

    # --- Plot 1: slope per 30%
    logging.info("\nPlot 1: Sensitivity normalized (W/m² per 30%)")
    plot_all_slopes(slope_30pct_dict, r2_dict=r2_dict, vmin=-3, vmax=3, cmap='RdBu_r', r2_thresh=r2_thresh,
                    filename='plot_slope_per30pct.png', label='TOA Net (W/m² per 30% param change)')

    # --- Plot 2: physical effect (total anomaly)
    logging.info("\nPlot 2: Total effect minus→plus (W/m²)")
    plot_all_slopes(anom_full_dict, r2_dict=r2_dict, vmin=-10, vmax=10, cmap='RdBu_r', r2_thresh=r2_thresh,
                    filename='plot_anom_full.png', label='TOA Net anomaly (W/m²)')

    return slope_dict, r2_dict, slope_30pct_dict, anom_full_dict

#####################################################################################################################################################################

# ============================================================
################################################ MAIN FUNCTION ###########################

def _select_years(data, time_window):
    """Restrict a time series (Dataset/DataArray with 'year' or 'time_counter') to time_window=(y0, y1), inclusive."""
    if data is None or time_window is None:
        return data
    y0, y1 = time_window
    if 'year' in data.dims:
        return data.sel(year=slice(y0, y1))
    if 'time_counter' in data.dims:
        years = data['time_counter'].dt.year
        return data.isel(time_counter=((years >= y0) & (years <= y1)).values)
    return data


def _clim_for_plots(clim_all, exps, colors, time_window):
    """
    Shallow copy of clim_all with the time series (*_mean, amoc_ts) restricted to time_window.
    Exps with no years in the window are dropped from the series; returns the kept exps and their colors,
    so each exp keeps its color.
    """
    if time_window is None:
        return clim_all, list(exps), colors
    ts_keys = [k for k in ('atm_mean', 'oce_mean', 'ice_mean', 'amoc_ts', 'rho_mean') if clim_all.get(k) is not None]
    clim_plot = dict(clim_all)
    for k in ts_keys:
        clim_plot[k] = {exp: _select_years(ds, time_window) for exp, ds in clim_all[k].items()}

    def _has_years(ds):
        return ds is not None and ds.sizes.get('year', ds.sizes.get('time_counter', 0)) > 0

    kept = [exp for exp in exps if _has_years(clim_plot['atm_mean'].get(exp))]
    dropped = [exp for exp in exps if exp not in kept]
    if dropped:
        logging.warning(f'time_window {time_window}: no data for {dropped}, they are left out of the time series plots')
    for k in ts_keys:
        clim_plot[k] = {exp: ds for exp, ds in clim_plot[k].items() if exp in kept and _has_years(ds)}
    colors_plot = [colors[list(exps).index(exp)] for exp in kept]
    return clim_plot, kept, colors_plot


######################## COMPONENTS / PLOT TYPES ########################

_ICE_VARS = ['siconc', 'sivolu', 'sithic', 'snvolu', 'hfxerr']

# component -> {domain: [time-series variables]}; domain is the clim_all key prefix (atm -> atm_mean, amoc -> amoc_ts)
# Variables missing from the data are skipped. Ice: N and S next to each other (2-column panels).
COMPONENTS = {
    'radiation': {'atm': ['toa_net', 'rsut', 'rlut', 'rsntcs', 'rlntcs', 'crf', 'alb']},
    'surface':   {'atm': ['tas', 'pr', 'prsn', 'evspsbl', 'E-P', 'hfss', 'hfls', 'rsns', 'rlns', 'ps']},
    'energy':    {'atm': ['atm_imb', 'srf_net'], 'oce': ['oce_imb']},
    'ocean':     {'oce': ['tos', 'sos', 'zos', 'heatc', 'qt_oce', 'mldr10_1'], 'amoc': ['amoc']},
    'ice':       {'ice': [v + h for v in _ICE_VARS for h in ('_N', '_S')]},
}
PLOT_TYPES = ['timeseries', 'gregory', 'boxplot', 'zonal', 'maps']


def _as_list(sel, allowed, what):
    """'all' / None -> all allowed; str -> [str]; checks the names."""
    if sel is None or sel == 'all':
        return list(allowed)
    if isinstance(sel, str):
        sel = [sel]
    bad = [s for s in sel if s not in allowed]
    if bad:
        raise ValueError(f'unknown {what} {bad}, choose among {list(allowed)}')
    return list(sel)


def _comp_base_vars(comp):
    """Base variable names (no _N/_S) of a component, used to filter maps and zonal means."""
    out = set()
    for vlist in COMPONENTS[comp].values():
        out.update(v[:-2] if v[-2:] in ('_N', '_S') else v for v in vlist)
    return out


def _annual_series(clim_plot, domain, var, exp):
    """Annual global-mean series (1D DataArray on 'year') of var for exp, or None if missing/all NaN."""
    key = 'amoc_ts' if domain == 'amoc' else f'{domain}_mean'
    d = (clim_plot.get(key) or {}).get(exp)
    if d is None:
        return None
    if isinstance(d, xr.Dataset):
        if var not in d:
            return None
        d = d[var]
    elif domain != 'amoc':
        return None
    if 'time_counter' in d.dims:
        d = d.groupby('time_counter.year').mean()
    d = d.squeeze(drop = True)
    if d.ndim != 1 or 'year' not in d.dims or not np.isfinite(d.values).any():
        return None
    return d


def _component_series(clim_plot, comp, exps):
    """[(domain, var, {exp: series})] for the variables of comp that at least one exp has."""
    out = []
    for domain, vlist in COMPONENTS[comp].items():
        for var in vlist:
            series = {exp: _annual_series(clim_plot, domain, var, exp) for exp in exps}
            series = {exp: s for exp, s in series.items() if s is not None}
            if series:
                out.append((domain, var, series))
    return out


def _panel_grid(n, ncols, w = 5.2, h = 3.6):
    nrows = int(np.ceil(n / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize = (w * ncols, h * nrows), squeeze = False)
    axs = axs.ravel()
    for ax in axs[n:]:
        ax.set_visible(False)
    return fig, axs[:n]


def _fig_exp_legend(fig, exps, colors, patches = False):
    """One legend for the whole figure, to the right of the panels."""
    if patches:
        handles = [Patch(facecolor = to_rgba(c, 0.5), edgecolor = c, label = exp_label(e)) for e, c in zip(exps, colors)]
    else:
        handles = [plt.Line2D([], [], color = c, label = exp_label(e)) for e, c in zip(exps, colors)]
    fig.legend(handles = handles, loc = 'upper left', bbox_to_anchor = (1.0, 0.98), frameon = False,
               fontsize = 9 if EXP_LABELS else 11)


def plot_component_ts(clim_plot, comp, exps, colors = None, rolling = None, cart_out = cart_out, ncols = None):
    """
    One figure per component: a panel per variable with the annual global-mean time series of all exps
    (rolling mean if rolling is set). Returns the figure, or None if no variable is available.
    """
    if colors is None:
        colors = get_colors(exps)
    series = _component_series(clim_plot, comp, exps)
    if not series:
        logging.warning(f'{comp}: no variables available for the time series')
        return None
    ncols = ncols or (2 if comp == 'ice' else 3)
    fig, axs = _panel_grid(len(series), ncols)
    for ax, (domain, var, sdict) in zip(axs, series):
        for exp, col in zip(exps, colors):
            if exp not in sdict:
                continue
            y = sdict[exp]
            if rolling is not None:
                y = y.rolling(year = rolling, min_periods = 1).mean()
            ax.plot(y.year, y, color = col, lw = 1)
        ax.set_title(var_label(var, domain = domain), fontsize = 10)
        ax.set_xlabel('')
        ax.grid(alpha = 0.3)
    _fig_exp_legend(fig, exps, colors)
    roll = f' ({rolling}-yr rolling mean)' if rolling else ''
    fig.suptitle(f'{comp}: global-mean time series{roll}')
    fig.tight_layout(rect = (0, 0, 1, 0.97))
    fig.savefig(cart_out + f"check_ts_{comp}_{'-'.join(exps)}.pdf", bbox_inches = 'tight')
    return fig


def plot_component_box(clim_plot, comp, exps, colors = None, box_years = None, year_clim = None, ref_exp = None,
                       cart_out = cart_out, ncols = None):
    """
    One figure per component: a panel per variable with a boxplot of the annual global means of each exp
    (absolute values). Years used, as for the climatologies: year_clim=(y0, y1) if given, else the last
    box_years (default CLIM_YEARS = 20) of each exp. An explicit box_years wins over year_clim. Box = interquartile range, black line = median,
    white dot = mean, whiskers = 1.5 IQR. Runs shorter than box_years use all their years and show n
    under their box. If ref_exp is among exps, its median is drawn as a dashed line.

    Returns (fig, stats): stats is a DataFrame with n, years, mean, std, median and linear trend
    (per century) of each exp and variable over the same years. Also saved as csv next to the figure.
    """
    if colors is None:
        colors = get_colors(exps)
    if box_years is None and year_clim is not None:
        n_full = year_clim[1] - year_clim[0] + 1
        select = lambda s: s.dropna('year').sel(year = slice(year_clim[0], year_clim[1]))
        win_txt, win_tag = f'years {year_clim[0]}-{year_clim[1]}', f'{year_clim[0]}-{year_clim[1]}'
    else:
        n_full = box_years or CLIM_YEARS
        select = lambda s: s.dropna('year').isel(year = slice(-n_full, None))
        win_txt, win_tag = f'last {n_full} years of each run', f'last{n_full}y'
    series = _component_series(clim_plot, comp, exps)
    if not series:
        logging.warning(f'{comp}: no variables available for the boxplots')
        return None, pd.DataFrame()
    ncols = ncols or (2 if comp == 'ice' else 3)
    fig, axs = _panel_grid(len(series), ncols, w = max(4.5, 0.75 * len(exps) + 1.5))
    rows = []
    pos = np.arange(len(exps))
    for ax, (domain, var, sdict) in zip(axs, series):
        ticklabels = []
        for i, (exp, col) in enumerate(zip(exps, colors)):
            s = sdict.get(exp)
            if s is not None:
                s = select(s)
            if s is None or s.size == 0:
                ticklabels.append(exp)
                continue
            vals, yrs = s.values, s.year.values
            ax.boxplot(vals, positions = [i], widths = 0.6, patch_artist = True, showmeans = True,
                       boxprops = dict(facecolor = to_rgba(col, 0.5), edgecolor = col),
                       whiskerprops = dict(color = col), capprops = dict(color = col),
                       medianprops = dict(color = 'black'),
                       meanprops = dict(marker = 'o', markerfacecolor = 'white', markeredgecolor = 'black', markersize = 4),
                       flierprops = dict(marker = '.', markeredgecolor = col, markersize = 4))
            ticklabels.append(exp if len(vals) >= n_full else f'{exp}\nn={len(vals)}')
            trend = np.polyfit(yrs, vals, 1)[0] * 100 if len(vals) > 2 else np.nan
            rows.append(dict(component = comp, var = var, exp = exp, n = len(vals),
                             first_year = int(yrs[0]), last_year = int(yrs[-1]),
                             mean = float(np.mean(vals)), std = float(np.std(vals, ddof = 1)) if len(vals) > 1 else np.nan,
                             median = float(np.median(vals)), trend_per_century = float(trend)))
        if ref_exp is not None and ref_exp in sdict:
            ref_vals = select(sdict[ref_exp]).values
            if ref_vals.size:
                ax.axhline(np.median(ref_vals), color = 'grey', ls = '--', lw = 0.8, zorder = 0)
        ax.set_xticks(pos)
        ax.set_xticklabels(ticklabels, fontsize = 8, rotation = 45 if len(exps) > 8 else 0)
        ax.set_xlim(-0.6, len(exps) - 0.4)
        ax.set_title(var_label(var, domain = domain), fontsize = 10)
        ax.grid(axis = 'y', alpha = 0.3)
    _fig_exp_legend(fig, exps, colors, patches = True)
    ref_txt = f'; dashed: {ref_exp} median' if ref_exp in exps else ''
    fig.suptitle(f'{comp}: annual global means, {win_txt}{ref_txt}')
    fig.tight_layout(rect = (0, 0, 1, 0.97))
    name = '-'.join(exps)
    fig.savefig(cart_out + f'check_box_{comp}_{win_tag}_{name}.pdf', bbox_inches = 'tight')
    box_df = pd.DataFrame(rows)
    box_df.to_csv(cart_out + f'box_stats_{comp}_{win_tag}_{name}.csv', index = False)
    return fig, box_df


def compare_multi_exps(exps, user = None, read_again = [], cart_exp = '/ec/res4/scratch/{}/ece4/', cart_out = './output/',
                       imbalance = 0., ref_exp = None, atm_only = False, 
                       atmvars = 'rsut rlut rsdt tas pr alb rsntcs rlntcs hfss hfls rsns rlns prsn ps evspsbl'.split(), 
                       ocevars = 'tos zos heatc qt_oce sos mldr10_1'.split(), 
                       icevars = 'siconc sivolu sithic snvolu hfxerr'.split(), year_clim = None, plot_diffref=False, plot_param=False, 
                       param_map={}, skip_first_year=False, exp_type = 'PD', density=False, colors=None, 
                       rolling = None, file_lists = None, plot_zonal_vars = None, plot_map_atm_vars = ['alb', 'tas', 'pr'],
                       plot_map_oce_vars = ['tos', 'zos', 'sos'], plot_map_ice_vars = ['siconc', 'sithic', 'snvolu'],
                       ongoing = [], do_all_from_scratch = False, time_window = None,
                       exp_labels = None, components = 'all', plot_types = 'all', box_years = None,
                       loglevel=logging.INFO):
    """
    Runs all multi-exps diagnostics.

    exps: list of experiments to consider
    cart_exp: base dir for experiments (defaults as $SCRATCH on hpc2020)
    user: to set experiment dir using cart_exp template. If a list, specifies a different user for every exp
    read_again: list of exps to read again. If set, overwrites existing clims for exp to update them (useful if sims are still running)
    time_window: optional (first_year, last_year), inclusive. Restricts the time series, Gregory, AMOC-vs-GTAS and
        imbalance plots to these years (the big dot is then the mean of the last 20 years inside the window, or year_clim).
        Maps and zonal means still use the saved climatologies (year_clim). The returned clim_all is not cut.
    exp_labels: optional {exp: short description}, shown in the legends as 'exp: description'
        (e.g. {'xb09': '3 ice steps (NEMO 3600 s)'}). Experiments without a description keep their id.
    components: 'all' or a list among COMPONENTS ('radiation', 'surface', 'energy', 'ocean', 'ice').
        Each component is a group of variables (edit COMPONENTS to change them).
    plot_types: 'all' or a list among PLOT_TYPES:
        'timeseries' one multi-panel figure per component (a panel per variable),
        'boxplot'    one multi-panel figure per component, annual means over the last box_years of each run;
                     the statistics go to clim_all['box_stats'] (DataFrame) and to csv files,
        'gregory'    TOA vs tas (radiation), AMOC vs GTAS (ocean), atm vs oce imbalance (energy),
        'zonal'      zonal means (radiation: vs CERES; surface: tas and the other atm vars),
        'maps'       lat-lon maps of the plot_map_*_vars in the component; AMOC section for ocean.
    box_years: number of final years used in the boxplots (inside time_window if set). None (default) = same
        years as the climatologies: year_clim if given, else the last CLIM_YEARS (20).
    The figures are also returned grouped as clim_all['figs'][(component, plot_type)].
    """
    logging.basicConfig(level=loglevel, format='%(asctime)s - %(levelname)s - %(message)s')
    global EXP_LABELS
    EXP_LABELS = dict(exp_labels or {})
    components = _as_list(components, COMPONENTS, 'components')
    plot_types = _as_list(plot_types, PLOT_TYPES, 'plot_types')

    if cart_out is None:
        raise ValueError('cart_out not specified!')
    
    if not os.path.exists(cart_out): os.mkdir(cart_out)

    if plot_zonal_vars is None:
        plot_zonal_vars = atmvars

    if plot_map_atm_vars is None:
        plot_map_atm_vars = atmvars

    if plot_map_oce_vars is None:
        plot_map_oce_vars = ocevars
    
    if plot_map_ice_vars is None:
        # hfxerr only as time series: it is NaN in runs without the SI3 process diagnostics (e.g. the reference)
        plot_map_ice_vars = [v for v in icevars if v != 'hfxerr']

    cart_out_nc = cart_out + '/exps_clim/'
    tw_suffix = f'_{time_window[0]}-{time_window[1]}' if time_window is not None else ''
    cart_out_figs = cart_out + f"/check_{'-'.join(exps)}{tw_suffix}/"

    if not os.path.exists(cart_out_nc): os.mkdir(cart_out_nc)
    if not os.path.exists(cart_out_figs): os.mkdir(cart_out_figs)

    ### read outputs for all exps
    clim_all = read_output(exps, user = user, read_again = read_again, cart_exp = cart_exp, cart_out = cart_out_nc, atm_only = atm_only, atmvars = atmvars, ocevars = ocevars, icevars = icevars, year_clim = year_clim, density=density, file_lists = file_lists, ongoing = ongoing, do_all_from_scratch=do_all_from_scratch)

    coupled = False
    if 'amoc_ts' in clim_all: coupled = True

    if colors is None:
        colors = get_colors(exps)
    # time series restricted to time_window (maps/zonal use clim_all, unchanged)
    clim_plot, exps_ts, colors_ts = _clim_for_plots(clim_all, exps, colors, time_window)

    allfigs = []
    figs_by = {}   # (component, plot_type) -> list of figures, also returned in clim_all['figs']

    def _add(comp, ptype, figs):
        figs = [f for f in (figs if isinstance(figs, list) else [figs]) if f is not None]
        figs_by.setdefault((comp, ptype), []).extend(figs)
        allfigs.extend(figs)

    has_atm_var = lambda v: v in list(clim_all['atm_mean'].items())[0][1]
    has_oce_var = lambda v: coupled and v in list(clim_all['oce_mean'].items())[0][1]
    box_stats = []

    for comp in components:
        if comp in ('ocean', 'ice') and not coupled:
            continue
        cvars = _comp_base_vars(comp)
        logging.info(f'--- {comp}: {plot_types}')

        if 'timeseries' in plot_types:
            _add(comp, 'timeseries', plot_component_ts(clim_plot, comp, exps_ts, colors = colors_ts,
                                                       rolling = rolling, cart_out = cart_out_figs))

        if 'boxplot' in plot_types:
            fig, box_df = plot_component_box(clim_plot, comp, exps_ts, colors = colors_ts, box_years = box_years,
                                            year_clim = year_clim, ref_exp = ref_exp, cart_out = cart_out_figs)
            _add(comp, 'boxplot', fig)
            box_stats.append(box_df)

        if 'gregory' in plot_types:
            if comp == 'radiation':
                _add(comp, 'gregory', plot_greg(clim_plot['atm_mean'], exps_ts, imbalance = imbalance, ylim = None,
                                                cart_out = cart_out_figs, exp_type = exp_type, year_clim = year_clim,
                                                colors = colors_ts, rolling = rolling))
            elif comp == 'ocean' and clim_all.get('amoc_ts') is not None:
                _add(comp, 'gregory', plot_amoc_vs_gtas(clim_plot, exps_ts, lw = 0.8, cart_out = cart_out_figs,
                                                        exp_type = exp_type, year_clim = year_clim,
                                                        colors = colors_ts, rolling = rolling))
            elif comp == 'energy' and has_atm_var('atm_imb') and has_oce_var('oce_imb'):
                _add(comp, 'gregory', plot_imbalance(clim_plot, cart_out = cart_out_figs, rolling = rolling))

        if 'zonal' in plot_types:
            if comp == 'radiation':
                _add(comp, 'zonal', plot_zonal_fluxes_vs_ceres(clim_all['atm_clim'], exps = exps,
                                                               cart_out = cart_out_figs, colors = colors))
            if comp == 'surface':
                _add(comp, 'zonal', plot_zonal_tas_vs_ref(clim_all['atm_clim'], exps = exps, ref_exp = ref_exp,
                                                          cart_out = cart_out_figs, colors = colors))
            for var in atmvars:
                if var in cvars and var in plot_zonal_vars and var not in 'rsut rlut rsdt tas'.split():
                    logging.info(f'Plotting zonal for {var}')
                    _add(comp, 'zonal', plot_zonal_var(clim_all['atm_clim'], exps = exps, var = var,
                                                       ref_exp = ref_exp, colors = colors))

        if 'maps' in plot_types:
            map_sets = [('atm', atmvars, plot_map_atm_vars)]
            if coupled:
                map_sets += [('oce', ocevars, plot_map_oce_vars), ('ice', icevars, plot_map_ice_vars)]
            for domain, dvars, mapvars in map_sets:
                for var in dvars:
                    if var in cvars and var in mapvars:
                        logging.info(f'Plotting map for {var}')
                        _add(comp, 'maps', plot_map_var(clim = clim_all[f'{domain}_clim'],
                                                        weights = clim_all[f'{domain}_weights'],
                                                        exps = exps, var = var, ref_exp = ref_exp))
            if comp == 'ocean' and clim_all.get('amoc_mean') is not None:
                _add(comp, 'maps', plot_amoc_2d_all(clim_all['amoc_mean'], exps, cart_out = cart_out_figs))

    if box_stats:
        clim_all['box_stats'] = pd.concat(box_stats, ignore_index = True)
    clim_all['figs'] = figs_by

    # --- Optional diagnostics for tuning experiments
    if plot_diffref:
        figs_diffref = plot_zonal_fluxes_vs_ref(clim_all['atm_clim'], exps=exps, ref_exp=ref_exp, cart_out=cart_out_figs)
        _debug_check(figs_diffref, 'zonal_fluxes_vs_ref')
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
        _debug_check(figs_param, 'zonal_fluxes_by_param')
        allfigs += figs_param

    logging.info(f'Done! Check results in {cart_out_figs}')

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
    
    # Example: logging.info loaded configuration
    logging.info(f"Loaded configuration for user: {user}")
    logging.info(f"Experiments: {exps}")
    logging.info(f"User: {user}")
    logging.info(f"Read again: {read_again}")
    logging.info(f"Cart exp: {cart_exp}")
    logging.info(f"Cart out: {cart_out}")

    clim_all, figs = compare_multi_exps(exps, user = user, read_again = read_again, cart_exp = cart_exp, cart_out = cart_out, imbalance = imbalance, ref_exp = ref_exp, plot_param=plot_param, plot_diffref=plot_diffref, param_map=param_map,skip_first_year=skip_first_year)

    return clim_all, figs
    

# Main execution
if __name__ == '__main__':
    main()