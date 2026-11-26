"""
Energy budget closure across the interfaces of EC-Earth4: TOA -> atmosphere (OIFS) -> OIFS surface ->
OASIS coupling -> sea ice (SI3) -> liquid ocean (NEMO) -> ocean heat content.

Builds on check_run (read_output, compute_interface_balance, geothermal_flux, areas and masks) and adds:
  - read_water_heat: the heat content of the water fluxes that NEMO books itself (qemp_oce, qemp_ice,
    hflx_{evap,rain,snow,rnf,cal,icb}_cea), available only in runs that write them (xax0, ACX0);
  - budget_terms: every term of the chain per run and year, with the residual of each interface;
  - budget_table, budget_checks: the terms per run and the closure check of each box;
  - plot_budget_flow, plot_residuals, plot_terms_ts: figures.

Sign convention: fluxes are downward (into the lower compartment); residuals are >0 when energy is created.
Areas: the atmosphere terms are per unit Earth area; the coupling, ice and ocean terms per unit NEMO ocean area.
"""
import logging
import os
import textwrap

import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import pyplot as plt
from matplotlib.patches import FancyBboxPatch

import check_run as cr

# heat content of the water fluxes, as NEMO writes them (W m-2 per cell area)
WATER_HEAT_OCE = ['qemp_oce', 'hflx_evap_cea', 'hflx_rain_cea', 'hflx_snow_cea',
                  'hflx_rnf_cea', 'hflx_cal_cea', 'hflx_icb_cea']        # _1m_T files
WATER_HEAT_ICE = ['qemp_ice']                                           # _1m ice files

# variables read through check_run.read_output (same lists as compare_multi_exps, so the caches are shared)
ATMVARS = 'rsut rlut rsdt tas pr alb rsntcs rlntcs hfss hfls rsns rlns prsn ps evspsbl ci snow_ice'.split()
OCEVARS = 'tos zos heatc qt_oce sos mldr10_1'.split()
ICEVARS = 'siconc sivolu sithic snvolu hfxerr qt_atm_oi qt_oce_ai'.split()

# terms of the chain: name -> (long name, area)
TERMS = {
    'toa':          ('Net TOA radiation', 'Earth'),
    'srf_net':      ('OIFS net surface flux (with snowfall melting)', 'Earth'),
    'atm_imb':      ('Atmosphere residual = -(TOA - surface)', 'Earth'),
    'srf_oce_glob': ('OIFS net surface flux over ocean, share of the global mean', 'Earth'),
    'srf_land_glob': ('OIFS net surface flux over land, share of the global mean', 'Earth'),
    'srf_oce':      ('OIFS net surface flux over ocean, mean per OIFS ocean area', 'OIFS ocean'),
    'srf_land':     ('OIFS net surface flux over land, mean per land area', 'OIFS land'),
    'land_res':     ('Land residual = calving to the ocean - land flux (no land storage assumed)', 'Earth'),
    'total_res':    ('Total residual = heat uptake of ice + ocean - TOA - geothermal', 'Earth'),
    'frac_oce':     ('OIFS ocean fraction of the Earth', '-'),
    'area_oce':     ('OIFS ocean area (1e12 m2)', '-'),
    'area_land':    ('OIFS land area (1e12 m2)', '-'),
    'area_nemo':    ('NEMO ocean area (1e12 m2)', '-'),
    'area_ratio':   ('OIFS ocean area / NEMO ocean area', '-'),
    'F_oifs':       ('OIFS net surface flux over ocean with snowfall melting, rescaled to the NEMO ocean area', 'ocean'),
    'F_oifs_nosn':  ('OIFS net surface flux over its ocean points without snowfall melting (grid-box mean; OASIS carries the ocean and ice tile fluxes)', 'ocean'),
    'snow_melt':    ('Latent heat of snowfall over the ocean = F_oifs - F_oifs_nosn (OIFS)', 'ocean'),
    'qt_atm_oi':    ('SI3: atmosphere -> ice + ocean (what NEMO receives, with qemp)', 'ocean'),
    'qemp':         ('E-P heat flux added by NEMO (qemp_oce + qemp_ice, with snowfall melting)', 'ocean'),
    'qemp_sens':    ('qemp - snow_melt = evap_rain_ow + snow_res', 'ocean'),
    'qemp_oce':     ('E-P heat flux over open water (NEMO)', 'ocean'),
    'qemp_ice':     ('Snowfall heat flux over sea ice (NEMO): snow mass added to the ice', 'ocean'),
    'snow_ow':      ('Snowfall over open water, melted by NEMO = hflx_snow_cea - qemp_ice', 'ocean'),
    'evap_rain_ow': ('Heat content of evaporation and rain over open water = qemp_oce - snow_ow', 'ocean'),
    'hflx_evap_cea': ('Heat content of evaporation (E x cp x SST, NEMO)', 'ocean'),
    'hflx_rain_cea': ('Heat content of rain (NEMO)', 'ocean'),
    'hflx_snow_cea': ('Heat content of snowfall incl. melting (NEMO)', 'ocean'),
    'cal_latent':   ('Calving from the OIFS land (runoff mapper): latent heat paid by the ocean (hflx_cal_cea + hflx_icb_cea)', 'ocean'),
    'interface_imb': ('Coupling residual = qt_atm_oi - F_oifs = qemp_sens + cal_latent + oasis_res', 'ocean'),
    'oasis_recv':   ('Received from OASIS = qt_atm_oi - qemp - cal_latent', 'ocean'),
    'oasis_res':    ('OASIS residual = received - sent = qt_atm_oi - qemp - cal_latent - F_oifs_nosn', 'ocean'),
    'nemo_added':   ('= qemp_sens = evap_rain_ow + snow_res', 'ocean'),
    'ice_storage':  ('Sea-ice heat storage = qt_atm_oi - qt_oce_ai', 'ocean'),
    'qt_ow':        ('NEMO: atmosphere -> open water (qt_oce, with qemp_oce)', 'ocean'),
    'qt_ai':        ('NEMO: atmosphere -> sea ice = qt_atm_oi - qt_oce (with qemp_ice)', 'ocean'),
    'ice_to_oce':   ('Sea ice -> liquid ocean = qt_oce_ai - qt_oce (melting, freezing, ...)', 'ocean'),
    'recv_ow':      ('Received from OASIS over open water = qt_ow - qemp_oce - cal_latent', 'ocean'),
    'recv_ice':     ('Received from OASIS over sea ice = qt_ai - qemp_ice', 'ocean'),
    'heat_uptake':  ('Heat uptake of sea ice + liquid ocean = ice_storage + ohc_trend', 'ocean'),
    'qt_oce_ai':    ('SI3: heat reaching the liquid ocean', 'ocean'),
    'runoff_heat':  ('Heat content of runoff, entering at max(SST, 0 C) (hflx_rnf_cea)', 'ocean'),
    'F_geo':        ('Geothermal flux (Goutorbe, NEMO ocean mean)', 'ocean'),
    'ohc_trend':    ('Ocean heat content tendency', 'ocean'),
    'oce_imb':      ('Ocean residual = OHC tendency - qt_oce_ai - F_geo (includes runoff heat)', 'ocean'),
    'oce_res':      ('Ocean residual without runoff heat = oce_imb - runoff_heat', 'ocean'),
    'snow_res':     ('Snowfall heat, NEMO vs OIFS = hflx_snow_cea - snow_melt (snow sensible heat, remapping)', 'ocean'),
    'water_cycle':  ('Water heat content booked by NEMO only (evaporation, rain, runoff at c_p T) = evap_rain_ow + runoff_heat', 'ocean'),
    'end_to_end':   ('End to end = interface_imb + oce_imb = oasis_res + cal_latent + snow_res + water_cycle + oce_res', 'ocean'),
    'e2e_ocean':    ('Ocean chain without the calving = oasis_res + snow_res + water_cycle + oce_res', 'ocean'),
    # OASIS band: each flux with what OIFS gives (in) and what NEMO/SI3 reads (out)
    'snow_ice_oifs': ('OIFS snowfall melting over its sea ice (-Lf prsn ci, monthly), rescaled to the NEMO ocean area', 'ocean'),
    'snow_ow_oifs': ('OIFS snowfall melting over open water = snow_melt - snow_ice_oifs', 'ocean'),
    'snow_res_ice': ('Snow on ice, NEMO vs OIFS = qemp_ice - snow_ice_oifs', 'ocean'),
    'snow_res_ow':  ('Snow on open water, NEMO vs OIFS = snow_ow - snow_ow_oifs', 'ocean'),
    'land_in':      ('OIFS land surface flux, leaves as calving (no land storage), per NEMO ocean area', 'ocean'),
    'band_in':      ('OASIS band in: OIFS surface flux over ocean (with snowfall) + calving = F_oifs + cal_latent', 'ocean'),
    'band_out':     ('OASIS band out: what NEMO books = qt_atm_oi + runoff_heat', 'ocean'),
    'band_res':     ('OASIS band residual = band_out - band_in = oasis_res + snow + evaporation/rain + runoff', 'ocean'),
    'band_in_std':  ('OASIS band in from standard output: calving taken = land flux (no HTESSEL residual) = F_oifs + land_in', 'ocean'),
    'band_res_std': ('OASIS band residual from standard output = qt_atm_oi - band_in_std (incl. HTESSEL; runoff heat in NEMO / SI3)', 'ocean'),
    'oifs_res':     ('OIFS / HTESSEL residual = atm_imb + land_res (Earth units)', 'Earth'),
}
RESIDUALS = ['atm_imb', 'land_res', 'oasis_res', 'snow_res', 'water_cycle', 'band_res', 'oce_res', 'total_res']
RESIDUAL_LABELS = {'atm_imb': 'ε OIFS\n(surface − TOA)', 'land_res': 'ε HTESSEL\n(calving − land)', 'oasis_res': 'OASIS heat\n(out − in)',
                   'snow_res': 'OASIS snow\n(out − in)', 'water_cycle': 'water heat\n(NEMO c$_p$T)', 'band_res': 'ε OASIS\n(band)',
                   'oce_res': 'ε NEMO / SI3', 'total_res': 'ε total', 'interface_imb': 'coupling, all', 'oce_imb': 'ocean, all\n(+ runoff heat)',
                   'end_to_end': 'end to end'}


def _ocean_mean(ds, exp, user, cart_exp, variables):
    """
    Annual means integrated over the NEMO ocean and divided by its area (W m-2). The spatial integral is
    taken first, month by month (lazy), so only the series are loaded, not the 2D fields.
    """
    area = cr.get_areas_nemo(exp, user, cart_exp = cart_exp) * cr.get_mask_nemo(exp, user, cart_exp = cart_exp)
    ds = cr._rename_xy(ds[variables])
    series = (ds * area).sum(['x', 'y'], min_count = 1) / float(np.nansum(area))
    return series.groupby('time_counter.year').mean().compute()


def _present(files, variables):
    """Variables of the list that are in the first readable file (cheap check, nothing loaded)."""
    import netCDF4
    for f in files:
        try:
            with netCDF4.Dataset(f) as nc:
                return [v for v in variables if v in nc.variables]
        except OSError:
            continue
    return []


def read_water_heat(exp, user, cart_exp = cr.cart_exp, cart_out = cr.cart_out, last_year = None):
    """
    Heat content of the water fluxes booked by NEMO, per NEMO ocean area (W m-2), annual means: qemp_oce,
    qemp_ice, hflx_{evap,rain,snow,rnf,cal,icb}_cea. Cached in cart_out/budget_water_<exp>.nc and recomputed
    when it is behind: behind last_year (give the last year of the saved means, so that all terms cover the
    same years), or, without last_year, behind the last output file by more than one year.
    Returns None if the run does not write them.
    """
    fa, _, fn, _, fi = cr.file_list(exp, user, cart_exp = cart_exp)
    files_t, files_i = cr.select_files(fn), cr.select_files(fi)
    vo = _present(files_t, WATER_HEAT_OCE)
    if not vo:      # qemp_ice alone is in the standard ice output of every run: not enough for the decomposition
        return None
    vi = _present(files_i, WATER_HEAT_ICE)
    cache = cart_out + f'budget_water_{exp}.nc'
    last = cr.last_file_year(files_t)
    if cr._nc_readable(cache):
        old = xr.load_dataset(cache)
        target = last_year if last_year is not None else (last or 0) - 1   # the last file may still be written
        if old.sizes['year'] and int(old.year[-1]) >= target:
            return old
    parts = []
    try:
        if vo:
            parts.append(_ocean_mean(cr.open_output(files_t), exp, user, cart_exp, vo))
        if vi:
            parts.append(_ocean_mean(cr.open_output(files_i), exp, user, cart_exp, vi))
    except cr.NoOutputYet as err:
        logging.warning(f'{exp}: no complete year of water heat terms yet ({err})')
        return None
    ds = xr.merge(parts, join = 'inner')
    for v in WATER_HEAT_OCE + WATER_HEAT_ICE:
        if v not in ds:
            ds[v] = xr.full_like(ds[list(ds.data_vars)[0]], np.nan)
    cr._save_nc(ds, cache)
    logging.info(f'[budget] {exp}: water heat terms for {int(ds.year[0])}-{int(ds.year[-1])} saved in {os.path.basename(cache)}')
    return ds


def budget_terms(clim_all, exps, users, water = None, cart_exp = cr.cart_exp):
    """
    All the terms of the chain (see TERMS) per run, as a Dataset on 'year'. clim_all from check_run.read_output
    (compute_interface_balance already applied); water: {exp: read_water_heat(...)}. Terms that need data a run
    does not have are NaN (e.g. qemp, oasis_res, runoff_heat, oce_res without the water heat terms).
    """
    users = dict(zip(exps, users)) if not isinstance(users, dict) else users
    out = {}
    for exp in exps:
        atm = clim_all.get('atm_mean', {}).get(exp)
        oce = clim_all.get('oce_mean', {}).get(exp)
        ice = clim_all.get('ice_mean', {}).get(exp)
        if atm is None:
            continue
        us = users[exp]
        # only the years every source has, so that all the terms are averaged over the same years
        years = atm.year.values
        for src in (oce, ice, (water or {}).get(exp)):
            if src is not None:
                years = np.intersect1d(years, src.year.values)
        atm = atm.sel(year = years)
        oce = oce.sel(year = years) if oce is not None else None
        ice = ice.sel(year = years) if ice is not None else None
        t = xr.Dataset(coords = {'year': atm.year})
        nan = xr.full_like(atm['toa_net'], np.nan)
        for k, v in [('toa', 'toa_net'), ('srf_net', 'srf_net'), ('atm_imb', 'atm_imb')]:
            t[k] = atm[v] if v in atm else nan
        # OIFS ocean/land split and the area ratios
        try:
            w_oce, w_land = cr.get_areas_oifs(exp, us, cart_exp = cart_exp)
            a_oce, a_land = float(np.nansum(w_oce)), float(np.nansum(w_land))
            a_nemo = float(np.nansum(cr.get_areas_nemo(exp, us, cart_exp = cart_exp) * cr.get_mask_nemo(exp, us, cart_exp = cart_exp)))
        except Exception as err:
            logging.warning(f'{exp}: areas not available ({err}): only the atmosphere terms')
            out[exp] = t
            continue
        a_earth = a_oce + a_land
        t['srf_oce'] = atm['srf_net_oce'] if 'srf_net_oce' in atm else nan
        t['srf_land'] = atm['srf_net_land'] if 'srf_net_land' in atm else nan
        t['frac_oce'] = xr.full_like(nan, a_oce / a_earth)
        t['area_oce'], t['area_land'], t['area_nemo'] = (xr.full_like(nan, a / 1e12) for a in (a_oce, a_land, a_nemo))
        t['area_ratio'] = xr.full_like(nan, a_oce / a_nemo)
        t['srf_oce_glob'] = atm['srf_net_oce'] * a_oce / a_earth if 'srf_net_oce' in atm else nan
        t['srf_land_glob'] = atm['srf_net_land'] * a_land / a_earth if 'srf_net_land' in atm else nan
        t['F_oifs'] = atm['srf_net_oce'] * a_oce / a_nemo if 'srf_net_oce' in atm else nan
        t['F_oifs_nosn'] = atm['srf_net_oce_nosn'] * a_oce / a_nemo if 'srf_net_oce_nosn' in atm else nan
        t['snow_melt'] = t['F_oifs'] - t['F_oifs_nosn']
        has_flux = ice is not None and all(f'{v}_N' in ice and bool(ice[f'{v}_N'].notnull().any()) for v in cr.ICE_HEAT_FLUXES)
        flux = lambda v: (ice[f'{v}_N'] + ice[f'{v}_S']) if has_flux else nan
        t['qt_atm_oi'], t['qt_oce_ai'] = flux('qt_atm_oi'), flux('qt_oce_ai')
        t['interface_imb'] = t['qt_atm_oi'] - t['F_oifs']
        t['ice_storage'] = t['qt_atm_oi'] - t['qt_oce_ai']
        # open water / ice split of what NEMO receives (qt_oce: atmosphere -> open water, iceupdate.F90)
        t['qt_ow'] = oce['qt_oce'] if oce is not None and 'qt_oce' in oce else nan
        t['qt_ai'] = t['qt_atm_oi'] - t['qt_ow']
        t['ice_to_oce'] = t['qt_oce_ai'] - t['qt_ow']
        t['F_geo'] = xr.full_like(nan, cr.geothermal_flux(exp, us, cart_exp = cart_exp))
        t['ohc_trend'] = oce['ohc_trend'] if oce is not None and 'ohc_trend' in oce else nan
        t['oce_imb'] = t['ohc_trend'] - t['qt_oce_ai'] - t['F_geo']
        w = (water or {}).get(exp)
        if w is not None:
            w = w.reindex(year = t.year)
            # sbccpl.F90: qns_tot = (received from OASIS) - calving/iceberg latent heat + qemp_oce + qemp_ice;
            # qemp holds -Lf * snowfall (melting), so OIFS's flux without snowfall melting is the counterpart
            t['qemp'] = w['qemp_oce'] + w['qemp_ice'].fillna(0.)
            t['cal_latent'] = w['hflx_cal_cea'].fillna(0.) + w['hflx_icb_cea'].fillna(0.)
            t['runoff_heat'] = w['hflx_rnf_cea']
            for v in ['hflx_evap_cea', 'hflx_rain_cea', 'hflx_snow_cea', 'qemp_oce', 'qemp_ice']:
                t[v] = w[v]
            # snowfall: over the ice it is mass added to the ice (qemp_ice), over open water NEMO melts it (in qemp_oce)
            t['snow_ow'] = w['hflx_snow_cea'] - w['qemp_ice'].fillna(0.)
            t['evap_rain_ow'] = w['qemp_oce'] - t['snow_ow']
        else:
            t['qemp'] = t['runoff_heat'] = t['cal_latent'] = nan
        t['qemp_sens'] = t['qemp'] - t['snow_melt']
        wq = (lambda v: t[v] if v in t else nan)
        t['recv_ow'] = t['qt_ow'] - wq('qemp_oce') - t['cal_latent']
        t['recv_ice'] = t['qt_ai'] - wq('qemp_ice')
        t['oasis_recv'] = t['qt_atm_oi'] - t['qemp'] - t['cal_latent']
        t['oasis_res'] = t['oasis_recv'] - t['F_oifs_nosn']
        t['nemo_added'] = t['qemp_sens']
        t['oce_res'] = t['oce_imb'] - t['runoff_heat']
        # snowfall: OIFS drops snow_melt from what OASIS carries, NEMO books hflx_snow_cea (= qemp_ice + snow_ow)
        t['snow_res'] = (w['hflx_snow_cea'] - t['snow_melt']) if w is not None else nan
        t['water_cycle'] = t['qemp_sens'] - t['snow_res'] + t['runoff_heat']          # = evap_rain_ow + runoff_heat
        t['e2e_ocean'] = t['oasis_res'] + t['snow_res'] + t['water_cycle'] + t['oce_res']
        # land: no storage; what it receives leaves with the calving (runoff mapper -> NEMO). Earth units.
        gn = a_nemo / a_earth
        t['land_res'] = t['cal_latent'] * gn - t['srf_land_glob']
        t['total_res'] = (t['ice_storage'] + t['ohc_trend'] - t['F_geo']) * gn - t['toa']
        t['heat_uptake'] = t['ice_storage'] + t['ohc_trend']
        t['end_to_end'] = t['interface_imb'] + t['oce_imb']
        # OASIS band (per NEMO ocean area): snowfall split with the OIFS ice fraction; calving fed by the land flux
        t['snow_ice_oifs'] = atm['snow_ice_oce'] * a_oce / a_nemo if 'snow_ice_oce' in atm else nan
        t['snow_ow_oifs'] = t['snow_melt'] - t['snow_ice_oifs']
        t['snow_res_ice'] = wq('qemp_ice') - t['snow_ice_oifs']
        t['snow_res_ow'] = wq('snow_ow') - t['snow_ow_oifs']
        t['land_in'] = t['srf_land_glob'] / gn
        # calving: mass conserved by the runoff mapper and OASIS, so its latent heat is the same on both sides; the
        # land residual (calving out - land flux in: storage or loss) belongs to HTESSEL
        t['band_in'] = t['F_oifs'] + t['cal_latent']
        t['band_out'] = t['qt_atm_oi'] + t['runoff_heat']
        t['band_res'] = t['band_out'] - t['band_in']
        t['band_in_std'] = t['F_oifs'] + t['land_in']
        t['band_res_std'] = t['qt_atm_oi'] - t['band_in_std']
        t['oifs_res'] = t['atm_imb'] + t['land_res']
        for k, (name, area) in TERMS.items():
            if k in t:
                t[k].attrs = dict(long_name = name, units = 'W m-2', area = area)
        out[exp] = t
    return out


# area each term is averaged over in budget_terms ('region' units), and its factor to the global mean
_AREA_FACTOR = {'Earth': lambda r: 1., '-': lambda r: 1.,
                'ocean': lambda r: r['area_nemo'] / (r['area_oce'] + r['area_land']),
                'OIFS ocean': lambda r: r['area_oce'] / (r['area_oce'] + r['area_land']),
                'OIFS land': lambda r: r['area_land'] / (r['area_oce'] + r['area_land'])}


def budget_table(terms, window = (1, 10), columns = None, units = 'global'):
    """
    Mean of each term per run over a window of years: (i0, i1) = positions in each run's series (default
    years 2-10, skipping the first year), or 'last N' (e.g. 'last 20'). Returns a DataFrame (runs x terms).
    units: 'global' (default): every term as a contribution to the global mean, W m-2 per Earth m2, so that
    all the terms can be summed; 'region': each term per unit area of its own region (atmosphere per Earth
    area, OIFS ocean / land per their own area, coupling, ice and ocean per NEMO ocean area).
    """
    columns = columns or [k for k in TERMS]
    rows = {}
    for exp, t in terms.items():
        if isinstance(window, str) and window.startswith('last'):
            sel = t.isel(year = slice(-int(window.split()[1]), None))
        else:
            sel = t.isel(year = slice(window[0], window[1]))
        row = {k: float(sel[k].mean(skipna = False)) if k in sel else np.nan for k in set(columns) | {'area_oce', 'area_land', 'area_nemo'}}
        if units == 'global':
            fac = {k: _AREA_FACTOR[TERMS[k][1]](row) for k in columns if k in TERMS}
            row = {k: (v * fac[k] if k in fac else v) for k, v in row.items()}
        elif units != 'region':
            raise ValueError("units must be 'global' or 'region'")
        rows[exp] = {k: row[k] for k in columns}
        rows[exp]['years'] = f'{int(sel.year[0])}-{int(sel.year[-1])}' if sel.sizes['year'] else ''
    df = pd.DataFrame(rows).T[['years'] + columns]
    df.attrs['units'] = units
    return df


def _badge(ax, x, y, val, scale, fs = 8.5, prefix = ''):
    """Residual badge: green |r| < 0.05, orange < 0.15, red above (W m-2); grey if not available."""
    if not np.isfinite(val):
        col, txt = '#dddddd', 'n/a'
    else:
        col = '#b7e1b0' if abs(val) < 0.05 else ('#ffd59e' if abs(val) < 0.15 else '#f4a6a6')
        txt = f'{val:+.2f}'
    ax.text(x, y, prefix + txt, ha = 'center', va = 'center', fontsize = fs, fontweight = 'bold',
            bbox = dict(boxstyle = 'round,pad=0.25', fc = col, ec = 'none'))


def plot_residuals(table, exps = None, colors = None, labels = None, cart_out = None):
    """Grouped bars of the interface residuals per run (W m-2), with the +-0.05 W m-2 band."""
    exps = exps or list(table.index)
    colors = colors or cr.get_colors(exps)
    fig, ax = plt.subplots(figsize = (11, 4.5))
    xs, wbar = np.arange(len(RESIDUALS)), 0.8 / len(exps)
    ax.axhspan(-0.05, 0.05, color = 'grey', alpha = 0.2, lw = 0)
    ax.axhline(0, color = 'k', lw = 0.6)
    for i, (exp, col) in enumerate(zip(exps, colors)):
        vals = [table.loc[exp, k] if exp in table.index else np.nan for k in RESIDUALS]
        ax.bar(xs + (i - (len(exps) - 1) / 2) * wbar, np.array(vals, dtype = float), wbar, color = col,
               label = (labels or {}).get(exp, exp))
    ax.set_xticks(xs)
    ax.set_xticklabels([RESIDUAL_LABELS.get(k, k) for k in RESIDUALS], fontsize = 8.5)
    ax.set_ylabel('W m$^{-2}$ (>0: energy created)')
    ax.legend(fontsize = 8, frameon = False, loc = 'upper left', bbox_to_anchor = (1.0, 1.0))
    ax.set_title('Residual at each interface; grey band = ±0.05 W m$^{-2}$')
    fig.tight_layout()
    if cart_out:
        fig.savefig(cart_out + f"budget_residuals_{'-'.join(exps)}.pdf", bbox_inches = 'tight')
    return fig


def plot_terms_ts(terms, variables = None, exps = None, colors = None, labels = None, rolling = None, cart_out = None):
    """Annual time series of the chosen terms (default: the residuals), one panel per term."""
    variables = variables or RESIDUALS
    exps = exps or list(terms)
    colors = colors or cr.get_colors(exps)
    ncols = 3; nrows = int(np.ceil(len(variables) / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize = (5.2 * ncols, 3.4 * nrows), squeeze = False)
    for ax in axs.ravel()[len(variables):]:
        ax.set_visible(False)
    for ax, v in zip(axs.ravel(), variables):
        for exp, col in zip(exps, colors):
            t = terms.get(exp)
            if t is None or v not in t or not np.isfinite(t[v]).any():
                continue
            y = t[v].rolling(year = rolling, min_periods = 1).mean() if rolling else t[v]
            ax.plot(y.year, y, color = col, lw = 1, label = (labels or {}).get(exp, exp))
        ax.axhline(0, color = 'k', lw = 0.5)
        ax.set_title(TERMS.get(v, (v,))[0], fontsize = 9)
        ax.grid(alpha = 0.3)
    axs.ravel()[0].legend(fontsize = 7, frameon = False)
    fig.tight_layout()
    if cart_out:
        fig.savefig(cart_out + f"budget_ts_{'-'.join(exps)}.pdf", bbox_inches = 'tight')
    return fig


# ------------------------------------------------------------------------------------------------
# detailed flow diagram: rows = runs, columns = stages, energy entering from the left
# ------------------------------------------------------------------------------------------------

POS, NEG, MISSING = '#e34948', '#2a78d6', '#9a9a96'   # diverging pair of the reference palette

def _sign_color(val):
    return MISSING if not np.isfinite(val) else (POS if val > 0 else NEG)


def _val(ax, x, y, text, val, fs, **kw):
    """Label of a flux, in the colour of its arrow (red > 0, blue < 0, grey if missing)."""
    ax.text(x, y, text, fontsize = fs, color = _sign_color(val), **kw)


def _path(ax, pts, val, lw = 2.2, reverse = False):
    """Orthogonal connector through the points pts, arrow head at the end, coloured by the sign of val."""
    if reverse and np.isfinite(val) and val < 0:
        pts = pts[::-1]
    col = _sign_color(val)
    ls = '-' if np.isfinite(val) else '--'
    xs, ys = zip(*pts)
    ax.plot(xs[:-1] + (xs[-1],), ys[:-1] + (ys[-1],), color = col, lw = lw, ls = ls, solid_capstyle = 'butt', zorder = 2)
    ax.annotate('', xy = pts[-1], xytext = pts[-2], zorder = 3,
                arrowprops = dict(arrowstyle = '-|>,head_width=0.35,head_length=0.7', color = col, lw = lw, shrinkA = 0, shrinkB = 0))


def _gbox(ax, cx, cy, w, h, title, fc, fs, ec = '#6b6b67', lw = 1.0):
    """Box centred on (cx, cy) with its title at the top."""
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle = 'round,pad=0.0,rounding_size=0.12',
                                fc = fc, ec = ec, lw = lw, zorder = 1))
    if title:
        ax.text(cx, cy + h / 2 - 0.18, title, ha = 'center', va = 'top', fontsize = fs + 1, fontweight = 'bold', color = '#0b0b0b', zorder = 4)


def _display_values(r, split, nd = 2):
    """
    Values for the diagram, rounded so that every box adds up in the digits shown: only the fluxes are rounded (to nd
    decimals); residuals, sums and storages are recomputed from the rounded fluxes with the same formulas, and where a
    flux splits one part is derived from the rounded total (snow on open water, heat to open water). Each residual
    can thus differ from its exact value (budget_table) by about 10**-nd.
    """
    d = {k: r[k] for k in r.index}
    g = lambda k: float(r[k]) if k in r.index and np.isfinite(r[k]) else np.nan
    rd = lambda v: float(np.round(v, nd)) + 0. if np.isfinite(v) else np.nan
    for k in ['toa', 'F_oifs', 'land_in', 'F_oifs_nosn', 'snow_ice_oifs', 'cal_latent', 'oasis_recv', 'recv_ice', 'qemp_ice',
              'snow_ow', 'evap_rain_ow', 'runoff_heat', 'ice_to_oce', 'ohc_trend', 'F_geo', 'qt_ai', 'qt_ow']:
        d[k] = rd(g(k))
    d['snow_melt'] = d['F_oifs'] - d['F_oifs_nosn']
    d['snow_ow_oifs'] = d['snow_melt'] - d['snow_ice_oifs']
    d['snow_res_ice'] = d['qemp_ice'] - d['snow_ice_oifs']
    d['snow_res_ow'] = d['snow_ow'] - d['snow_ow_oifs']
    d['srf_net'] = d['F_oifs'] + d['land_in']
    d['atm_imb'] = d['srf_net'] - d['toa']
    d['land_res'] = d['cal_latent'] - d['land_in']
    d['oifs_res'] = d['atm_imb'] + d['land_res']
    if split:
        d['recv_ow'] = d['oasis_recv'] - d['recv_ice']
        d['oasis_res'] = d['oasis_recv'] - d['F_oifs_nosn']
        d['qt_ai'] = d['recv_ice'] + d['qemp_ice']
        d['qt_ow'] = d['recv_ow'] + d['snow_ow'] + d['evap_rain_ow'] + d['cal_latent']
        d['band_in'] = d['F_oifs'] + d['cal_latent']
        d['band_out'] = d['qt_ai'] + d['qt_ow'] + d['runoff_heat']
        d['band_res'] = d['band_out'] - d['band_in']
        oce_in = d['qt_ow'] + d['runoff_heat']
    else:
        d['qt_atm_oi'] = d['qt_ai'] + d['qt_ow']
        d['band_in_std'] = d['F_oifs'] + d['land_in']
        d['band_res_std'] = d['qt_atm_oi'] - d['band_in_std']
        oce_in = d['qt_ow']
    d['ice_storage'] = d['qt_ai'] - d['ice_to_oce']
    d['heat_uptake'] = d['ice_storage'] + d['ohc_trend']
    d['oce_res' if split else 'oce_imb'] = d['heat_uptake'] - d['qt_ai'] - oce_in - d['F_geo']
    d['total_res'] = d['heat_uptake'] - d['toa'] - d['F_geo']
    for k in ['snow_melt', 'snow_ow_oifs', 'snow_res_ice', 'snow_res_ow', 'srf_net', 'atm_imb', 'land_res', 'oifs_res', 'recv_ow',
              'oasis_res', 'qt_ai', 'qt_ow', 'band_in', 'band_out', 'band_res', 'qt_atm_oi', 'band_in_std', 'band_res_std',
              'ice_storage', 'heat_uptake', 'oce_res', 'oce_imb', 'total_res']:
        if k in d:
            d[k] = rd(float(d[k]))
    return pd.Series(d)


def plot_budget_flow(table, exps = None, labels = None, cart_out = None, fs = 13):
    """
    Energy flow through the model, one row per run (boxes on a regular grid, orthogonal connectors, same arrow
    width, red > 0 / blue < 0 with the values in the same colour).
    table: budget_table(...). With units = 'global' (default) every number is a contribution to the global mean
    (W m-2 per Earth m2) and they all add up; with units = 'region' each term is per unit area of its region.
      OIFS atmosphere: TOA in, surface flux out (with snowfall melting): over the ocean points to OASIS, over the
        land to HTESSEL; residual atm_imb
      HTESSEL land: no storage, its flux leaves as calving (runoff mapper)
      OASIS band: every flux crosses it with what OIFS gives (in) and what NEMO / SI3 books (out), and the mismatch
        (out - in) in between: heat without snowfall, snow on ice and on open water (OIFS ice fraction), evaporation
        + rain and runoff (water, mass only: OIFS gives no heat), calving (in: the land flux). Sums under the band
      NEMO / SI3 frame: what reaches the sea ice and the ocean is summed before entering each box; each box shows
        its storage (SI3: sea-ice storage, NEMO: OHC tendency), the frame the uptake and the residual oce_res
      Total = OIFS + OASIS + NEMO / SI3
    Runs without the water heat terms show only the band sum, with the runoff heat in the NEMO / SI3 residual ('*').
    Formulas: ENERGY_BUDGET.md.
    """
    exps = exps or list(table.index)
    n = len(exps)
    glob = table.attrs.get('units', 'global') == 'global'
    f = lambda r, k: float(r[k]) if k in r and np.isfinite(r[k]) else np.nan
    s = lambda v: f'{v:+.2f}' if np.isfinite(v) else 'n/a'
    D = '†'                                                     # not an output field: obtained by difference
    ink, ink2, purple, warn, grey = '#0b0b0b', '#52514e', '#6250d6', '#b03030', '#9a9a96'
    # box fill = physical domain (the arrows carry the sign: red > 0, blue < 0)
    C_ATM, C_OCE, C_LAND, C_ICE, C_OASIS, C_ADD = '#fbf0c4', '#d3e5f5', '#d9eed2', '#f1f6fa', '#e6e6e4', '#efebfa'
    X1, XL, XB, XF, X5 = 3.6, 3.9, 11.0, 17.0, 21.0           # column centres (XL: land, XB: OASIS band, XF: NEMO / SI3)
    OL, OR = 2.05, 5.8                                         # OIFS / HTESSEL frame
    BW = 3.2; BL, BR = XB - BW / 2, XB + BW / 2                # band edges
    XI, XO, XN = XB - 1.0, XB, XB + 1.0                        # in, mismatch, out columns inside the band
    FL, FR, FB, FT = 15.3, 18.95, 1.25, 9.75                    # NEMO / SI3 frame
    Y = 7.0                                                    # chain (atmosphere, heat row)
    YI, HI = 8.05, 1.3                                         # SI3 box centre and height
    YW, HWB = 4.25, 2.9                                        # NEMO box centre and height
    BXL = XF - 1.45                                            # left edge of the inner boxes
    xsp = 7.2                                                  # split of the OIFS ocean flux: heat and snowfall
    xk, xc = BR + 0.35, BR + 1.25                              # heat split, collectors (sums before the boxes)
    YIN, YOUT = YI, 3.8                                        # merged arrows into SI3 and NEMO
    mass = dict(color = grey, lw = 1.3, ls = (0, (4, 3)), zorder = 1)
    marrow = dict(arrowstyle = '-|>', color = grey, lw = 1.3)
    vfs, lfs, sfs, bfs = fs, fs - 2, fs - 3, fs + 0.5          # values, labels, notes, residual badges
    E = 'ε '                                                   # residual symbol (> 0: energy created)
    rowh = 5.2
    H = rowh * n + 1.55
    fig, axs = plt.subplots(n, 1, figsize = (22, H), squeeze = False)
    fig.subplots_adjust(left = 0.005, right = 0.995, top = 1 - 1.55 / H, bottom = 0.45 / H, hspace = 0.04)
    sub = ['global mean'] * 4 if glob else ['per Earth area', 'per NEMO ocean area', 'per NEMO ocean area', 'global']
    heads = [((OL + OR) / 2, 'Atmosphere + land'), (XB, 'Coupler'), (XF, 'Sea ice + ocean'), (X5, 'Total')]
    for i, exp in enumerate(exps):
        ax = axs[i, 0]
        ax.set_xlim(0, 22.6); ax.set_ylim(-0.35, 10.2); ax.axis('off')
        split = np.isfinite(f(table.loc[exp], 'oasis_res'))
        # global units: rounded so that every box adds up in the digits shown; region units: exact values
        r = _display_values(table.loc[exp], split) if glob else table.loc[exp]
        star = '' if split else '*'
        fo = f(r, 'frac_oce'); pc = (lambda v: f'{100 * v:.1f}%') if np.isfinite(fo) else (lambda v: '?')
        if i == 0:
            for (x, h1), h2 in zip(heads, sub):
                ax.text(x, 10.75, h1, ha = 'center', va = 'bottom', fontsize = fs + 2, fontweight = 'bold', color = ink, clip_on = False)
                ax.text(x, 10.7, h2, ha = 'center', va = 'top', fontsize = lfs, color = ink2, clip_on = False)
        if i > 0:
            ax.axhline(10.15, color = '#d8d7d2', lw = 0.8)
        head, _, rest = ((labels or {}).get(exp, exp)).partition(': ')
        ax.text(0.15, 9.6, head, fontsize = fs + 5, fontweight = 'bold', color = ink, va = 'center')
        ax.text(0.15, 9.1, textwrap.fill(rest, 16), fontsize = lfs, color = ink2, va = 'top')
        ax.text(0.15, 0.6, f'years {r["years"]}', fontsize = lfs, color = ink2)
        # ---- OIFS / HTESSEL: one frame, atmosphere and land with their residuals
        ax.add_patch(FancyBboxPatch((OL, FB), OR - OL, FT - FB, boxstyle = 'round,pad=0.0,rounding_size=0.18',
                                    fc = '#fdfbf3', ec = '#6b6b67', lw = 1.2, zorder = 0.5))
        ax.text(OL + 0.2, FT - 0.3, 'OIFS / HTESSEL', ha = 'left', va = 'center', fontsize = fs, fontweight = 'bold', color = ink)
        _path(ax, [(0.85, Y), (X1 - 1.3, Y)], f(r, 'toa'))             # TOA, from the left
        _val(ax, (0.85 + OL) / 2, Y - 0.15, f'TOA\n{s(f(r, "toa"))}', f(r, 'toa'), vfs, ha = 'center', va = 'top')
        _gbox(ax, X1, Y, 2.6, 2.4, 'OIFS atmosphere', C_ATM, fs)
        _badge(ax, X1, Y + 0.05, f(r, 'atm_imb'), 1, bfs, E)
        ax.text(X1, Y - 0.35, '= surface − TOA', ha = 'center', va = 'top', fontsize = sfs, color = ink2)
        _path(ax, [(X1 + 1.3, Y), (xsp, Y)], f(r, 'F_oifs'))
        _val(ax, (OR + xsp) / 2, Y + 0.15, f'ocean {s(f(r, "F_oifs"))}', f(r, 'F_oifs'), vfs, ha = 'center', va = 'bottom')
        ax.text((OR + xsp) / 2, Y - 0.12, f'{pc(fo)} of Earth', ha = 'center', va = 'top', fontsize = sfs - 0.5, color = ink2)
        xld = X1 - 0.4
        _path(ax, [(xld, Y - 1.2), (xld, 3.8)], f(r, 'land_in'))
        _val(ax, xld + 0.12, 4.85, f'land {s(f(r, "land_in"))}', f(r, 'land_in'), vfs, ha = 'left', va = 'center')
        ax.text(xld + 0.12, 4.5, f'{pc(1 - fo)} of Earth', ha = 'left', va = 'center', fontsize = sfs - 0.5, color = ink2)
        _gbox(ax, XL, 2.6, 3.0, 2.4, 'HTESSEL land', C_LAND, fs)
        _badge(ax, XL, 2.88, f(r, 'land_res'), 1, bfs, E)
        ax.text(XL, 2.5, '= calving − land flux\nloss (storage ≈ 0)', ha = 'center', va = 'top', fontsize = sfs, color = ink2)
        # ---- OASIS band
        ax.add_patch(FancyBboxPatch((BL, 1.6), BW, 8.3, boxstyle = 'round,pad=0.0,rounding_size=0.15',
                                    fc = C_OASIS, ec = '#6b6b67', lw = 1.0, zorder = 2.5))
        ax.text(XB, 9.6, 'OASIS', ha = 'center', va = 'center', fontsize = fs, fontweight = 'bold', color = ink, zorder = 4)
        for x, t in [(XI, 'OIFS in'), (XO, 'ε = out − in'), (XN, 'NEMO out')]:
            ax.text(x, 9.18, t, ha = 'center', va = 'center', fontsize = sfs, color = ink2, zorder = 4)
        rows = [  # name, y, in, out, mismatch, mass only
            ('snow on ice', 8.6, 'snow_ice_oifs', 'qemp_ice', 'snow_res_ice', False),
            ('heat (no snowfall)', Y, 'F_oifs_nosn', 'oasis_recv', 'oasis_res', False),
            ('snow on open water', 5.4, 'snow_ow_oifs', 'snow_ow', 'snow_res_ow', False),
            ('evaporation + rain ᵐ', 4.4, None, 'evap_rain_ow', 'evap_rain_ow', True),
            ('calving', 3.2, 'cal_latent', 'cal_latent', None, False),
            ('runoff ᵐ', 2.1, None, 'runoff_heat', 'runoff_heat', True)]
        for name, yy, kin, kout, kres, mo in rows:
            vin, vout = (0. if kin is None else f(r, kin)), f(r, kout)
            ax.text(BL - 0.1, yy + 0.12, name, ha = 'right', va = 'bottom', fontsize = lfs, color = ink2)
            if mo:
                ax.text(XI, yy, '0 (mass)', ha = 'center', va = 'center', fontsize = sfs, color = grey, zorder = 4)
            else:
                _val(ax, XI, yy, s(vin) + (D if kin == 'snow_ow_oifs' else ''), vin, vfs, ha = 'center', va = 'center', zorder = 4)
            dag = D if kout in ('oasis_recv', 'snow_ow', 'evap_rain_ow') and np.isfinite(vout) else ''
            _val(ax, XN, yy, s(vout) + dag, vout, vfs, ha = 'center', va = 'center', zorder = 4)
            res = np.nan if kres is None else (f(r, kres) if split or kres in ('snow_res_ice', 'snow_res_ow') else np.nan)
            if kres is None:
                ax.text(XO, yy, '≡ 0', ha = 'center', va = 'center', fontsize = sfs, color = grey, zorder = 4)
                ax.text(XO, yy - 0.3, 'assumed', ha = 'center', va = 'center', fontsize = sfs - 1.5, color = grey, zorder = 4)
            elif np.isfinite(res):
                fc = ('#e4e1f3' if mo else '#b7e1b0' if abs(res) < 0.05 else ('#ffd59e' if abs(res) < 0.15 else '#f4a6a6'))
                ax.text(XO, yy, f'{res:+.2f}', ha = 'center', va = 'center', fontsize = vfs - 0.5, fontweight = 'bold', zorder = 4,
                        bbox = dict(boxstyle = 'round,pad=0.2', fc = fc, ec = 'none'))
            else:
                ax.text(XO, yy, '·', ha = 'center', va = 'center', fontsize = fs, color = grey, zorder = 4)
        # into the band (OIFS side)
        _path(ax, [(xsp, Y), (xsp, 8.6), (BL, 8.6)], f(r, 'snow_ice_oifs'))
        _path(ax, [(xsp, Y), (BL, Y)], f(r, 'F_oifs_nosn'))
        _path(ax, [(xsp, Y), (xsp, 5.4), (BL, 5.4)], f(r, 'snow_ow_oifs'))
        _val(ax, xsp + 0.12, 7.8, f'snowfall\n{s(f(r, "snow_melt"))}{D}', f(r, 'snow_melt'), lfs, ha = 'left', va = 'center')
        _path(ax, [(XL + 1.5, 3.2), (BL, 3.2)], f(r, 'cal_latent'))
        xer = X1 + 1.75                                                 # evaporation + rain: water from the atmosphere
        ax.plot([X1 + 1.3, xer, xer, BL - 0.25], [Y - 0.7, Y - 0.7, 4.4, 4.4], **mass)
        ax.annotate('', xy = (BL, 4.4), xytext = (BL - 0.3, 4.4), arrowprops = marrow)
        ax.plot([XL + 1.5, BL - 0.25], [2.1, 2.1], **mass)             # runoff: water from the land
        ax.annotate('', xy = (BL, 2.1), xytext = (BL - 0.3, 2.1), arrowprops = marrow)
        # out of the band (NEMO side): rows to the collectors, then one summed arrow into each box
        ow, ic = (f(r, 'recv_ow'), f(r, 'recv_ice')) if split else (np.nan, np.nan)
        line = lambda pts, v: ax.plot(*zip(*pts), color = _sign_color(v), lw = 2.2, ls = '-' if np.isfinite(v) else '--',
                                      solid_capstyle = 'butt', zorder = 2)
        line([(BR, Y), (xk, Y)], f(r, 'oasis_recv'))
        line([(xk, Y), (xk, 7.75), (xc, 7.75)], ic)                     # heat over the ice
        _val(ax, xk + 0.1, 7.38, s(ic) + (D if split else ''), ic, lfs, ha = 'left', va = 'center')
        line([(xk, Y), (xk, 5.9), (xc, 5.9)], ow)                       # heat over open water
        _val(ax, xk + 0.1, 6.45, s(ow) + (D if split else ''), ow, lfs, ha = 'left', va = 'center')
        line([(BR, 8.6), (xc, 8.6)], f(r, 'qemp_ice'))
        for yy, k in [(5.4, 'snow_ow'), (4.4, 'evap_rain_ow'), (3.2, 'cal_latent'), (2.1, 'runoff_heat')]:
            line([(BR, yy), (xc, yy)], f(r, k))
        ax.plot([xc, xc], [7.75, 8.6], color = ink2, lw = 1.6, zorder = 2.2)    # collectors
        ax.plot([xc, xc], [2.1, 5.9], color = ink2, lw = 1.6, zorder = 2.2)
        for yy in (7.75, 8.6, 5.9, 5.4, 4.4, 3.2, 2.1):
            ax.plot(xc, yy, 'o', ms = 3.5, color = ink2, zorder = 2.3)
        v_ice = f(r, 'qt_ai')
        v_oce = f(r, 'qt_ow') + f(r, 'runoff_heat') if split else f(r, 'qt_ow')
        _path(ax, [(xc, YIN), (BXL, YIN)], v_ice)
        _val(ax, (xc + FL) / 2, YIN + 0.12, f'{s(v_ice)}{D}', v_ice, vfs, ha = 'center', va = 'bottom', fontweight = 'bold')
        ax.text((xc + FL) / 2, YIN - 0.12, 'to sea ice', ha = 'center', va = 'top', fontsize = sfs, color = ink2)
        _path(ax, [(xc, YOUT), (BXL, YOUT)], v_oce)
        _val(ax, (xc + FL) / 2, YOUT + 0.12, f'{s(v_oce)}{star}', v_oce, vfs, ha = 'center', va = 'bottom', fontweight = 'bold')
        ax.text((xc + FL) / 2, YOUT - 0.12, 'to ocean', ha = 'center', va = 'top', fontsize = sfs, color = ink2)
        # sums under the band
        if glob:
            bres = f(r, 'band_res') if split else f(r, 'band_res_std')
            bout = f(r, 'band_out') if split else f(r, 'qt_atm_oi')
            bin_ = f(r, 'band_in') if split else f(r, 'band_in_std')
            ax.text(BL - 0.1, 1.1, 'sum', ha = 'right', va = 'center', fontsize = vfs, fontweight = 'bold', color = ink)
            _val(ax, XI, 1.1, s(bin_) + star, bin_, vfs, ha = 'center', va = 'center', fontweight = 'bold')
            _badge(ax, XO, 1.1, bres, 1, bfs)
            _val(ax, XN, 1.1, s(bout) + star, bout, vfs, ha = 'center', va = 'center', fontweight = 'bold')
            ax.text(XB, 0.62, 'ᵐ water, mass only: OIFS gives no heat, NEMO books c$_p$T (lilac: bookkeeping, not an error)' +
                    ('' if split else '\n* standard output: sum only, land flux as calving (incl. HTESSEL), runoff heat in NEMO / SI3'),
                    ha = 'center', va = 'top', fontsize = sfs, color = ink2 if split else warn)
        else:
            ax.text(XB, 1.1, 'sums need global units', ha = 'center', va = 'center', fontsize = sfs, color = ink2)
        # ---- NEMO / SI3: one frame, two reservoirs with their storage
        ax.add_patch(FancyBboxPatch((FL, FB), FR - FL, FT - FB, boxstyle = 'round,pad=0.0,rounding_size=0.18',
                                    fc = '#f6f9fc', ec = '#6b6b67', lw = 1.2, zorder = 0.5))
        ax.text(XF, FT - 0.3, 'NEMO / SI3', ha = 'center', va = 'center', fontsize = fs, fontweight = 'bold', color = ink)
        _gbox(ax, XF, YI, 2.9, HI, 'SI3 sea ice', C_ICE, fs)
        _val(ax, XF, YI - 0.3, f'storage {s(f(r, "ice_storage"))}{D}', f(r, 'ice_storage'), vfs, ha = 'center', va = 'center')
        _gbox(ax, XF, YW, 2.9, HWB, 'NEMO ocean', C_OCE, fs)
        _val(ax, XF, YW + 0.3, f'OHC tendency {s(f(r, "ohc_trend"))}', f(r, 'ohc_trend'), vfs, ha = 'center', va = 'center')
        xio = XF + 0.9
        _path(ax, [(xio, YI - HI / 2), (xio, YW + HWB / 2)], f(r, 'ice_to_oce'))
        _val(ax, xio - 0.12, (YI - HI / 2 + YW + HWB / 2) / 2, f'ice → ocean\n{s(f(r, "ice_to_oce"))}{D}', f(r, 'ice_to_oce'),
             lfs, ha = 'right', va = 'center')
        _val(ax, FL + 0.25, 2.38, f'heat uptake {s(f(r, "heat_uptake"))}', f(r, 'heat_uptake'), vfs, ha = 'left', va = 'center',
             fontweight = 'bold')
        ax.text(FL + 0.25, 2.03, '= storage + OHC tendency', ha = 'left', va = 'center', fontsize = sfs, color = ink2)
        _badge(ax, FL + 0.75, 1.58, f(r, 'oce_res') if split else f(r, 'oce_imb'), 1, bfs, E)
        ax.text(FL + 1.45, 1.58, '= uptake − inputs' if split else '* incl. runoff heat', ha = 'left', va = 'center',
                fontsize = sfs, color = ink2 if split else warn)
        xg = FR - 0.65
        _gbox(ax, xg - 0.1, 0.5, 1.5, 0.65, '', C_ADD, fs, ec = purple)
        ax.text(xg - 0.1, 0.5, f'geothermal {s(f(r, "F_geo"))}', ha = 'center', va = 'center', fontsize = sfs, color = ink)
        _path(ax, [(xg, 0.825), (xg, YW - HWB / 2)], f(r, 'F_geo'))
        # ---- total = OIFS + OASIS + NEMO / SI3
        if glob:
            ax.text(X5, 6.75, 'total', ha = 'center', va = 'center', fontsize = fs, fontweight = 'bold', color = ink)
            _badge(ax, X5, 6.25, f(r, 'total_res'), 1, bfs + 1, E)
            ax.text(X5, 5.8, '= uptake − TOA − geothermal', ha = 'center', va = 'center', fontsize = sfs, color = ink2)
            parts = [('OIFS / HTESSEL' + star, f(r, 'oifs_res') if split else f(r, 'atm_imb')),
                     ('OASIS' + star, f(r, 'band_res') if split else f(r, 'band_res_std')),
                     ('NEMO / SI3' + star, f(r, 'oce_res') if split else f(r, 'oce_imb'))]
            for k, (lab, v) in enumerate(parts):
                yy = 5.3 - 0.42 * k
                ax.text(X5 - 1.3, yy, ('= ε ' if k == 0 else '+ ε ') + lab, ha = 'left', va = 'center', fontsize = lfs, color = ink)
                _val(ax, X5 + 1.45, yy, s(v), v, vfs, ha = 'right', va = 'center')
    from matplotlib.patches import Patch
    handles = [plt.Line2D([], [], color = '#9a9a96', lw = 1.3, ls = (0, (4, 3)), label = 'water, mass only'),
               Patch(fc = C_ATM, ec = '#6b6b67', label = 'atmosphere'), Patch(fc = C_OCE, ec = '#6b6b67', label = 'ocean'),
               Patch(fc = C_LAND, ec = '#6b6b67', label = 'land'), Patch(fc = C_ICE, ec = '#6b6b67', label = 'sea ice'),
               Patch(fc = C_OASIS, ec = '#6b6b67', label = 'OASIS coupler'), Patch(fc = C_ADD, ec = purple, label = 'geothermal'),
               plt.Line2D([], [], color = POS, lw = 2.2, label = 'flux > 0'), plt.Line2D([], [], color = NEG, lw = 2.2, label = 'flux < 0'),
               Patch(fc = '#ffd59e', ec = 'none', label = 'ε residual (> 0: energy created)')]
    fig.legend(handles = handles, loc = 'lower center', ncol = 10, frameon = False, fontsize = fs - 1, bbox_to_anchor = (0.5, -0.005))
    units_txt = 'W m$^{-2}$, contributions to the global mean (all numbers add up)' if glob else 'W m$^{-2}$ per unit area of each region'
    fig.text(0.5, 0.999, f'Energy budget across the model components ({units_txt}). Arrows and values: red > 0, blue < 0.\n'
             'Residuals ε (> 0: energy created): green |ε| < 0.05, orange < 0.15, red above.   † not an output field (difference of '
             'other terms); the sea-ice storage closes the SI3 box by construction.' +
             ('\nFluxes rounded to 0.01; residuals, sums and storages computed from the rounded fluxes, so every box adds up '
              '(exact values: budget_table).' if glob else ''),
             ha = 'center', va = 'top', fontsize = fs - 1, color = ink2)
    if cart_out:
        fig.savefig(cart_out + f"budget_flow_{'-'.join(exps)}{'' if glob else '_region'}.pdf", bbox_inches = 'tight')
    return fig


# ------------------------------------------------------------------------------------------------
# closure checks, one per box: inputs - outputs - storage (+ residual where the box has one) = 0
# ------------------------------------------------------------------------------------------------
# (box, [(sign, term), ...], tolerance in W m-2, note). Tolerance 1e-6: identity by construction (checks that the
# code and ENERGY_BUDGET.md agree); larger: a real check between independent estimates.
CHECKS = [
    ('1 Atmosphere',           [(+1, 'toa'), (-1, 'srf_net'), (+1, 'atm_imb')], 1e-6,
     'TOA in - surface out + residual'),
    ('2 Surface split',        [(+1, 'srf_net'), (-1, 'srf_oce_glob'), (-1, 'srf_land_glob')], 5e-3,
     'global mean (cos-latitude weights) vs ocean + land shares (OASIS areas)'),
    ('2 Snowfall taken out',   [(+1, 'F_oifs'), (-1, 'F_oifs_nosn'), (-1, 'snow_melt')], 1e-6,
     'ocean flux = sent through OASIS + snowfall melting'),
    ('3 OASIS, OIFS snowfall', [(+1, 'snow_ice_oifs'), (+1, 'snow_ow_oifs'), (-1, 'snow_melt')], 1e-6,
     'OIFS snowfall melting = on its sea ice + on open water'),
    ('3 OASIS, in',            [(+1, 'band_in'), (-1, 'F_oifs_nosn'), (-1, 'snow_ice_oifs'), (-1, 'snow_ow_oifs'), (-1, 'cal_latent')], 1e-6,
     'in = heat sent + snow on ice + snow on water + calving (same on both sides); evaporation, rain, runoff: mass only'),
    ('3 OASIS, out',           [(+1, 'band_out'), (-1, 'oasis_recv'), (-1, 'qemp_ice'), (-1, 'snow_ow'), (-1, 'evap_rain_ow'),
                                (-1, 'cal_latent'), (-1, 'runoff_heat')], 1e-6,
     'out = heat received + snow on ice + snow melting + evaporation and rain + calving + runoff (NEMO)'),
    ('3 OASIS, rows',          [(+1, 'band_res'), (-1, 'oasis_res'), (-1, 'snow_res_ice'), (-1, 'snow_res_ow'), (-1, 'evap_rain_ow'),
                                (-1, 'runoff_heat')], 1e-6,
     'band residual = sum of the row mismatches (out - in)'),
    ('3 OASIS, sum',           [(+1, 'band_out'), (-1, 'band_in'), (-1, 'band_res')], 1e-6, 'out - in = residual'),
    ('4 Sea ice',              [(+1, 'recv_ice'), (+1, 'qemp_ice'), (-1, 'ice_to_oce'), (-1, 'ice_storage')], 1e-6,
     'heat received over ice + snow on ice - to the ocean - storage'),
    ('4 Sea ice, NEMO fields', [(+1, 'qt_ai'), (-1, 'ice_to_oce'), (-1, 'ice_storage')], 1e-6,
     'qt_atm_oi - qt_oce - to the ocean - storage'),
    ('3 E-P from diagnostics', [(+1, 'qemp_oce'), (+1, 'hflx_evap_cea'), (-1, 'hflx_rain_cea'), (-1, 'hflx_snow_cea'), (+1, 'qemp_ice')], 1e-3,
     'real check: qemp_oce = -evaporation + rain + snow - snow on ice (hflx_evap_cea is written positive)'),
    ('3 E-P heat box',         [(+1, 'qemp_oce'), (-1, 'evap_rain_ow'), (-1, 'snow_ow')], 1e-6,
     'E-P heat = evaporation and rain + snow melting over open water'),
    ('4 Liquid ocean',         [(+1, 'recv_ow'), (+1, 'snow_ow'), (+1, 'evap_rain_ow'), (+1, 'cal_latent'), (+1, 'runoff_heat'),
                                (+1, 'ice_to_oce'), (+1, 'F_geo'), (-1, 'ohc_trend'), (+1, 'oce_res')], 1e-6,
     'inputs - OHC tendency + residual'),
    ('5 Ice + ocean',          [(+1, 'ice_storage'), (+1, 'ohc_trend'), (-1, 'heat_uptake')], 1e-6,
     'storage + OHC tendency = uptake'),
    ('2 Snowfall, OIFS vs NEMO', [(+1, 'snow_melt'), (+1, 'snow_res'), (-1, 'qemp_ice'), (-1, 'snow_ow')], 1e-6,
     'snowfall dropped by OIFS + mismatch = snow on ice + snow melting on water booked by NEMO'),
    ('Whole chain',            [(+1, 'F_oifs'), (+1, 'interface_imb'), (+1, 'runoff_heat'), (+1, 'F_geo'), (-1, 'heat_uptake'),
                                (+1, 'oce_res')], 1e-6,
     'OIFS ocean flux + coupling residual + runoff + geothermal - uptake + residual'),
    ('Land',                   [(+1, 'srf_land_glob'), (-1, 'cal_latent'), (+1, 'land_res')], 1e-6,
     'land flux in - calving out + residual (no land storage)'),
    ('End to end',             [(+1, 'end_to_end'), (-1, 'oasis_res'), (-1, 'cal_latent'), (-1, 'snow_res'), (-1, 'water_cycle'),
                                (-1, 'oce_res')], 1e-6,
     'end to end = OASIS + calving + snowfall mismatch + water heat content + ice + ocean'),
    ('Total, standard output', [(+1, 'total_res'), (-1, 'atm_imb'), (-1, 'end_to_end'), (+1, 'srf_land_glob')], 1e-3,
     'total = atmosphere + end to end - land flux (needs only standard output; cos-latitude vs OASIS weights)'),
    ('1 OIFS / HTESSEL',       [(+1, 'oifs_res'), (-1, 'atm_imb'), (-1, 'land_res')], 1e-6, 'atmosphere + land residuals'),
    ('Total, three boxes',     [(+1, 'total_res'), (-1, 'oifs_res'), (-1, 'band_res'), (-1, 'oce_res')], 1e-3,
     'total = OIFS / HTESSEL + OASIS band + NEMO / SI3 (cos-latitude vs OASIS weights)'),
    ('Total, three boxes, standard output', [(+1, 'total_res'), (-1, 'atm_imb'), (-1, 'band_res_std'), (-1, 'oce_imb')], 1e-3,
     'as above with the runoff heat in the ice + ocean box'),
    ('Total, all residuals',   [(+1, 'total_res'), (-1, 'atm_imb'), (-1, 'land_res'), (-1, 'oasis_res'), (-1, 'snow_res'),
                                (-1, 'water_cycle'), (-1, 'oce_res')], 1e-3,
     'total = atmosphere + land + OASIS + snowfall mismatch + water heat content + ice + ocean'),
]


def budget_checks(table, exps = None):
    """
    Closure check of each box (CHECKS) for each run: the signed sum of its terms, which must be 0 within the
    tolerance. Returns a DataFrame (one row per check) with the sum per run, the tolerance, 'ok' per run
    (True / False / None if a term is missing) and the note. table: budget_table(..., units = 'global'): the
    checks that mix regions (land, total) hold only in global units.
    """
    if table.attrs.get('units', 'global') != 'global':
        raise ValueError("budget_checks needs a table in global units: budget_table(..., units = 'global')")
    exps = exps or list(table.index)
    rows = []
    for box, terms, tol, note in CHECKS:
        row = {'check': box, 'expression': ' '.join(f'{"+" if sg > 0 else "-"} {t}' for sg, t in terms).lstrip('+ '),
               'tolerance': tol}
        for e in exps:
            vals = [sg * float(table.loc[e, t]) if t in table.columns else np.nan for sg, t in terms]
            tot = float(np.sum(vals))
            row[e] = tot
            row[f'{e} ok'] = None if not np.isfinite(tot) else bool(abs(tot) <= tol)
        row['note'] = note
        rows.append(row)
    return pd.DataFrame(rows).set_index('check')
