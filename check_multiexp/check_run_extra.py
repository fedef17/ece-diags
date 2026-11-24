"""
Functions moved out of check_run.py on 2026-11-24: not used by compare_multi_exps, main() or the
notebooks (anomaly/base-state analysis, heat transport, atmospheric streamfunction, map readers,
region means, ...). Kept importable as they were; not maintained with check_run.

    import check_run_extra as cre
"""
from check_run import *   # numpy as np, xarray as xr, plt, logging, cart_exp, cart_out, datadir, time_coder, ...
from check_run import _annual, _rename_xy, _save_nc, _exp_legend
from scipy import stats

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
                    logging.debug('Vertical levels: ', v_levels)
                    levels = (v_levels[:-1] + v_levels[1:])/ 2 # this line is not working, is summing the same values with each other!!!

                    dz_mid = np.diff(v_levels)   # shape (n_lev-1,), thickness between cell centers                                  # shape (n_lev,) same as v_levels
        
                    levels = (v_levels[1:].values + v_levels[:-1].values)/2

                    l1 = np.where(levels>lev_bounds[0])[0][0]
                    l2 = np.where(levels>lev_bounds[1])[0][0]
                    #logging.info(levels[l1], levels[l2])
                    thick_weights = (area_reg * v_mask *dz_mid).transpose('depth_mid', 'y', 'x')[l1:l2]

                    total_depth = thick_weights.sum()
                    ds_time_mean[var] = (ds[var].where(mask_reg)[l1:l2] * thick_weights.values).sum() / total_depth.values
                
                else:

                    v_area = (v_mask*area_reg).sum(axis=(1,2))
                    ds_time_mean[var] = (ds[var].where(mask_reg) *area_reg*v_mask).sum(['x', 'y']).compute()
                    ds_time_mean[var] = (ds_time_mean[var]/v_area.values).compute()

    return ds_time_mean

def compute_atm_map(ds, exp, cart_out = cart_out, atmvars = 'rsut rlut rsdt tas pr rsnt rsntcs rlnt rlntcs'.split(), year_clim = None):
    logging.debug('Computing atmospheric map climatology')
    ds = _annual(ds, atmvars).compute()
    if cart_out is not None:
        _save_nc(ds, cart_out + f'map_tuning_{exp}.nc')
    return ds

def compute_atm3d_map(ds, exp, cart_out = cart_out, atmvars = 'wap'.split(), year_clim = None):
    logging.debug('Computing 3D atmospheric map climatology')
    ds = _annual(ds, atmvars).compute()
    if cart_out is not None:
        _save_nc(ds, cart_out + f'map_atm3d_tuning_{exp}.nc')
    return ds

def compute_oce_map(ds, exp, user, cart_exp = cart_exp, cart_out = cart_out, ocevars = 'tos heatc qt_oce sos'.split(), year_clim = None, grid = 'T'):
    logging.debug('Computing ocean map climatology')
    ds = _rename_xy(_annual(ds, ocevars), grid)
    if cart_out is not None:
        _save_nc(ds, cart_out + f'map_oce_tuning_{exp}.nc')
    return ds

def compute_ice_map(ds, exp, user, cart_exp = cart_exp, cart_out = cart_out, icevars = 'siconc'.split(), year_clim = None, grid = 'T'):
    logging.debug('Computing ice map climatology')
    ds = _rename_xy(_annual(ds, icevars), grid)
    if cart_out is not None:
        _save_nc(ds, cart_out + f'map_ice_tuning_{exp}.nc')
    return ds

def read_output_map(exps, user = None, read_again = [], cart_exp = cart_exp, cart_out = cart_out, atmvars = 'rsut rlut rsdt tas rlnt rlntcs rsnt rsntcs'.split(), atmvars3d='ta ua va wap cl'.split(), ocevars = 'tos qt_oce'.split(), icevars='siconc'.split(), atm_only = False, year_clim = None, density=False, density_only=False, flag_omega = False):
    """
    Reads outputs and computes global means.

    exps: list of experiment names to read
    user: list of users
    read_again: list of exps to read again (if run has proceeded)
    atm_only: compute only atm diags
    year_clim: set years for computing climatologies (if None, considers the last CLIM_YEARS = 20 years)
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
        logging.info(exp)
        if os.path.exists(cart_out + f'map_tuning_{exp}.nc') and exp not in read_again:
            logging.info('Already computed, reading clim..')
            existing_atm = xr.load_dataset(cart_out + f'map_tuning_{exp}.nc')
            atmmap_exp[exp] = existing_atm            

            existing_oce = xr.load_dataset(cart_out + f'map_oce_tuning_{exp}.nc')
            existing_vars = list(existing_oce.data_vars)
            missing_vars = [v for v in ocevars if v not in existing_vars]
            
            if len(missing_vars) == 0:
                ocemap_exp[exp] = existing_oce           
            else:
                logging.info(f'Computing missing OCE vars: {missing_vars}')
                
                ds = xr.open_mfdataset(filz_nemo[exp],chunks={'time_counter': 240})

                new_atm = compute_oce_map(ds,exp,user=user, cart_out=cart_out,ocevars=missing_vars,year_clim=year_clim)

                # merge old + new
                updated_atm = xr.merge([existing_oce, new_atm])
                _save_nc(updated_atm, cart_out + f'map_oce_tuning_{exp}.nc')

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
            logging.info('Computing clim...')

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

def plot_amoc_ts(amoc_max, exp, ylim = (5, 20), ax = None, color = None, text_xshift = 10):
    if ax is None:
        fig, ax = plt.subplots()
    
    if isinstance(amoc_max, xr.Dataset):
        amoc_max = amoc_max['msftyz']

    amoc_max = amoc_max.groupby('time_counter.year').mean()

    amoc_max.plot(ylim=ylim, label = exp_label(exp), ax = ax, color = color)
    ax.text(amoc_max.year[-1]+text_xshift, amoc_max[-1], exp, fontsize=12, ha='right', color = color)

    return ax

def plot_map_ocean(oce_clim, exps, var, ref_exp = None, vmin = None, vmax = None, xlabel = None, ylabel = None):
    """
    oce_clim is clim_all['oce_clim'] produced by read_output
    exps: list of experiments
    var: var name to plot
    ref_exp: if specified, plot differences to ref_exp

    TO BE IMPROVED: regrid and plot with cartopy

    """
    if ref_exp is not None and ref_exp not in exps:
        logging.info(f'WARNING: {ref_exp} not in exps! plotting absolute values')
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
        logging.info('NO data to plot!')
        return fig
    
    if len(ts_dataset.keys()) == 0: 
        logging.info('NO data to plot!')
        return fig
    
    ts_dataset = create_ds_exp(ts_dataset)

    if isinstance(ts_dataset, xr.Dataset):
        if vname not in ts_dataset.data_vars:
            logging.info(f'Variable {vname} cannot be found in dataset for domain {domain}. Skipping..')
            return fig

    if ref_exp is not None and ref_exp not in exps:
        logging.info(f'WARNING: {ref_exp} not in exps! plotting absolute values')
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
            y.rolling(year = rolling, min_periods = 1).mean().plot(label = exp_label(exp), color = col, ax = ax)
        else:
            y.plot(label = exp_label(exp), color = col, ax = ax)

        # ax.text(y.year[-1] + 5, np.nanmean(y.values[-30:]), exp, fontsize=12, ha='right', color = col) # not working for some evil reason
    
    ylabel = var_label(vname, ts_dataset, domain = domain)
    if norm_factor != 1.:
        ylabel += f' × {norm_factor:g}'
    if ref_exp is not None:
        ylabel = f'Δ {ylabel} (wrt {ref_exp})'
    ax.set_ylabel(ylabel)
    ax.set_title(var_label(vname, ts_dataset, units = False, domain = domain))
    _exp_legend(ax)
    name = '-'.join([exp for exp in exps])
    fig.savefig(cart_out + f'check_ts_{domain}_{vname}_{name}.pdf', bbox_inches = 'tight')
    
    return fig

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

        delta = (var.sel(exp=exps[0]) - (np.arange(0,len(var.sel(exp=exps[0]))))[:, np.newaxis, np.newaxis]*trend-intercept).rolling(year=30, min_periods = 1).mean()
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
            omega = var.sel(exp=exps[0]).rolling(year=30, min_periods = 1).mean(dim='year')[85]
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
    logging.info(f"Saved {variables} at {level} {level_units} → {output_file}")

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
        logging.info(f"Read hfbasin → {hfbasin_path}")

    except Exception as e:
        logging.info(f"Warning: could not read hfbasin: {e}")
        has_hfbasin = False

    # ── Save to netCDF ────────────────────────────────────────────────────────
    ds_out = xr.Dataset({'THT': tht, 'AHT': aht, 'OHT': oht})
    if has_hfbasin:
        ds_out['OHT_explicit'] = oht_explicit
        ds_out['OHT_atl']      = oht_atl

    ds_out.attrs['description'] = f'Meridional heat transport timeseries, exp={exp_name}'
    outfile = cart_out_nc + f'heat_transport_{exp_name}.nc'
    ds_out.to_netcdf(outfile)
    logging.info(f"Saved heat transports → {outfile}")

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
            logging.info(f'-------- {exp} --------')
        
            # test full length trend
            trend = stats.linregress(np.arange(0,np.shape(var_pi[mask])[0]), var_pi[mask])[0]
            logging.info(f'{var} trend for {exp}: {trend}')
            logging.info(f'{var} change in {np.shape(var_pi[mask])[0]} years: {trend*np.shape(var_pi[mask])[0]}\n')

            # test 500 years trend
            trend = stats.linregress(np.arange(0,499), var_pi[mask][:499])[0]
            logging.info(f'{var} trend for {exp} for first 500 ys: {trend}')
            logging.info(f'{var} change in 500 years: {trend*500}\n')

            # test 300 years trend
            trend = stats.linregress(np.arange(0,300), var_pi[mask][-300:])[0]
            logging.info(f'{var} trend for {exp} for last 300 ys: {trend}')
            logging.info(f'{var} change in 300 years: {trend*300}\n')

            # test trend in 300 ys before 500
            trend = stats.linregress(np.arange(0,299), var_pi[mask][200:499])[0]
            logging.info(f'{var} trend for {exp} for 200-500 ys: {trend}')
            logging.info(f'{var} change in 300 years: {trend*300}\n')

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
