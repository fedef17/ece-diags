#!/usr/bin/env python3
"""
AMOC Diagnostic Tool

This tool analyzes Atlantic Meridional Overturning Circulation (AMOC) data
from ocean model output files and generates diagnostic plots including
mean AMOC sections and timeseries.

Author: AMOC Analysis Team
"""

import argparse
import os
import sys
from glob import glob
from pathlib import Path
import yaml
import gsw_xarray as gsw
import xarray as xr
from matplotlib import pyplot as plt
import numpy as np


def load_config(config_path: str = None) -> dict:
    """
    Load configuration from YAML file.
    
    Parameters
    ----------
    config_path : str, optional
        Path to the configuration file. If None, looks for config.yaml
        in the same directory as this script.
        
    Returns
    -------
    dict
        Configuration dictionary with default values
        
    Raises
    ------
    FileNotFoundError
        If the configuration file is not found
    yaml.YAMLError
        If the YAML file is malformed
    """
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing YAML configuration: {e}")


def find_oce_files(experiment: str, input_dir: str, config: dict) -> list:
    """
    Find ocean files for the given experiment.
    
    Parameters
    ----------
    experiment : str
        Experiment name/identifier
    input_dir : str
        Base input directory containing experiment outputs
    config : dict
        Configuration dictionary containing file patterns
        
    Returns
    -------
    list
        Sorted list of ocean file paths
        
    Raises
    ------
    FileNotFoundError
        If no ocean files are found for the experiment
    """
    # Construct the full path to NEMO output directory
    nemo_dir = os.path.join(input_dir, experiment, 'output', 'nemo')
    
    # Use file pattern from config
    file_pattern = config['file_patterns']['base'].format(exp=experiment)
    full_pattern = os.path.join(nemo_dir, file_pattern)
    
    files = sorted(glob(full_pattern))
    
    if not files:
        raise FileNotFoundError(
            f"No ocean files found for experiment '{experiment}' "
            f"in directory '{nemo_dir}' with pattern '{file_pattern}'"
        )
    
    return files

def compute_density(experiment: str, files:list, output_dir: str):
    time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
    oce_data = xr.open_mfdataset(files, decode_times=time_coder).groupby("time_counter.year").mean()

    pressure = gsw.p_from_z(-oce_data.deptht, oce_data.nav_lat)

    # compute density for each year
        
    for year in oce_data.year.values:
        print('Processing year '+str(year))
        salinity = oce_data.so.sel(year=year)
        temperature = oce_data.thetao.sel(year=year)
        density = gsw.rho_t_exact(salinity, temperature, pressure)

        con_temp = gsw.CT_from_t(salinity, temperature, pressure)
        Nsquared, p_mid = gsw.Nsquared(salinity, con_temp, pressure, axis=0)

        ds_year = xr.Dataset(
            {
                "density": density.expand_dims(year=[year]),
                "Nsquared": xr.DataArray(
                    Nsquared,
                    dims=("depth_mid", "y", "x"),
                    coords={
                        "depth_mid": np.arange(Nsquared.shape[0]),
                        "nav_lon": oce_data.nav_lon,  # grid_T in version 4.1.0
                        "nav_lat": oce_data.nav_lat,
                    }
                ).expand_dims(year=[year]),
                "p_mid": (("depth_mid", "y", "x"), p_mid)
            }
        )

        out_file = experiment+'_'+str(year)+'_density.nc'
        out_path = os.path.join(output_dir, out_file)
        ds_year.to_netcdf(out_path, mode="w")
    # concatenate all years along year dim


    print('Computation of density and N2 successful!')



def amoc_timeseries(experiment: str, files: list, config: dict) -> plt.Figure:
    """
    Create AMOC maximum timeseries plot.
    
    This function generates a timeseries of the maximum AMOC strength
    in the Atlantic basin, computed over specified depth and latitude ranges.
    
    Parameters
    ----------
    experiment : str
        Experiment name for plot title
    files : list
        List of AMOC diagnostic NetCDF files
    config : dict
        Configuration dictionary containing plot settings
        
    Returns
    -------
    matplotlib.pyplot.Figure
        The generated figure object
        
    Raises
    ------
    ValueError
        If the data cannot be processed or plotted
    """
    plot_config = config['plot_settings']
    timeseries_config = plot_config['timeseries_plot']
    
    # Create figure with specified size
    fig = plt.figure(figsize=(plot_config['figsize']['width'], 
                             plot_config['figsize']['height']))
    
    try:
        # Load data (exclude last incomplete file)
        data = xr.open_mfdataset(files[:-1])
        
        # Select depth range and Atlantic basin
        depth_min, depth_max = timeseries_config['depth_range']
        lat_min, lat_max = timeseries_config['lat_range']
        
        amoc = data.sel(
            depthw=slice(depth_min, depth_max), 
            basin=timeseries_config['basin']
        )['msftyz']
        
        # Apply latitude constraint and compute
        amoc = amoc.where(
            (data['nav_lat'] > lat_min) & (data['nav_lat'] < lat_max)
        ).compute()
        
        # Resample to yearly means and find maximum
        amoc_yearly = amoc.resample(time_counter='YS').mean()
        amoc_max = amoc_yearly.max(dim=['depthw', 'y'])
        
        # Plot timeseries
        amoc_max.plot(ylim=timeseries_config['ylim'])
        plt.title(f'AMOC max {experiment} (Sv) timeseries')
        plt.xlabel('Year')
        plt.ylabel('AMOC Strength (Sv)')
        
    except Exception as e:
        raise ValueError(f"Error creating AMOC timeseries plot: {e}")
    
    return fig


def save_figure(fig: plt.Figure, filename: str, output_dir: str) -> None:
    """
    Save figure in both PNG and PDF formats in separate subdirectories.
    
    Parameters
    ----------
    fig : matplotlib.pyplot.Figure
        The figure object to save
    filename : str
        Base filename without extension
    output_dir : str
        Base output directory where png/ and pdf/ subdirectories will be created
    """
    # Create subdirectories for PNG and PDF
    png_dir = os.path.join(output_dir, 'png')
    pdf_dir = os.path.join(output_dir, 'pdf')
    
    os.makedirs(png_dir, exist_ok=True)
    os.makedirs(pdf_dir, exist_ok=True)
    
    # Save in PNG format
    png_path = os.path.join(png_dir, f'{filename}.png')
    fig.savefig(png_path, dpi=300, bbox_inches='tight', format='png')
    print(f"PNG saved to: {png_path}")
    
    # Save in PDF format
    pdf_path = os.path.join(pdf_dir, f'{filename}.pdf')
    fig.savefig(pdf_path, bbox_inches='tight', format='pdf')
    print(f"PDF saved to: {pdf_path}")


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(
        description='AMOC Diagnostic Tool - Analyze Atlantic Meridional Overturning Circulation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        'experiment',
        type=str,
        help='Experiment name/identifier (required)'
    )
    
    parser.add_argument(
        '-i', '--input-dir',
        type=str,
        default=None,
        help='Input directory containing experiment outputs (overrides config default)'
    )
    
    parser.add_argument(
        '-o', '--output-dir', 
        type=str,
        default=None,
        help='Output directory for generated figures (overrides config default)'
    )
    
    parser.add_argument(
        '-n', '--nyear',
        type=int,
        default=None,
        help='Number of years to average for mean plots (overrides config default)'
    )
    
    parser.add_argument(
        '-c', '--config',
        type=str,
        default=None,
        help='Path to configuration YAML file'
    )
    
    parser.add_argument(
        '--section-only',
        action='store_true',
        help='Generate only the AMOC section plot'
    )
    
    parser.add_argument(
        '--timeseries-only',
        action='store_true',
        help='Generate only the AMOC timeseries plot'
    )
    
    return parser.parse_args()


def main():
    """
    Main execution function.
    
    This function orchestrates the entire density analysis workflow:
    1. Parse command-line arguments
    2. Load configuration
    3. Find input files
    4. Generate diagnostic plots
    5. Save outputs
    """
    try:
        # Parse command-line arguments
        args = parse_arguments()
        
        # Load configuration
        config = load_config(args.config)
        
        # Override config values with command-line arguments where provided
        input_dir = args.input_dir or config['input_directory']
        output_dir = args.output_dir or config['output_directory']
        nyear = args.nyear or config['nyear']
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, args.experiment), exist_ok=True)
        output_dir = os.path.join(output_dir, args.experiment)
        # Find ocean files
        print(f"Looking for ocean files for experiment '{args.experiment}'...")
        oce_files = find_oce_files(args.experiment, input_dir, config)
        print(f"Found {len(oce_files)} ocean files")
        
        print(f"Computing density for experiment '{args.experiment}'...")
        compute_density(args.experiment, oce_files, output_dir)

        # # Generate plots based on user selection

        # if not args.section_only:
        #     print("Generating AMOC timeseries plot...")
        #     fig_timeseries = amoc_timeseries(args.experiment, amoc_files, config)
        #     save_figure(fig_timeseries, f'amoc_timeseries_{args.experiment}', output_dir)
        #     plt.close(fig_timeseries)
        
        print("Density analysis completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()