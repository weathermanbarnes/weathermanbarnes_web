# app.py
# This file contains a Flask application to trigger your plotting script,
# now configured to load ECMWF GRIB2 data directly from a Google Cloud Storage bucket.

import os
import uuid
import sys
import tempfile # For creating temporary files
import shutil   # For cleaning up temporary directories
import io
from flask import Flask, request, jsonify
from google.cloud import storage # Still needed for other GCS operations like uploading plots

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr # For handling scientific data
#import cfgrib # Required for opening GRIB files with xarray
from datetime import datetime
from dateutil.relativedelta import relativedelta

from metpy.calc import equivalent_potential_temperature,dewpoint_from_relative_humidity,potential_vorticity_baroclinic
from metpy.calc import wind_speed,potential_temperature,isentropic_interpolation_as_dataset
from metpy.calc import q_vector,lat_lon_grid_deltas,divergence,smooth_n_point
from metpy.calc import vorticity
from metpy.units import units

#sys.path.append('weathermanbarnes_web/Forecasts/ops_scripts')
from weathermanbarnes_web.Forecasts.ops_scripts.plot_map_functions import *

# Corrected Flask initialization: use __name__ instead of __init__
app = Flask(__name__)

# Initialize Google Cloud Storage client
storage_client = storage.Client()

def delete_directory(path):
    """
    Deletes a directory and all of its contents.
    
    Args:
        path (str): The path to the directory to be deleted.
    """
    if os.path.exists(path):
        try:
            shutil.rmtree(path)
            print(f"Successfully deleted the directory: {path}")
        except OSError as e:
            print(f"Error: {path} : {e.strerror}")
    else:
        print(f"The directory {path} does not exist.")

def replace_phrase_in_file(input_file_path, output_file_path, old_phrase, new_phrase):
    try:
        # Open the input file in read mode and read its content
        with open(input_file_path, 'r') as file:
            file_content = file.read()
        
        # Replace the old phrase with the new phrase
        updated_content = file_content.replace(old_phrase, new_phrase)
        
        # Open the output file in write mode and write the updated content
        with open(output_file_path, 'w') as file:
            file.write(updated_content)
        
        #print("Replacement completed successfully. Updated file saved as", output_file_path)
    
    except Exception as e:
        print(f"An error occurred in replacephrase in file function: {e}")

# Create a single directory
def create_directory(dir_name):
    try:
        os.mkdir(dir_name)
        print(f"Directory '{dir_name}' created successfully.")
    except FileExistsError:
        print(f"Directory '{dir_name}' already exists.")
    except Exception as e:
        print(f"An error occurred: {e}")

def remove_temp_files(temp_dir):
    # Clean up the temporary directory and file
    if temp_dir and os.path.exists(temp_dir):
        print(f"Cleaning up temporary directory: {temp_dir}")
        shutil.rmtree(temp_dir)

def set_cache_control(bucket_name, blob_name, cache_control_value):
    """Set the Cache-Control metadata for a GCS blob."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    blob.cache_control = cache_control_value
    blob.patch()

    #print(f"Cache-Control for {blob_name} set to {cache_control_value}.")

def upload_to_bucket(bucket_name, source_file_name, destination_blob_name):
    """
    Uploads a file to the bucket.
    
    Args:
    bucket_name (str): Name of the bucket.
    source_file_name (str): Path to the file to upload.
    destination_blob_name (str): Name of the blob in the bucket.
    """
    # Initialize a storage client
    storage_client = storage.Client()
    # Get the bucket
    bucket = storage_client.bucket(bucket_name)
    # Create a blob object
    blob = bucket.blob(destination_blob_name)
    # Upload the file
    blob.upload_from_filename(source_file_name)
    
    print(f"File {source_file_name} uploaded to {destination_blob_name}.")

def save_blob_to_bucket(img_buffer,output_bucket_name,output_blob_name,cache_control='no-store'):
    # 3. Get the output bucket and upload the image
    output_bucket = storage_client.bucket(output_bucket_name)
    output_blob = output_bucket.blob(output_blob_name)

    output_blob.upload_from_file(img_buffer, content_type='image/png')
    set_cache_control(output_bucket_name, output_blob_name, 'no-store')

    print(f"Plot '{output_blob_name}' uploaded to bucket '{output_bucket_name}'.")

def load_required_ifs_data_from_gcs(bucket_name, blob_name, precip_data_blob):
    """
    Downloads an ECMWF data file (GRIB2) from GCS to a temporary local file,
    then opens it with xarray using cfgrib.

    Args:
        bucket_name (str): The name of the GCS bucket where the ECMWF data is stored.
        blob_name (str): The path/name of the data file in the bucket.

    Returns:
        xarray.Dataset: The loaded ECMWF data as an xarray Dataset.
    """
    gcs_uri = f"gs://{bucket_name}/{blob_name}"
    gcs_precip_uri = f"gs://{bucket_name}/{precip_data_blob}"
    print(f"Attempting to download GRIB2 data from: {gcs_uri} to a temporary local file.")

    # Create a temporary directory and file
    import uuid

    # Generate a unique identifier for your "temporary directory"
    temp_dir = None
    temp_file_path = None
    try:
        #temp_dir = tempfile.mkdtemp()
        temp_dir = os.path.join("/mnt/storage/tmp/",str(uuid.uuid4()))
        #temp_dir = os.path.join("/Users/mbar0087/Downloads/",str(uuid.uuid4()))
        try:
            os.makedirs(temp_dir, exist_ok=True)
            print(f"Directory created successfully at: {temp_dir}")
        except OSError as e:
            print(f"Error creating directory: {e}")
        create_directory(temp_dir)
        temp_file_path = os.path.join(temp_dir, os.path.basename(blob_name))

        # Get the blob and download it to the temporary file
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.download_to_filename(temp_file_path)
        print(f"Successfully downloaded {gcs_uri} to temporary file: {temp_file_path}")

        #temp_tp_dir = tempfile.mkdtemp()
        #temp_tp_dir = temp_dir
        temp_tp_file_path = os.path.join(temp_dir, os.path.basename(precip_data_blob))

        # Get the blob and download it to the temporary file
        bucket_tp = storage_client.bucket(bucket_name)
        blob_tp = bucket.blob(precip_data_blob)
        blob_tp.download_to_filename(temp_tp_file_path)
        print(f"Successfully downloaded {gcs_precip_uri} to temporary file: {temp_tp_file_path}")
        print(bucket_name,blob_name)

        # Open the data with xarray from the local temporary file
        #### Get the raw data
        ds_pl = xr.open_dataset(temp_file_path,engine='cfgrib', backend_kwargs={'indexpath': '', 'filter_by_keys':{ 'typeOfLevel': 'isobaricInhPa'}}, decode_timedelta=True)
        u10 = xr.open_dataset(temp_file_path,engine='cfgrib', backend_kwargs={'indexpath': '', 'filter_by_keys': {'shortName': '10u'}}, decode_timedelta=True)
        mslp = xr.open_dataset(temp_file_path,engine='cfgrib', backend_kwargs={'indexpath': '', 'filter_by_keys': {'shortName': 'msl'}}, decode_timedelta=True)
        v10 = xr.open_dataset(temp_file_path,engine='cfgrib', backend_kwargs={'indexpath': '', 'filter_by_keys': {'shortName': '10v'}}, decode_timedelta=True)
        tp1 = xr.open_dataset(temp_file_path,engine='cfgrib', backend_kwargs={'indexpath': '', 'filter_by_keys': {'shortName': 'tp'}}, decode_timedelta=True)
        tp0 = xr.open_dataset(temp_tp_file_path,engine='cfgrib', backend_kwargs={'indexpath': '', 'filter_by_keys': {'shortName': 'tp'}}, decode_timedelta=True)

        #### Configure the raw data fields for use
        mslp=(mslp['msl']/100).drop_vars(['meanSea']).rename('mslp')
        u10=u10['u10'].drop_vars(['heightAboveGround'])
        v10=v10['v10'].drop_vars(['heightAboveGround'])
        tp=np.abs((tp1-tp0)['tp']*1000).drop_vars(['surface']).rename('precip')
        t850=ds_pl['t'].sel(isobaricInhPa=850).drop_vars(['isobaricInhPa']).rename('t850')
        z500=ds_pl['gh'].sel(isobaricInhPa=500).drop_vars(['isobaricInhPa']).rename('z500')
        u500=ds_pl['u'].sel(isobaricInhPa=500).drop_vars(['isobaricInhPa']).rename('u500')
        v500=ds_pl['v'].sel(isobaricInhPa=500).drop_vars(['isobaricInhPa']).rename('v500')
        u300=ds_pl['u'].sel(isobaricInhPa=300).drop_vars(['isobaricInhPa']).rename('u300')
        v300=ds_pl['v'].sel(isobaricInhPa=300).drop_vars(['isobaricInhPa']).rename('v300')
        z700=ds_pl['gh'].sel(isobaricInhPa=700).drop_vars(['isobaricInhPa']).rename('z700')

        vort500=vorticity(ds_pl['u'].sel(isobaricInhPa=500) * units['m/s'], 
                          ds_pl['v'].sel(isobaricInhPa=500) * units['m/s']).drop_vars(['isobaricInhPa']).metpy.dequantify().rename('vort500')
        thickness=((ds_pl['gh'].sel(isobaricInhPa=500)-ds_pl['gh'].sel(isobaricInhPa=1000))/10).rename('thickness')
        
        uIVT = (z700*0 -1/9.8*np.trapz(ds_pl['q'].sel(isobaricInhPa=slice(1000,300)) * ds_pl['u'].sel(isobaricInhPa=slice(1000,300)),
                            ds_pl['q'].sel(isobaricInhPa=slice(1000,300)).isobaricInhPa*100, axis=0)).rename('uIVT')
        vIVT = (z700*0 -1/9.8*np.trapz(ds_pl['q'].sel(isobaricInhPa=slice(1000,300)) * ds_pl['v'].sel(isobaricInhPa=slice(1000,300)),
                            ds_pl['q'].sel(isobaricInhPa=slice(1000,300)).isobaricInhPa*100, axis=0)).rename('vIVT')
        IVT = (z700*0 -1/9.8*np.trapz(ds_pl['q'].sel(isobaricInhPa=slice(1000,300)) * np.sqrt(ds_pl['u'].sel(isobaricInhPa=slice(1000,300))**2 + ds_pl['v'].sel(isobaricInhPa=slice(1000,300))**2),
                            ds_pl['q'].sel(isobaricInhPa=slice(1000,300)).isobaricInhPa*100, axis=0)).rename('IVT')

        #### Calculate additional fields
        #### Upper plot
        spd300=wind_speed(u300 * units('m/s'), v300  * units('m/s')).metpy.dequantify().rename('spd300')
        ujet300=u300.where(spd300>50*0.514444).rename('ujet300')
        vjet300=v300.where(spd300>50*0.514444).rename('vjet300')
        jet300=spd300.where(spd300>50*0.514444).rename('jet300')
        wMID=ds_pl['w'].sel(isobaricInhPa=slice(700,400)).mean(dim='isobaricInhPa')
        wMID=wMID.where(wMID<0).rename('wMID')

        #### Produce the xarray dataset for plotting
        ds = xr.merge([mslp,u10,v10,t850,z500,u500,v500,spd300,ujet300,vjet300,jet300,wMID,tp,vort500,thickness,
                       z700,uIVT,vIVT,IVT])
        ds = ds.rename({'time':'analysis_time'})
        ds=add_cyclic_to_dataset(ds)

        #print(f"Successfully loaded data from local temporary file using cfgrib.")
        return ds, temp_dir
    
    except Exception as e:
        # Log the full exception details
        import traceback
        print(f"ERROR during GRIB2 data loading (download or xarray/cfgrib open): {e}")
        print("Full traceback:")
        traceback.print_exc()
        raise # Re-raise to be caught by the outer try-except in generate_ecmwf_plot_and_save

def generate_ecmwf_plot_and_save(ecmwf_data_bucket, ecmwf_data_blob, ecmwf_precip_data_blob, output_bucket_name, fignum):
    """
    Loads ECMWF data from a GCS bucket, generates a plot,
    and saves it to another Google Cloud Storage bucket.

    Args:
        ecmwf_data_bucket (str): The GCS bucket containing the ECMWF data.
        ecmwf_data_blob (str): The blob name of the ECMWF data file.
        output_bucket_name (str): The name of your GCS bucket for saving plots.
        fignum (int): The figure number.
    """
    try:
        # 1. Load ECMWF data from GCS
        data_to_plot, temp_dir = load_required_ifs_data_from_gcs(ecmwf_data_bucket, ecmwf_data_blob, ecmwf_precip_data_blob)

        #names=['SH','Australia','SouthernAfrica','SouthAmerica','IndianOcean','NH','NorthAmerica','Europe','NorthAtlantic','Asia']
        names=['SH','Australia','NH']

        for name in names:

            img_buffer, figname, plottype = plot_IVT(io.BytesIO(), fignum, data_to_plot, name=name, model_name='ECMWF-IFS', save_type='GCS', dpi=300)
            bucket_dir=f'data/ECMWF/IFS/{name}/{plottype}/'
            save_blob_to_bucket(img_buffer,output_bucket_name,f'{bucket_dir}{figname}')#; plt.close(fig)

            img_buffer, figname, plottype = plot_thickness(io.BytesIO(), fignum, data_to_plot, name=name, model_name='ECMWF-IFS', save_type='GCS', dpi=300)
            bucket_dir=f'data/ECMWF/IFS/{name}/{plottype}/'
            save_blob_to_bucket(img_buffer,output_bucket_name,f'{bucket_dir}{figname}')#; plt.close(fig)

            img_buffer, figname, plottype = plot_upper(io.BytesIO(), fignum, data_to_plot, name=name, model_name='ECMWF-IFS', save_type='GCS', dpi=300)
            bucket_dir=f'data/ECMWF/IFS/{name}/{plottype}/'
            save_blob_to_bucket(img_buffer,output_bucket_name,f'{bucket_dir}{figname}')#; plt.close(fig)

            img_buffer, figname, plottype = plot_precip6h(io.BytesIO(), fignum, data_to_plot, name=name, model_name='ECMWF-IFS', save_type='GCS', dpi=300)
            bucket_dir=f'data/ECMWF/IFS/{name}/{plottype}/'
            save_blob_to_bucket(img_buffer,output_bucket_name,f'{bucket_dir}{figname}')#; plt.close(fig)

        #for temp_dir in temp_dirs:
        delete_directory(temp_dir)
        
        return True, f"gs://{output_bucket_name}/{figname}"
    
    except Exception as e:
        print(f"Error generating or uploading plot: {e}")
        return False, str(e)

@app.route('/generate-plot', methods=['POST'])
def handle_generate_plot():
    """
    HTTP endpoint to trigger plot generation and upload.
    Expects a JSON payload with:
    - 'ecmwf_data_bucket': GCS bucket for ECMWF data (e.g., 'ecmwf-open-data')
    - 'ecmwf_data_blob': Blob name for ECMWF data file (GRIB2 format)
    - 'output_bucket_name': GCS bucket for saving plots
    - 'fignum': fignum
    """
    data = request.get_json()
    if not data or 'init_date' not in data or 'ecmwf_data_bucket' not in data or 'ecmwf_data_blob' not in data or 'ecmwf_precip_data_blob' not in data \
                 or 'output_bucket_name' not in data or 'fignum' not in data:
        return jsonify({
            "error": "Missing one or more required parameters: 'ecmwf_data_bucket', 'ecmwf_data_blob', 'ecmwf_precip_data_blob', 'output_bucket_name', or 'fignum'."
        }), 400

    init_date = data['init_date']
    ecmwf_data_bucket = data['ecmwf_data_bucket']
    ecmwf_data_blob = data['ecmwf_data_blob']
    ecmwf_precip_data_blob = data['ecmwf_precip_data_blob']
    output_bucket_name = data['output_bucket_name']
    fignum = data['fignum']

    ################### Add the new html file to the website  ###################
    input_html_path = 'weathermanbarnes_web/default_web/'+'forecast_IFS_default.html' 
    output_html_file = 'forecast_IFS.html'
    output_html_path = 'weathermanbarnes_web/default_web/'+output_html_file
    init_date_replace_phrase = 'DREFDREFDREF'
    replace_phrase_in_file(input_html_path, output_html_path, init_date_replace_phrase, init_date)
    upload_to_bucket(output_bucket_name, output_html_path, output_html_file)

    ################### Do the plotting and add it to the website ###################
    success, result_message = generate_ecmwf_plot_and_save(
        ecmwf_data_bucket, ecmwf_data_blob, ecmwf_precip_data_blob, output_bucket_name, fignum
    )

    if success:
        return jsonify({
            "message": "Plot generated and uploaded successfully.",
            "gcs_path": result_message
        }), 200
    else:
        return jsonify({
            "error": "Failed to generate or upload plot.",
            "details": result_message
        }), 500

@app.route('/')
def health_check():
    return "Service is running!", 200

if __name__ == '__main__':
    # This block is for local testing. Cloud Run will use gunicorn.
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))

