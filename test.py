import xarray as xr
import gcsfs

fs = gcsfs.GCSFileSystem()

# Construct the full GCS path to the file
gcs_path = "gs://ecmwf-open-data/20240814/12z/ifs/0p25/oper/20240814120000-0h-oper-fc.grib2"

# Open the dataset directly from GCS using the gcsfs backend
with fs.open(gcs_path) as f:
    ds = xr.open_dataset(f, engine="cfgrib", backend_kwargs={'indexpath': '', 'filter_by_keys': {'shortName': '10u'}}, decode_timedelta=True)

print(ds)

# Now you can use ds for your plotting as before