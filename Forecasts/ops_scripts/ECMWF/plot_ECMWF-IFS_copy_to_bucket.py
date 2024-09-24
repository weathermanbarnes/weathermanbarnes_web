import os
import shutil
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from google.cloud import storage

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
        
        print("Replacement completed successfully. Updated file saved as", output_file_path)
    
    except Exception as e:
        print(f"An error occurred: {e}")

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

def set_cache_control(bucket_name, blob_name, cache_control_value):
    """Set the Cache-Control metadata for a GCS blob."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    blob.cache_control = cache_control_value
    blob.patch()

    print(f"Cache-Control for {blob_name} set to {cache_control_value}.")


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("date",help="init date",type=str)
parser.add_argument("run",help="init run",type=int)
parser.add_argument("set_hemisphere",help="init date",type=str)
args = parser.parse_args()
RUN = args.run
INDATEstr = args.date
set_hemisphere = args.set_hemisphere

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/565/mb0427/gadi_web_bucket_access.json"

bucket_name = "www.weathermanbarnes.com"  # Replace with your bucket name
home_path = os.getcwd()+"/"
indir=home_path+'scratch/IFS/'
webdir='data/ECMWF/IFS/'

input_html_path = home_path+'forecast_IFS_default.html'  # Replace with the path to your input file
output_html_file='forecast_IFS.html'
output_html_path = home_path+output_html_file  # Path for the updated output file

model='ECMWF-IFS'
if set_hemisphere=='SH':
    domains=['SH','Australia','SouthernAfrica','SouthAmerica','IndianOcean']
if set_hemisphere=='NH':    
    domains=['NH','NorthAmerica','Europe','NorthAtlantic','Asia']

plottypes=['UpperLevel',
            'Precip6H',
            'QvecPTE850',
            'IVT',
            'IPV-320K',
            'IPV-330K',
            'IPV-350K',
            'IrrotPV']

indatetime=datetime.strptime(INDATEstr,'%Y%m%d')
indatetime=indatetime+relativedelta(hours=RUN)

old_phrase = 'DREFDREFDREF'
new_phrase = indatetime.strftime("%Y%m%d%H")

replace_phrase_in_file(input_html_path, output_html_path, old_phrase, new_phrase)

upload_to_bucket(bucket_name, output_html_path, output_html_file)
cache_control_value = "max-age=60"
set_cache_control(bucket_name, output_html_file, cache_control_value)

for t in range(-264,6,6):#range(-264,6,6):
    tdt=indatetime+relativedelta(hours=t)
    init_date = tdt.strftime("%Y%m%d")
    init_time = tdt.strftime("%H%M%S")
    init_hour = tdt.strftime("%H")

    for type in plottypes:
        for domain in domains:
            infile=init_date+init_time+"_"+model+"_"+domain+"_"+type+"_0.jpg"
            newfile=model+"_"+domain+"_"+type+"_"+str(t)+".jpg"
            if os.path.isfile(os.path.join(indir+"analysis/",infile)):
                shutil.copyfile(os.path.join(indir+"analysis/",infile), 
                                os.path.join(indir+"images/",newfile))

files = os.listdir(indir+"images/")
for type in plottypes:
    for domain in domains:
        outdir=webdir+domain+'/'+type+'/'
        for file in files:
            full_file_name = os.path.join(indir+"images/", file)
            if file.startswith(model+"_"+domain+"_"+type+"_"):
                upload_to_bucket(bucket_name, full_file_name, outdir+file)
                cache_control_value = "max-age=60"
                set_cache_control(bucket_name, outdir+file, cache_control_value)

check_file_path = indir+"copy_ECMWF-IFS_data_"+set_hemisphere+".check"
with open(check_file_path, 'w') as file:
    file.write("File size check passed!")
