import subprocess
import json
from datetime import datetime
from dateutil.relativedelta import relativedelta
import gc

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("date",help="init date",type=str)
parser.add_argument("run",help="init run",type=int)
#parser.add_argument("tstep",help="in timestep",type=int)
args = parser.parse_args()
INDATEstr = args.date
RUN = args.run
#tstep = args.tstep

indatetime=datetime.strptime(INDATEstr,'%Y%m%d')
indatetime=indatetime+relativedelta(hours=RUN)

#server_run="http://localhost:8080/generate-plot"
server_run="https://ecmwf-plot-service-dfsedjpfbq-uc.a.run.app/generate-plot"

for tstep in range(-24,246,6):
    print(tstep)
    if tstep>0:
        init_date = indatetime.strftime("%Y%m%d")
        init_time = indatetime.strftime("%H%M%S")
        init_hour = indatetime.strftime("%H")

        command = [
            "curl",
            "-X",
            "POST",
            "-H",
            "Content-Type: application/json",
            "-d",
            json.dumps({ # Use json.dumps to convert the Python dict to a JSON string
                "ecmwf_data_bucket": "ecmwf-open-data",
                "ecmwf_data_blob": f"{init_date}/{init_hour}z/ifs/0p25/oper/{init_date}{init_time}-{tstep}h-oper-fc.grib2",
                "ecmwf_precip_data_blob": f"{init_date}/{init_hour}z/ifs/0p25/oper/{init_date}{init_time}-{int(tstep-6)}h-oper-fc.grib2",
                "output_bucket_name": "www.weathermanbarnes.com",
                "fignum": tstep
            }),
            server_run
        ]
    else:
        hist_indatetime=indatetime+relativedelta(hours=tstep)
        init_date = hist_indatetime.strftime("%Y%m%d")
        init_time = hist_indatetime.strftime("%H%M%S")
        init_hour = hist_indatetime.strftime("%H")
        if int(init_hour) == 6 or int(init_hour)==18:
            fctype='scda'
        else:
            fctype='oper'

        t6dt=hist_indatetime-relativedelta(hours=6)
        t6_init_date = t6dt.strftime("%Y%m%d")
        t6_init_time = t6dt.strftime("%H%M%S")
        t6_init_hour = t6dt.strftime("%H")
        if int(t6_init_hour) == 6 or int(t6_init_hour)==18:
            t6fctype='scda'
        else:
            t6fctype='oper'

        command = [
            "curl",
            "-X",
            "POST",
            "-H",
            "Content-Type: application/json",
            "-d",
            json.dumps({ # Use json.dumps to convert the Python dict to a JSON string
                "ecmwf_data_bucket": "ecmwf-open-data",
                "ecmwf_data_blob": f"{init_date}/{init_hour}z/ifs/0p25/{fctype}/{init_date}{init_time}-{int(0)}h-{fctype}-fc.grib2",
                "ecmwf_precip_data_blob": f"{t6_init_date}/{t6_init_hour}z/ifs/0p25/{t6fctype}/{t6_init_date}{t6_init_time}-{int(6)}h-{t6fctype}-fc.grib2",
                "output_bucket_name": "www.weathermanbarnes.com",
                "fignum": tstep
            }),
            server_run
        ]

    try:
        # Run the command
        # capture_output=True will capture stdout and stderr
        # text=True decodes stdout/stderr as text using default encoding
        result = subprocess.run(command, capture_output=True, text=True, check=True)

        # Print the output from the curl command
        print("STDOUT:")
        print(result.stdout)
        print("\nSTDERR:")
        print(result.stderr)

        gc.collect()
        print("\nCurl command executed successfully!")

    except subprocess.CalledProcessError as e:
        # This block will be executed if the curl command returns a non-zero exit code
        print(f"Error executing curl command: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
    except FileNotFoundError:
        print("Error: 'curl' command not found. Make sure curl is installed and in your system's PATH.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
