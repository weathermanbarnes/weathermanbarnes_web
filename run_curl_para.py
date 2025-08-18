import os
import tempfile
import atexit
import shutil
import subprocess
import json
from datetime import datetime
from dateutil.relativedelta import relativedelta
import gc
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)
from multiprocessing import Pool # Import the Pool class
from tqdm import tqdm

def process_tstep(tstep, indatetime, server_run):
    """
    Function to execute the curl command for a single tstep.
    """
    print(f"Processing tstep: {tstep}")
    
    if tstep > 0:
        init_date = indatetime.strftime("%Y%m%d")
        init_time = indatetime.strftime("%H%M%S")
        init_hour = indatetime.strftime("%H")

        command = [
            "curl",
            "--max-time", "600",
            "-s",
            "-X",
            "POST",
            "-H",
            "Content-Type: application/json",
            "-d",
            json.dumps({
                "init_date": f"{indatetime.strftime('%Y%m%d%H')}",
                "ecmwf_data_bucket": "ecmwf-open-data",
                "ecmwf_data_blob": f"{init_date}/{init_hour}z/ifs/0p25/oper/{init_date}{init_time}-{tstep}h-oper-fc.grib2",
                "ecmwf_precip_data_blob": f"{init_date}/{init_hour}z/ifs/0p25/oper/{init_date}{init_time}-{int(tstep-6)}h-oper-fc.grib2",
                "output_bucket_name": "www.weathermanbarnes.com",
                "fignum": tstep
            }),
            server_run
        ]
    else:
        # (your existing 'else' logic for tstep <= 0)
        hist_indatetime = indatetime + relativedelta(hours=tstep)
        init_date = hist_indatetime.strftime("%Y%m%d")
        init_time = hist_indatetime.strftime("%H%M%S")
        init_hour = hist_indatetime.strftime("%H")
        if int(init_hour) == 6 or int(init_hour) == 18:
            fctype = 'scda'
        else:
            fctype = 'oper'

        t6dt = hist_indatetime - relativedelta(hours=6)
        t6_init_date = t6dt.strftime("%Y%m%d")
        t6_init_time = t6dt.strftime("%H%M%S")
        t6_init_hour = t6dt.strftime("%H")
        if int(t6_init_hour) == 6 or int(t6_init_hour) == 18:
            t6fctype = 'scda'
        else:
            t6fctype = 'oper'

        command = [
            "curl",
            "--max-time", "600",
            "-s",
            "-X",
            "POST",
            "-H",
            "Content-Type: application/json",
            "-d",
            json.dumps({
                "init_date": f"{indatetime.strftime('%Y%m%d%H')}",
                "ecmwf_data_bucket": "ecmwf-open-data",
                "ecmwf_data_blob": f"{init_date}/{init_hour}z/ifs/0p25/{fctype}/{init_date}{init_time}-{int(0)}h-{fctype}-fc.grib2",
                "ecmwf_precip_data_blob": f"{t6_init_date}/{t6_init_hour}z/ifs/0p25/{t6fctype}/{t6_init_date}{t6_init_time}-{int(6)}h-{t6fctype}-fc.grib2",
                "output_bucket_name": "www.weathermanbarnes.com",
                "fignum": tstep
            }),
            server_run
        ]

    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        print(f"Curl command for tstep {tstep} executed successfully!")
        print("STDOUT:")
        print(result.stdout)
        print("\nSTDERR:")
        print(result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Error executing curl command for tstep {tstep}: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
    except FileNotFoundError:
        print(f"Error: 'curl' command not found for tstep {tstep}.")
    except Exception as e:
        print(f"An unexpected error occurred for tstep {tstep}: {e}")

    gc.collect()

# ... (all your imports and parser setup)
import argparse
from multiprocessing import Pool

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("date", help="init date", type=str)
    parser.add_argument("run", help="init run", type=int)
    args = parser.parse_args()
    INDATEstr = args.date
    RUN = args.run

    multiprocessing.set_start_method('spawn', force=True)

    indatetime = datetime.strptime(INDATEstr, '%Y%m%d')
    indatetime = indatetime + relativedelta(hours=RUN)

    #server_run="http://localhost:8080/generate-plot"
    server_run = "https://ecmwf-plot-service-dfsedjpfbq-uc.a.run.app/generate-plot"

    # Create a list of all the tstep values you want to process
    tsteps_to_process = list(range(-24, 246, 6))#246, 6))

    # Determine the number of processes to use.
    # It's often a good practice to use the number of CPU cores available.
    num_processes = 2#os.cpu_count() or 4 # Default to 4 if not available

    # Use a 'with' statement for the Pool to ensure processes are cleaned up properly
    with Pool(processes=num_processes) as pool:
        # Use a partial function to "freeze" the other arguments (indatetime, server_run)
        from functools import partial
        worker_func = partial(process_tstep, indatetime=indatetime, server_run=server_run)
        
        # 'map' will apply the worker_func to each item in tsteps_to_process
        # pool.map(worker_func, tsteps_to_process)
        # The list() call is necessary to consume the iterator and wait for all tasks to complete.
        results = list(tqdm(pool.imap(worker_func, tsteps_to_process), total=len(tsteps_to_process)))

    print("All tsteps have been processed.")