import subprocess
import json
from datetime import datetime
from dateutil.relativedelta import relativedelta
import gc

server_run="http://localhost:8080/trigger-plot-job"
#server_run="https://ecmwf-plot-service-dfsedjpfbq-uc.a.run.app/generate-plot"

command = [
    "curl",
    "-X",
    "POST",
    "-H",
    "Content-Type: application/json",
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
