# app.py
# This file provides a simple web endpoint to trigger the execution of another Python script.

import os
import subprocess
from flask import Flask, jsonify, request

# Initialize the Flask application
app = Flask(__name__)

# A simple health check route
@app.route('/')
def health_check():
    """Returns a simple message to confirm the service is running."""
    return "Service is running!", 200

@app.route('/trigger-plot-job', methods=['POST'])
def trigger_plot_job():
    """
    HTTP endpoint to trigger the execution of the get_and_plot.py script.
    
    This endpoint does not require any input data as the script itself
    is self-contained.
    """
    try:
        # Define the path to the script relative to the working directory (/app)
        script_path = 'stations/get_and_plot_weatherdata.py'

        # Execute the get_and_plot.py script using a subprocess.
        # This runs the script as a separate process and waits for it to finish.
        # The capture_output=True and text=True arguments capture the output for logging.
        result = subprocess.run(
            ['python3', script_path],
            capture_output=True,
            text=True,
            check=True
        )

        # Log the standard output and standard error from the script
        print("Script stdout:", result.stdout)
        print("Script stderr:", result.stderr)

        return jsonify({
            "message": "Script executed successfully.",
            "stdout": result.stdout,
            "stderr": result.stderr
        }), 200

    except subprocess.CalledProcessError as e:
        # Handle cases where the script exits with a non-zero status code (an error)
        print(f"Script failed with exit code {e.returncode}:")
        print("Script stdout:", e.stdout)
        print("Script stderr:", e.stderr)
        return jsonify({
            "error": "Failed to run script.",
            "details": str(e),
            "stdout": e.stdout,
            "stderr": e.stderr
        }), 500
    except Exception as e:
        # Handle any other unexpected errors
        print(f"An unexpected error occurred: {e}")
        return jsonify({
            "error": "An internal server error occurred.",
            "details": str(e)
        }), 500

if __name__ == '__main__':
    # This block is for local testing. Cloud Run will use gunicorn.
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
