# Use an official Python runtime as a parent image
# Updated to use 'bookworm' (Debian 12), a more recent and supported Debian release
FROM python:3.10-slim-bookworm

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required for plotting libraries (e.g., matplotlib)
# This is crucial for libraries like matplotlib that might rely on C libraries. python3-numpy
RUN apt-get update && apt-get install -y \
    build-essential \
    libffi-dev \
    libssl-dev \
    libjpeg-dev \
    libeccodes-dev \
    zlib1g-dev \
    --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

# Copy the requirements.txt file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# This includes your data science libraries (e.g., xarray, netCDF4, matplotlib, numpy, pandas)
# and the Google Cloud Storage client library.
# Install numpy first as some packages have dependencies on numpy
RUN pip install --upgrade pip
RUN pip install --no-cache-dir numpy && pip cache purge
#Install reequirements
RUN pip install --no-cache-dir -r requirements.txt && pip cache purge

COPY weathermanbarnes_web/ weathermanbarnes_web/

# Copy your Python script into the container
COPY app.py .

# Define the command to run your application
# Cloud Run expects your application to listen on the port specified by the PORT environment variable.
# We'll use gunicorn to serve a simple Flask app that triggers your plotting script.
CMD exec gunicorn --bind :$PORT --workers 4 --threads 1 app:app

