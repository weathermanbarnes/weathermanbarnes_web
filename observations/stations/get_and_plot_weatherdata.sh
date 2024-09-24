#!/bin/bash

###### Load conda modules ######
module use /g/data/hh5/public/modules
module load conda/analysis3

###### Change to the repo directory ######
cd /home/565/mb0427/gdata-w40/Forecasts/ops_scripts/observations/stations/

###### Run python scripts ######\n\
python3 get_and_plot_weatherdata.py 

sleep 0.5
