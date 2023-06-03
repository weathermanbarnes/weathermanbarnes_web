#!/bin/bash
export DATE=$(date +%Y%m%d -d "1 day ago")
export RUN=12

###### Load conda modules ######
module use /g/data/hh5/public/modules
module load conda/analysis3

###### Change to the repo directory ######
cd /home/565/mb0427/MonashWeb/monash_weather_web/

###### Run python scripts ######\n\
python3 plot_copy_to_dir.py

sleep 0.5
