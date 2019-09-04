#!/bin/sh
#SBATCH --job-name=AvFRmsHOC # Job name
#SBATCH --ntasks=9 # Run on a single CPU
#SBATCH --time=10:30:00 # Time limit hrs:min:sec
#SBATCH --output=logs/AvFRmsHOC%j.out # Standard output and error log
#SBATCH --partition=cl1_48h-1G
#SBATCH --chdir=../
PYTHONPATH='.' python3.7 start_scripts/hocAv.py --env FourRooms-v0 --workers 8 --save-model-dir saved/hocAvFRmssave/ --log-dir saved/hocAvFRmssave/ --options 3 --delib 0.1 --num-steps 50
