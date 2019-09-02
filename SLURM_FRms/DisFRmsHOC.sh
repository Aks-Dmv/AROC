#!/bin/sh
#SBATCH --job-name=DisFRmsHOC # Job name
#SBATCH --ntasks=16 # Run on a single CPU
#SBATCH --time=00:30:00 # Time limit hrs:min:sec
#SBATCH --output=logs/DisFRmsHOC%j.out # Standard output and error log
#SBATCH --partition=cl1_48h-1G
#SBATCH --chdir=../
PYTHONPATH='.' python3.7 start_scripts/hoc.py --env FourRooms-v0 --workers 16 --save-model-dir saved/hocDisFRmssave/ --log-dir saved/hocDisFRmssave/ --options 4 --delib 0.1 --num-steps 100
