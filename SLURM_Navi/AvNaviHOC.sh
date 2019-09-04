#!/bin/sh
#SBATCH --job-name=AvNaviHOC # Job name
#SBATCH --ntasks=17 # Run on a single CPU
#SBATCH --time=00:30:00 # Time limit hrs:min:sec
#SBATCH --output=logs/AvNaviHOC%j.out # Standard output and error log
#SBATCH --partition=cl1_48h-1G
#SBATCH --chdir=../
PYTHONPATH='.' python3.7 start_scripts/hocAv.py --env Navi-v0 --workers 16 --save-model-dir saved/hocAvNavisave/ --log-dir saved/hocAvNavisave/ --options 4 --delib 0.1
