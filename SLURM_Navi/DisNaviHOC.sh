#!/bin/sh
#SBATCH --job-name=DisNaviHOC # Job name
#SBATCH --ntasks=17 # Run on a single CPU
#SBATCH --time=00:30:00 # Time limit hrs:min:sec
#SBATCH --output=logs/DisNaviHOC%j.out # Standard output and error log
#SBATCH --partition=cl2_48h-1G
#SBATCH --chdir=../
PYTHONPATH='.' python3.7 start_scripts/hoc.py --env Navi-v0 --workers 16 --save-model-dir saved/hocDisNavisave/ --log-dir saved/hocDisNavisave/ --options 4 --delib 0.1
