#!/bin/sh
#SBATCH --job-name=AvNaviAC # Job name
#SBATCH --ntasks=17 # Run on a single CPU
#SBATCH --time=00:30:00 # Time limit hrs:min:sec
#SBATCH --output=logs/AvNaviAC%j.out # Standard output and error log
#SBATCH --partition=cl1_48h-1G
#SBATCH --chdir=../
PYTHONPATH='.' python3.7 start_scripts/acAv.py --env Navi-v0 --workers 16 --save-model-dir saved/acAvNavisave/ --log-dir saved/acAvNavisave/ --options 4 --delib 0.1
