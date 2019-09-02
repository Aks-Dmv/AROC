#!/bin/sh
#SBATCH --job-name=DisNaviAC # Job name
#SBATCH --ntasks=16 # Run on a single CPU
#SBATCH --time=00:30:00 # Time limit hrs:min:sec
#SBATCH --output=logs/DisNaviAC%j.out # Standard output and error log
#SBATCH --partition=cl1_48h-1G
#SBATCH --chdir=../
PYTHONPATH='.' python3.7 start_scripts/ac.py --env Navi-v0 --workers 16 --save-model-dir saved/acDisNavisave/ --log-dir saved/acDisNavisave/ --options 4 --delib 0.1
