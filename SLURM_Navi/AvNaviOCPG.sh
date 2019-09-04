#!/bin/sh
#SBATCH --job-name=AvNaviOCPG # Job name
#SBATCH --ntasks=17 # Run on a single CPU
#SBATCH --time=00:30:00 # Time limit hrs:min:sec
#SBATCH --output=logs/AvNaviOCPG%j.out # Standard output and error log
#SBATCH --partition=cl1_48h-1G
#SBATCH --chdir=../
PYTHONPATH='.' python3.7 start_scripts/ocpgAv.py --env Navi-v0 --workers 16 --save-model-dir saved/ocpgAvNavisave/ --log-dir saved/ocpgAvNavisave/ --options 4 --delib 0.1
