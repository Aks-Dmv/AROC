#!/bin/sh
#SBATCH --job-name=DisFRmsOCPG # Job name
#SBATCH --ntasks=16 # Run on a single CPU
#SBATCH --time=00:30:00 # Time limit hrs:min:sec
#SBATCH --output=logs/DisFRmsOCPG%j.out # Standard output and error log
#SBATCH --partition=cl1_48h-1G
#SBATCH --chdir=../
PYTHONPATH='.' python3.7 start_scripts/ocpg.py --env FourRooms-v0 --workers 16 --save-model-dir saved/ocpgDisFRmssave/ --log-dir saved/ocpgDisFRmssave/ --options 4 --delib 0.1 --num-steps 100