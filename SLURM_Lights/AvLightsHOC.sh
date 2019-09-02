#!/bin/sh
#SBATCH --job-name=AvLightsHOC# Job name
#SBATCH --ntasks=16 # Run on a single CPU
#SBATCH --time=00:30:00 # Time limit hrs:min:sec
#SBATCH --output=logs/AvLightsHOC%j.out # Standard output and error log
#SBATCH --partition=cl1_48h-1G
#SBATCH --chdir=../
PYTHONPATH='.' python3.7 start_scripts/hocAv.py --env TL-v0 --workers 16 --save-model-dir saved/hocAvLightssave/ --log-dir saved/hocAvLightssave/ --options 4 --delib 0.1 --num-steps 1000
