#!/bin/sh
#SBATCH --job-name=AvLightsAC# Job name
#SBATCH --ntasks=16 # Run on a single CPU
#SBATCH --time=00:30:00 # Time limit hrs:min:sec
#SBATCH --output=logs/AvLightsAC%j.out # Standard output and error log
#SBATCH --partition=cl1_48h-1G
#SBATCH --chdir=../
PYTHONPATH='.' python3.7 start_scripts/acAv.py --env TL-v0 --workers 16 --save-model-dir saved/acAvLightssave/ --log-dir saved/acAvLightssave/ --options 4 --delib 0.1 --num-steps 1000
