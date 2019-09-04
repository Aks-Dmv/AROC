#!/bin/sh
#SBATCH --job-name=DisLightsOCPG # Job name
#SBATCH --ntasks=17 # Run on a single CPU
#SBATCH --time=00:30:00 # Time limit hrs:min:sec
#SBATCH --output=logs/DisLightsOCPG%j.out # Standard output and error log
#SBATCH --partition=cl2_48h-1G
#SBATCH --chdir=../
PYTHONPATH='.' python3.7 start_scripts/ocpg.py --env TL-v0 --workers 16 --save-model-dir saved/ocpgDisLightssave/ --log-dir saved/ocpgDisLightssave/ --options 4 --delib 0.1 --num-steps 1000
