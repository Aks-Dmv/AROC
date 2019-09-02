#!/bin/sh
cd ..
PYTHONPATH='.' python3.7 start_scripts/ocpgAv.py --env TL-v0 --workers 16 --save-model-dir saved/ocpgAvLightssave/ --log-dir saved/ocpgAvLightssave/ --options 4 --delib 0.1
