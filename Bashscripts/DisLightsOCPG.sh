#!/bin/sh
cd ..
PYTHONPATH='.' python3.7 start_scripts/ocpg.py --env TL-v0 --workers 16 --save-model-dir saved/ocpgDisLightssave/ --log-dir saved/ocpgDisLightssave/ --options 4 --delib 0.1
