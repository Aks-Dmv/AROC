#!/bin/sh
cd ..
PYTHONPATH='.' python3.7 start_scripts/ocpg.py --env Navi-v0 --workers 16 --save-model-dir saved/ocpgDisNavisave/ --log-dir saved/ocpgDisNavisave/ --options 4 --delib 0.1
