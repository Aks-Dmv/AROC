#!/bin/sh
cd ..
PYTHONPATH='.' python3.7 start_scripts/ocpgAv.py --env Navi-v0 --workers 16 --save-model-dir saved/ocpgAvNavisave/ --log-dir saved/ocpgAvNavisave/ --options 4 --delib 0.1
