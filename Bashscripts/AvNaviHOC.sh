#!/bin/sh
cd ..
PYTHONPATH='.' python3.7 start_scripts/hocAv.py --env Navi-v0 --workers 16 --save-model-dir saved/hocAvNavisave/ --log-dir saved/hocAvNavisave/ --options 4 --delib 0.1
