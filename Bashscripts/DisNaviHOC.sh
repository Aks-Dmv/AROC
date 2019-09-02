#!/bin/sh
cd ..
PYTHONPATH='.' python3.7 start_scripts/hoc.py --env Navi-v0 --workers 16 --save-model-dir saved/hocDisNavisave/ --log-dir saved/hocDisNavisave/ --options 4 --delib 0.1
