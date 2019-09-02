#!/bin/sh
cd ..
PYTHONPATH='.' python3.7 start_scripts/hoc.py --env TL-v0 --workers 16 --save-model-dir saved/hocDisLightssave/ --log-dir saved/hocDisLightssave/ --options 4 --delib 0.1
