#!/bin/sh
cd ..
PYTHONPATH='.' python3.7 start_scripts/hocAv.py --env TL-v0 --workers 16 --save-model-dir saved/hocAvLightssave/ --log-dir saved/hocAvLightssave/ --options 4 --delib 0.1
