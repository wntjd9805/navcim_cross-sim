#!/bin/bash
python -u ${1}_adc_limits_1slice.py --NrowsMax=64 --NcolsMax=64 --gpu=0 &
python -u ${1}_adc_limits_1slice.py --NrowsMax=64 --NcolsMax=128 --gpu=1 &
python -u ${1}_adc_limits_1slice.py --NrowsMax=64 --NcolsMax=256 --gpu=2 &
python -u ${1}_adc_limits_1slice.py --NrowsMax=128 --NcolsMax=64 --gpu=3 &
python -u ${1}_adc_limits_1slice.py --NrowsMax=128 --NcolsMax=128 --gpu=4 &
python -u ${1}_adc_limits_1slice.py --NrowsMax=128 --NcolsMax=256 --gpu=5 &
python -u ${1}_adc_limits_1slice.py --NrowsMax=256 --NcolsMax=64 --gpu=6 &
python -u ${1}_adc_limits_1slice.py --NrowsMax=256 --NcolsMax=128 --gpu=7 & 
python -u ${1}_adc_limits_1slice.py --NrowsMax=256 --NcolsMax=256 --gpu=0 &
wait