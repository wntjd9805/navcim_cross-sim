#!/bin/bash
python run_inference_greedy.py --model=$1 --adc_bits=4 --cell_bits=1
python run_inference_greedy.py --model=$1 --adc_bits=4 --cell_bits=2
python run_inference_greedy.py --model=$1 --adc_bits=4 --cell_bits=4
python run_inference_greedy.py --model=$1 --adc_bits=5 --cell_bits=1
python run_inference_greedy.py --model=$1 --adc_bits=5 --cell_bits=2
python run_inference_greedy.py --model=$1 --adc_bits=5 --cell_bits=4
python run_inference_greedy.py --model=$1 --adc_bits=6 --cell_bits=1
python run_inference_greedy.py --model=$1 --adc_bits=6 --cell_bits=2
python run_inference_greedy.py --model=$1 --adc_bits=6 --cell_bits=4