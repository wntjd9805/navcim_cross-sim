#!/bin/bash


python run_inference_profiling.py --model=$1 --profile_ADC_inputs=1 --NrowsMax=64 --NcolsMax=64 --cell_bits=$2
python -u ${NAVCIM_DIR}/cross-sim/applications/dnn/inference/adc/${1}_adc_limits_bitsliced.py --NrowsMax=64 --NcolsMax=64 --gpu=0 --cell_bits=$2
rm -r ${NAVCIM_DIR}/cross-sim/applications/dnn/inference/adc/profiled_adc_inputs/imagenet_${1}*

python run_inference_profiling.py --model=$1 --profile_ADC_inputs=1 --NrowsMax=64 --NcolsMax=128 --cell_bits=$2
python -u ${NAVCIM_DIR}/cross-sim/applications/dnn/inference/adc/${1}_adc_limits_bitsliced.py --NrowsMax=64 --NcolsMax=128 --gpu=0 --cell_bits=$2
rm -r ${NAVCIM_DIR}/cross-sim/applications/dnn/inference/adc/profiled_adc_inputs/imagenet_${1}*

python run_inference_profiling.py --model=$1 --profile_ADC_inputs=1 --NrowsMax=64 --NcolsMax=256 --cell_bits=$2
python -u ${NAVCIM_DIR}/cross-sim/applications/dnn/inference/adc/${1}_adc_limits_bitsliced.py --NrowsMax=64 --NcolsMax=256 --gpu=0 --cell_bits=$2
rm -r ${NAVCIM_DIR}/cross-sim/applications/dnn/inference/adc/profiled_adc_inputs/imagenet_${1}*

python run_inference_profiling.py --model=$1 --profile_ADC_inputs=1 --NrowsMax=128 --NcolsMax=64 --cell_bits=$2
python -u ${NAVCIM_DIR}/cross-sim/applications/dnn/inference/adc/${1}_adc_limits_bitsliced.py --NrowsMax=128 --NcolsMax=64 --gpu=0 --cell_bits=$2
rm -r ${NAVCIM_DIR}/cross-sim/applications/dnn/inference/adc/profiled_adc_inputs/imagenet_${1}*

python run_inference_profiling.py --model=$1 --profile_ADC_inputs=1 --NrowsMax=128 --NcolsMax=128 --cell_bits=$2
python -u ${NAVCIM_DIR}/cross-sim/applications/dnn/inference/adc/${1}_adc_limits_bitsliced.py --NrowsMax=128 --NcolsMax=128 --gpu=0 --cell_bits=$2
rm -r ${NAVCIM_DIR}/cross-sim/applications/dnn/inference/adc/profiled_adc_inputs/imagenet_${1}*

python run_inference_profiling.py --model=$1 --profile_ADC_inputs=1 --NrowsMax=128 --NcolsMax=256 --cell_bits=$2
python -u ${NAVCIM_DIR}/cross-sim/applications/dnn/inference/adc/${1}_adc_limits_bitsliced.py --NrowsMax=128 --NcolsMax=256 --gpu=0 --cell_bits=$2
rm -r ${NAVCIM_DIR}/cross-sim/applications/dnn/inference/adc/profiled_adc_inputs/imagenet_${1}*

python run_inference_profiling.py --model=$1 --profile_ADC_inputs=1 --NrowsMax=256 --NcolsMax=64 --cell_bits=$2
python -u ${NAVCIM_DIR}/cross-sim/applications/dnn/inference/adc/${1}_adc_limits_bitsliced.py --NrowsMax=256 --NcolsMax=64 --gpu=0 --cell_bits=$2
rm -r ${NAVCIM_DIR}/cross-sim/applications/dnn/inference/adc/profiled_adc_inputs/imagenet_${1}*

python run_inference_profiling.py --model=$1 --profile_ADC_inputs=1 --NrowsMax=256 --NcolsMax=128 --cell_bits=$2
python -u ${NAVCIM_DIR}/cross-sim/applications/dnn/inference/adc/${1}_adc_limits_bitsliced.py --NrowsMax=256 --NcolsMax=128 --gpu=0 --cell_bits=$2
rm -r ${NAVCIM_DIR}/cross-sim/applications/dnn/inference/adc/profiled_adc_inputs/imagenet_${1}*

python run_inference_profiling.py --model=$1 --profile_ADC_inputs=1 --NrowsMax=256 --NcolsMax=256 --cell_bits=$2
python -u ${NAVCIM_DIR}/cross-sim/applications/dnn/inference/adc/${1}_adc_limits_bitsliced.py --NrowsMax=256 --NcolsMax=256 --gpu=0 --cell_bits=$2
rm -r ${NAVCIM_DIR}/cross-sim/applications/dnn/inference/adc/profiled_adc_inputs/imagenet_${1}*


