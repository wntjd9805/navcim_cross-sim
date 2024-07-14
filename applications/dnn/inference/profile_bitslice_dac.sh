#!/bin/bash
python run_inference_profiling.py --model=$1 --profile_DAC_inputs=1
python -u ${NAVCIM_DIR}/cross-sim/applications/dnn/inference/adc/${1}_dac_limits.py 
