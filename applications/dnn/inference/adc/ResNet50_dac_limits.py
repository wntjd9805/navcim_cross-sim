#
# Copyright 2017-2023 Sandia Corporation. Under the terms of Contract DE-AC04-94AL85000 with
# Sandia Corporation, the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#

import numpy as np
from scipy.optimize import minimize
import time
import os

# Use this file to produce a list of DAC (min, max) ranges for each layer of ResNet50
# DAC limits should be independent of most hardware settings such as input bitslicing, 
# weight bitslicing, # rows, and negative number handling, but are affected by fold_batchnorm, 
# weight_bits, etc

# Folder where profiled values are stored
profiling_folder = "./profiled_dac_inputs/imagenet_ResNet50/"
# npy output file path
output_name = "./adc_limits/examples/dac_limits_ResNet50.npy"

# Quantization loss function settings
Nbits = 8 # not necessarily the ADC resolution to be used
norm_ord = 1

# Enable GPU
useGPU = True

if useGPU:
	import cupy as cp
	gpu_num = 0
	os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu_num)
	ncp = cp
else:
	ncp = np

# Layer nums for ResNet conv and dense layers
layerNums = [4, 12, 17, 22, 27, 36, 41, 46, 55, 60, 65, 74, 79, 84, 89, 98, 103, 108, 117, 122, 127, 136, 141, 146, 155, 160, 165, 170, 179, 184, 189, 198, 203, 208, 217, 222, 227, 236, 241, 246, 255, 260, 265, 274, 279, 284, 289, 298, 303, 308, 317, 322, 327, 342]

# Quantization loss function
def quantizationError_max(eta,x,Nbits,norm_ord):
	# Clip
	P = 100*(1 - pow(10,eta))
	P = ncp.clip(P,0,100)

	ADCmin = ncp.min(x)
	ADCmax = ncp.percentile(x,P)
	x_Q = x.copy()
	x_Q = x_Q.clip(ADCmin,ADCmax)

	# Quantize
	qmult = (2**Nbits-1) / (ADCmax - ADCmin)
	x_Q = (x_Q - ADCmin)*qmult
	x_Q = ncp.rint(x_Q,out=x_Q)
	x_Q /= qmult
	x_Q += ADCmin

	err = ncp.linalg.norm(x-x_Q,ord=norm_ord)
	return float(err)

################
##
##  Determine DAC ranges
##
################
DAC_limits = np.zeros((len(layerNums),2))

# First layer's DAC limits come directly from the limits of the preprocessed ImageNet test set
# Using these limits for the first layer ensures no clipping will occur
# DAC_limits[0,:] = np.array([-124,152])

for k in range(0,len(layerNums)):
	print('Layer '+str(k))	
	x_dist_DAC = ncp.load(profiling_folder+"dac_inputs_layer"+str(layerNums[k])+".npy")

	DAC_max = ncp.asnumpy(ncp.max(x_dist_DAC))
	DAC_min = ncp.asnumpy(ncp.min(x_dist_DAC))

	# Optimize the DAC percentile
	if Nbits > 0:
		eta0 = -4
		eta = minimize(quantizationError_max,eta0,args=(x_dist_DAC,Nbits,norm_ord),method='nelder-mead',tol=0.1)
		percentile_DAC = 100*(1-pow(10,eta.x[0]))
		print('    DAC Percentile: {:.3f}'.format(percentile_DAC))
		xmin_DAC = ncp.asnumpy(ncp.min(x_dist_DAC))
		xmax_DAC = ncp.asnumpy(ncp.percentile(x_dist_DAC,percentile_DAC))
	else:			
		xmin_DAC = DAC_min
		xmax_DAC = DAC_max
	clipped = (DAC_max-xmax_DAC)/(DAC_max-DAC_min)
	print('    Percentage of DAC range clipped: {:.3f}'.format(clipped*100)+'%')

	# Set the limits
	DAC_limits[k,:] = np.array([xmin_DAC,xmax_DAC])

print(f"Profiled activation limits: in {output_name}")
print(DAC_limits)

# Save the generated limits to a npy file
np.save(output_name,DAC_limits)