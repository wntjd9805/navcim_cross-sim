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
import argparse
import gc
# Use this file to produce a list of ADC (min, max) ranges for each layer of ResNet50
# This is to be used for files that do not use either input or weight bit slicing
# This allows an optimizer function to be used to set the limits because there are no real hardware constraints on the limits

# Max number of rows
parser = argparse.ArgumentParser(description='PyTorch CIFAR-X Example')
parser.add_argument('--NrowsMax', type=int, default=1152, help='NrowsMax')
parser.add_argument('--NcolsMax', type=int, default=1152, help='NcolsMax')
parser.add_argument('--gpu', type=int, default=0, help='NcolsMax')
args = parser.parse_args()

NrowsMax = args.NrowsMax
NcolsMax = args.NcolsMax
# Mapping style
xbar_style = "BALANCED"

# Whether profile_ADC_biased was True for the ADC inputs
reluAwareLimits = True

# Input bit slicing (must be False if reluAwareLmits = True)
input_bitslicing = False

##### Folder where profiled values are stored
##### Make sure the correct file is selected that is consistent with the options above!
ibit_msg = ("_ibits" if input_bitslicing else "")
relu_msg = ("_reluAware" if reluAwareLimits else "")

profiling_folder = "./profiled_adc_inputs/imagenet_Resnet50_"+\
			str(NrowsMax)+"rows_"+str(NcolsMax)+"cols_"+"1slices_"+xbar_style+ibit_msg+relu_msg+"/"

##### npy output file path
output_name = "./adc_limits/examples/adc_limits_Resnet50_"+\
				str(NrowsMax)+"rows_"+str(NcolsMax)+"cols_"+"1slices_"+xbar_style+ibit_msg+relu_msg+".npy"

output_name_candidate = "./adc_limits/examples/adc_limits_Resnet50_"+\
				str(NrowsMax)+"rows_"+str(NcolsMax)+"cols_"+"1slices_"+xbar_style+ibit_msg+relu_msg+"_candidate.npy"

# Quantization loss function settings
Nbits = 8 # not necessarily the ADC resolution to be used
norm_ord = 1

# Enable GPU
useGPU = True

if useGPU:
	import cupy as cp
	os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
	ncp = cp
else:
	ncp = np

###
### ResNet50 specific parameters
###

# Layer nums for ResNet conv and dense layers
layerNums = [4, 12, 17, 22, 27, 36, 41, 46, 55, 60, 65, 74, 79, 84, 89, 98, 103, 108, 117, 122, 127, 136, 141, 146, 155, 160, 165, 170, 179, 184, 189, 198, 203, 208, 217, 222, 227, 236, 241, 246, 255, 260, 265, 274, 279, 284, 289, 298, 303, 308, 317, 322, 327, 342] 

# Number of matrix partitions for each layer, the list below assumes NrowsMax = 1152 has been used
# This is used to determine whether ADC minimum should be optimized separately, if using ReLU-aware limits
# if NrowsMax == 1152:
# 	ncoresList =[1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 5, 1, 1, 5, 1, 1, 5, 1, 1, 8, 1, 1, 8, 1, 1, 8, 1, 1, 2]
# elif NrowsMax == 576:
# 	ncoresList = [1, 1, 1, 1, 2, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 6, 1, 1, 6, 1, 1, 6, 1, 1, 6, 1, 1, 9, 1, 1, 9, 1, 1, 9, 1, 1, 15, 2, 1, 15, 2, 1, 15, 2, 1, 3]
# elif NrowsMax == 288:
# 	ncoresList =  [1, 1, 1, 1, 1, 1, 1, 3, 3, 1, 1, 1, 5, 5, 1, 1, 1, 5, 5, 1, 1, 1, 6, 6, 1, 1, 1, 6, 6, 1, 1, 1, 6, 6, 1, 1, 1, 12, 12, 2, 1, 1, 12, 12, 2, 1, 1, 12, 12, 2, 1, 1, 12, 12, 2, 1, 1, 18, 18, 2, 1, 1, 18, 18, 2, 1, 1, 18, 18, 2, 1, 1, 30, 30, 4, 1, 1, 30, 30, 4, 1, 1, 30, 30, 4, 2, 2, 5]
# elif NrowsMax == 144:
# 	ncoresList = [1, 1, 2, 2, 1, 1, 1, 6, 6, 1, 1, 1, 9, 9, 1, 1, 1, 9, 9, 1, 1, 1, 12, 12, 2, 1, 1, 12, 12, 2, 1, 1, 12, 12, 2, 1, 1, 24, 24, 3, 1, 1, 24, 24, 3, 1, 1, 24, 24, 3, 1, 1, 24, 24, 3, 1, 1, 36, 36, 4, 1, 1, 36, 36, 4, 1, 1, 36, 36, 4, 2, 2, 60, 60, 7, 2, 2, 60, 60, 7, 2, 2, 60, 60, 7, 3, 3, 9]
# elif NrowsMax == 256:
# 	ncoresList = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 8, 10, 20]
# elif NrowsMax == 128:
# 	ncoresList = [1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 5, 5, 5, 5, 5, 10, 16, 16, 16, 16, 16, 24, 30, 80]
# elif NrowsMax == 64:
# 	ncoresList = [1, 1, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 6, 6, 6, 6, 6, 6, 6, 12, 18, 18, 18, 18, 18, 27, 45, 45, 45, 45, 45, 75, 100, 320]

if NrowsMax == 64:
	if NcolsMax == 64:
		ncoresList = [3, 1, 9, 4, 4, 4, 9, 4, 4, 9, 4, 8, 36, 16, 32, 16, 36, 16, 16, 36, 16, 16, 36, 16, 32, 144, 64, 128, 64, 144, 64, 64, 144, 64, 64, 144, 64, 64, 144, 64, 64, 144, 64, 128, 576, 256, 512, 256, 576, 256, 256, 576, 256, 512]
	elif NcolsMax == 128:
		ncoresList = [3, 1, 9, 2, 2, 4, 9, 2, 4, 9, 2, 4, 18, 8, 16, 8, 18, 8, 8, 18, 8, 8, 18, 8, 16, 72, 32, 64, 32, 72, 32, 32, 72, 32, 32, 72, 32, 32, 72, 32, 32, 72, 32, 64, 288, 128, 256, 128, 288, 128, 128, 288, 128, 256]
	elif NcolsMax == 256:
		ncoresList = [3, 1, 9, 1, 1, 4, 9, 1, 4, 9, 1, 4, 18, 4, 8, 8, 18, 4, 8, 18, 4, 8, 18, 4, 8, 36, 16, 32, 16, 36, 16, 16, 36, 16, 16, 36, 16, 16, 36, 16, 16, 36, 16, 32, 144, 64, 128, 64, 144, 64, 64, 144, 64, 128]
elif NrowsMax == 128:
	if NcolsMax == 64:
		ncoresList = [2, 1, 5, 4, 4, 2, 5, 4, 2, 5, 4, 4, 18, 8, 16, 8, 18, 8, 8, 18, 8, 8, 18, 8, 16, 72, 32, 64, 32, 72, 32, 32, 72, 32, 32, 72, 32, 32, 72, 32, 32, 72, 32, 64, 288, 128, 256, 128, 288, 128, 128, 288, 128, 256]
	elif NcolsMax == 128:
		ncoresList = [2, 1, 5, 2, 2, 2, 5, 2, 2, 5, 2, 2, 9, 4, 8, 4, 9, 4, 4, 9, 4, 4, 9, 4, 8, 36, 16, 32, 16, 36, 16, 16, 36, 16, 16, 36, 16, 16, 36, 16, 16, 36, 16, 32, 144, 64, 128, 64, 144, 64, 64, 144, 64, 128]
	elif NcolsMax == 256:
		ncoresList = [2, 1, 5, 1, 1, 2, 5, 1, 2, 5, 1, 2, 9, 2, 4, 4, 9, 2, 4, 9, 2, 4, 9, 2, 4, 18, 8, 16, 8, 18, 8, 8, 18, 8, 8, 18, 8, 8, 18, 8, 8, 18, 8, 16, 72, 32, 64, 32, 72, 32, 32, 72, 32, 64]
elif NrowsMax == 256:
	if NcolsMax == 64:
		ncoresList = [1, 1, 3, 4, 4, 1, 3, 4, 1, 3, 4, 2, 10, 8, 8, 4, 10, 8, 4, 10, 8, 4, 10, 8, 8, 36, 16, 32, 16, 36, 16, 16, 36, 16, 16, 36, 16, 16, 36, 16, 16, 36, 16, 32, 144, 64, 128, 64, 144, 64, 64, 144, 64, 128]
	elif NcolsMax == 128:
		ncoresList = [1, 1, 3, 2, 2, 1, 3, 2, 1, 3, 2, 1, 5, 4, 4, 2, 5, 4, 2, 5, 4, 2, 5, 4, 4, 18, 8, 16, 8, 18, 8, 8, 18, 8, 8, 18, 8, 8, 18, 8, 8, 18, 8, 16, 72, 32, 64, 32, 72, 32, 32, 72, 32, 64]
	elif NcolsMax == 256:
		ncoresList = [1, 1, 3, 1, 1, 1, 3, 1, 1, 3, 1, 1, 5, 2, 2, 2, 5, 2, 2, 5, 2, 2, 5, 2, 2, 9, 4, 8, 4, 9, 4, 4, 9, 4, 4, 9, 4, 4, 9, 4, 4, 9, 4, 8, 36, 16, 32, 16, 36, 16, 16, 36, 16, 32]
print(ncoresList)
# Which layers use ReLU (1) and which do not (0)
# This is used to determine whether ADC minimum should be optimized separately, if using ReLU-aware limits
reluList =   [1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0]

###
### Quantization loss functions
###

# Optimize the max only
# Use for the ADC range of non-split cores preceding a ReLU
# Use for all DAC ranges
def quantizationError_max(eta,x,Nbits,norm_ord):
	# Clip
	P = 100*(1 - pow(10,eta))
	P = np.clip(P,0,100)

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

# Optimize both the min and the max
# Use for ADC range of split cores and any layers not immediately preceding a ReLU
def quantizationError_minMax(etas,x,Nbits,norm_ord):
	# Clip
	etaMin, etaMax = etas
	P_min = 100*pow(10,etaMin)
	P_max = 100*(1 - pow(10,etaMax))
	P_min = np.clip(P_min,0,100)
	P_max = np.clip(P_max,0,100)

	ADCmin = ncp.percentile(x,P_min)
	ADCmax = ncp.percentile(x,P_max)
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
##  Determine ADC ranges
##
################

candidate = [0, 1, -1]

ADC_limits = np.zeros((len(layerNums),2))
ADC_limits_candidate = np.zeros((len(layerNums),len(candidate),2))

for k in range(len(layerNums)):
	print('Layer '+str(k)+' ('+str(layerNums[k])+')')
	
	x_dist_ADC = ncp.load(profiling_folder+"adc_inputs_layer"+str(layerNums[k])+".npy")
	x_dist_ADC = x_dist_ADC.flatten()

	mean_value = ncp.mean(x_dist_ADC)
	variance_value = ncp.var(x_dist_ADC)

	if useGPU:
		ADC_max = ncp.asnumpy(ncp.max(x_dist_ADC))
		ADC_min = ncp.asnumpy(ncp.min(x_dist_ADC))
	else:
		ADC_max = np.max(x_dist_ADC)
		ADC_min = np.min(x_dist_ADC)

	# Optimize the ADC percentile
	# If ADC inputs are ReLU-aware, layer is not split, and layer has a ReLU, optimize only the max
	# The min will be set to the minimum value in the profiled set which will correspond to a post-ReLU value of zero for the
	# output channel with the most positive bias value
	if reluAwareLimits and ncoresList[k] == 1 and reluList[k] == 1:
		xmin_ADC = ADC_min
		if Nbits > 0:
			eta0 = -2.8
			eta = minimize(quantizationError_max,eta0,args=(x_dist_ADC,Nbits,norm_ord),method='nelder-mead',tol=0.1)
			percentile_ADC = 100*(1-pow(10,eta.x[0]))
			print('    ADC Percentiles:, {:.3f}'.format(percentile_ADC))
			
			if useGPU:
				xmax_ADC = ncp.asnumpy(ncp.percentile(x_dist_ADC,percentile_ADC))
			else:
				xmax_ADC = np.percentile(x_dist_ADC,percentile_ADC)
			ADC_cadidate=[]
			for i in candidate:
				tmp=[]
				if useGPU:
					tmp.append(xmin_ADC)
					tmp.append(ncp.asnumpy(ncp.percentile(x_dist_ADC,percentile_ADC-i*0.05)))
					
				else:
					tmp.append(xmin_ADC)
					tmp.append(np.percentile(x_dist_ADC,percentile_ADC-i*0.05))
					
				ADC_cadidate.append(tmp)
		else:
			xmax_ADC = ADC_max
		clipped = ((ADC_max-xmax_ADC) + (xmin_ADC-ADC_min))/(ADC_max-ADC_min)
		print('    Percentage of ADC range clipped: {:.3f}'.format(clipped*100)+'%')

	else:
		if Nbits > 0:
			etas0 = (-2.8, -2.8)
			eta = minimize(quantizationError_minMax,etas0,args=(x_dist_ADC,Nbits,norm_ord),method='nelder-mead',tol=0.1)
			Pmin_ADC = 100*pow(10,eta.x[0])
			Pmax_ADC = 100*(1-pow(10,eta.x[1]))
			print('    ADC Percentiles: {:.3f}'.format(Pmin_ADC)+', {:.3f}'.format(Pmax_ADC))
			if useGPU:
				xmin_ADC = ncp.asnumpy(ncp.percentile(x_dist_ADC,Pmin_ADC))
				xmax_ADC = ncp.asnumpy(ncp.percentile(x_dist_ADC,Pmax_ADC))
			else:
				xmin_ADC = np.percentile(x_dist_ADC,Pmin_ADC)
				xmax_ADC = np.percentile(x_dist_ADC,Pmax_ADC)	
			
			ADC_cadidate=[]
			for i in candidate:
				tmp=[]
				if useGPU:
					tmp.append(ncp.asnumpy(ncp.percentile(x_dist_ADC,Pmin_ADC+i*0.05)))
					tmp.append(ncp.asnumpy(ncp.percentile(x_dist_ADC,Pmax_ADC-i*0.05)))
					
				else:
					tmp.append(np.percentile(x_dist_ADC,Pmin_ADC+i*0.05))
					tmp.append(np.percentile(x_dist_ADC,Pmax_ADC-i*0.05))
					
				ADC_cadidate.append(tmp)
		else:
			xmin_ADC = ADC_min
			xmax_ADC = ADC_max
		clipped = ((ADC_max-xmax_ADC) + (xmin_ADC-ADC_min))/(ADC_max-ADC_min)
		print('    Percentage of profiled ADC range clipped: {:.3f}'.format(clipped*100)+'%')

	# Set the limits
	ADC_limits[k,:] = np.array([xmin_ADC,xmax_ADC])
	ADC_limits_candidate[k,:] = np.array(ADC_cadidate)
	
	del x_dist_ADC
	del ADC_max
	gc.collect() 
	# print(ADC_limits_candidate)
print("Calibrated ADC limits:")
print(ADC_limits)
print(ADC_limits_candidate)
# Save the generated limits to a npy file
np.save(output_name,ADC_limits)
np.save(output_name_candidate,ADC_limits_candidate)