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
import os

navcim_dir = os.getenv('NAVCIM_DIR')
# Use this file to determine ADC limits for systems with weight bit slicing
# Express ADC limits instead of power of 2 by which uncalibrated range is clipped
# This constraint makes it simple to combine bit slices, even if they have different ADC limits
parser = argparse.ArgumentParser(description='PyTorch CIFAR-X Example')
parser.add_argument('--NrowsMax', type=int, default=64, help='NrowsMax')
parser.add_argument('--NcolsMax', type=int, default=64, help='NcolsMax')
parser.add_argument('--cell_bits', type=int, default=2, help='cell_bits')
parser.add_argument('--gpu', type=int, default=0, help='NcolsMax')
args = parser.parse_args()

Nslices = np.ceil(7 / args.cell_bits).astype(int)
NrowsMax = args.NrowsMax
NcolsMax = args.NcolsMax
Wbits_slice = int(8 / Nslices)
xbar_style = "BALANCED"
input_bitslicing = False

##### Folder where profiled values are stored
##### Make sure the correct file is selected that is consistent with the options above!

ibit_msg = ("_ibits" if input_bitslicing else "")
profiling_folder = f"{navcim_dir}/cross-sim/applications/dnn/inference/adc/profiled_adc_inputs/imagenet_MobileNetV2_"+\
			str(NrowsMax)+"rows_"+str(NcolsMax)+"cols_"+str(Nslices)+"slices_"+xbar_style+ibit_msg+"/"

# npy output file path
output_name = f"{navcim_dir}/cross-sim/applications/dnn/inference/adc/adc_limits/examples/adc_limits_MobileNetV2_"+\
			str(NrowsMax)+"rows_"+str(NcolsMax)+"cols_"+str(Nslices)+"slices_"+xbar_style+ibit_msg+".npy"

# Calibrated limits must include this percentile on both sides
# e.g. 99.99 means 99.99% and 0.01% percentile will be included in the range
# Warning: for some layers, a vast majority of ADC inputs may be zero and pct may have to be raised to
# get a meaningful range
pct = 99.99

# Enable GPU
useGPU = True
if useGPU:
	import cupy as cp
	gpu_num = 7
	os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
	ncp = cp
else:
	ncp = np

###
### ResNet50 specific parameters
###

# Layer nums for ResNet conv and dense layers
layerNums = [4, 14, 19, 29, 34, 44, 53, 63, 68, 78, 87, 97, 106, 116, 121, 131, 140, 150, 159, 169, 178, 188, 193, 203, 212, 222, 231, 241, 246, 256, 265, 275, 284, 294, 299, 310] 


NrowsList_unsplit = np.array([27, 288, 32, 16, 864, 96, 24, 1296, 144, 24, 1296, 144, 32, 1728, 192, 32, 1728, 192, 32, 1728, 192, 64, 3456, 384, 64, 3456, 384, 64, 3456, 384, 64, 3456, 384, 96, 5184, 576, 96, 5184, 576, 96, 5184, 576, 160, 8640, 960, 160, 8640, 960, 160, 8640, 960, 320, 1280])

# Get the actual number of rows
NrowsList = np.zeros(len(layerNums),dtype=int)
for k in range(len(layerNums)):
	if NrowsList_unsplit[k] <= NrowsMax:
		NrowsList[k] = NrowsList_unsplit[k]
	else:
		Ncores = (NrowsList_unsplit[k]-1)//NrowsMax + 1
		if NrowsList_unsplit[k] % Ncores == 0:
			NrowsList[k] = (NrowsList_unsplit[k] // Ncores).astype(int)
		else:
			Nrows1 = np.round(NrowsList_unsplit[k] / Ncores)
			Nrows2 = NrowsList_unsplit[k] - (Ncores-1)*Nrows1
			if Nrows1 != Nrows2:
				NrowsList[k] = np.maximum(Nrows1,Nrows2).astype(int)

################
##
##  Determine ADC ranges
##
################
candidate = [0, 1, -1]
ADC_limits = np.zeros((len(layerNums),len(candidate),Nslices,2))

# To display some clipping statistics only (not needed to get limits)
percentile_factors = np.zeros((len(layerNums),Nslices))

print('# rows: '+str(NrowsMax))
print('Percentile: {:.4f}'.format(pct))


for k in range(len(layerNums)):

	if xbar_style == "OFFSET":
		# Bring # rows to nearest power of 2
		ymax = pow(2,np.round(np.log2(NrowsList[k])))
		# Correct to make level separation a multiple of the min cell current
		ymax *= pow(2,Wbits_slice)/(pow(2,Wbits_slice)-1)
		Nbits_in = 8
		if k == 0:
			ymax *= pow(2,Nbits_in-1)/(pow(2,Nbits_in-1)-1)
		else:
			ymax *= pow(2,Nbits_in)/(pow(2,Nbits_in)-1)
	else:
		ymax = NrowsList[k]


	for i_slice in range(Nslices):
		if Nslices == 1:
			x_dist_ADC_i = ncp.load(profiling_folder+"adc_inputs_layer"+str(layerNums[k])+".npy")
		else:
			x_dist_ADC_i = ncp.load(profiling_folder+"adc_inputs_layer"+str(layerNums[k])+"_slice"+str(i_slice)+".npy")
		x_dist_ADC_i = x_dist_ADC_i.flatten()		
		x_dist_ADC_i /= ymax
		pct_k = pct

		# ADC inputs can be signed: always true for balanced, also true for first layer (which has signed inputs)
		if k == 0 or xbar_style == "BALANCED":
			p_neg = ncp.percentile(x_dist_ADC_i,100-pct_k)
			p_pos = ncp.percentile(x_dist_ADC_i,pct_k)
			p_out = np.maximum(np.abs(p_neg),np.abs(p_pos))
			clip_power_k = np.floor(np.log2(1/p_out)).astype(int)
		else:
			p_out = ncp.percentile(x_dist_ADC_i,pct_k)
			clip_power_k = np.floor(np.log2(1/p_out)).astype(int)

		if clip_power_k > 25 and pct < 99.999:
			print("Redo with higher pct")
			pct_k = 99.999
			if k == 0 or xbar_style == "BALANCED":
				p_neg = ncp.percentile(x_dist_ADC_i,100-pct_k)
				p_pos = ncp.percentile(x_dist_ADC_i,pct_k)
				p_out = np.maximum(np.abs(p_neg),np.abs(p_pos))
				clip_power_k = np.floor(np.log2(1/p_out)).astype(int)
			else:
				p_out = ncp.percentile(x_dist_ADC_i,pct_k)
				clip_power_k = np.floor(np.log2(1/p_out)).astype(int)

		print("Layer "+str(layerNums[k])+" ("+str(NrowsList[k])+" rows), slice "+str(i_slice)+" clip power: "+str(clip_power_k)+" bits")
		if k == 0 or xbar_style == "BALANCED":
			ADC_limits[k,0,i_slice,0] = -ymax / 2**clip_power_k
			ADC_limits[k,0,i_slice,1] = ymax / 2**clip_power_k
		else:
			ADC_limits[k,0,i_slice,0] = 0
			ADC_limits[k,0,i_slice,1] = ymax / 2**clip_power_k
		
		percentile_factors[k,i_slice] = p_out

print("Calibrated ADC limits:")
print(ADC_limits)

# Save the generated limits to a npy file
np.save(output_name,ADC_limits)