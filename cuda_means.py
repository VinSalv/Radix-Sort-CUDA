#
# Course: High Performance Computing 2021/2022
#
# Lecturer: Francesco Moscato   fmoscato@unisa.it
#
# Group:
# Lamberti      Martina     0622701476  m.lamberti61@studenti.unisa.it
# Salvati       Vincenzo    0622701550  v.salvati10@studenti.unisa.it
# Silvitelli    Daniele     0622701504  d.silvitelli@studenti.unisa.it
# Sorrentino    Alex        0622701480  a.sorrentino120@studenti.unisa.it
#
# Copyright (C) 2021 - All Rights Reserved
#
# This file is part of EsameHPC.
#
# Contest-CUDA is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Contest-CUDA is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Contest-CUDA.  If not, see <http://www.gnu.org/licenses/>.
#

#
#   @file cuda_means.py
#

# PURPOSE OF THE FILE: Calculate the means of each measure.

import pandas as pd

def init(memory_type, thread_per_block):
	df = pd.read_csv ("measures/" + memory_type + "_measures_" + str(thread_per_block) + ".csv")
	return df

def calculate_mean(df):
	mean_mflop = df[' Mflops'].mean()
	mean_gpu = df[' elapsed_from_GPU'].mean()
	mean_cpu = df[' elapsed_from_CPU'].mean()
	return mean_mflop, mean_gpu, mean_cpu

if __name__ == '__main__':
	memory_type_list = ["global", "shared", "texture"]
	thread_per_block_list = [64, 128, 256, 512, 1024]
	for memory_type in memory_type_list:
		print ("Mean of 200 iteration of:\n")
		for thread_per_block in thread_per_block_list:
			df = init(memory_type, thread_per_block)
			mflop, gpu, cpu = calculate_mean(df)
			print ("memory: " + memory_type + " memory - thread per block: " + str(thread_per_block))
			print(" - Mflops: " + str(mflop))
			print(" - Elapsed time GPU: " + str(gpu))
			print(" - Elapsed time CPU: " + str(cpu) + "\n")
			