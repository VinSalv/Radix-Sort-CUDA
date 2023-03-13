#!/bin/bash

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
#   @file cuda_execute.bash
#

# PURPOSE OF THE FILE: Automation for executing 200 times GLOBAL_MEMORY.exe, SHARED_MEMORY.exe and TEXTURE_MEMORY.exe.

for i in {0..199}; do  (cd global/ && ./GLOBAL_MEMORY.exe) 1>> Error ; done

for i in {0..199}; do  (cd shared/ && ./SHARED_MEMORY.exe) 1>> Error ; done

for i in {0..199}; do  (cd texture/ && ./TEXTURE_MEMORY.exe) 1>> Error ; done
