# Radix Sort - CUDA

Course: High Performance Computing 2021/2022
Lecturer: Francesco Moscato fmoscato@unisa.it

Group:

- Lamberti Martina 0622701476 m.lamberti61@studenti.unisa.it
- Salvati Vincenzo 0622701550 v.salvati10@studenti.unisa.it
- Silvitelli Daniele 0622701504 d.silvitelli@studenti.unisa.it
- Sorrentino Alex 0622701480 a.sorrentino120@studenti.unisa.it

## Folders

-  `global`: contains the script GLOBAL_MEMORY.cu;
-  `shared`: contains the script SHARED_MEMORY.cu;
-  `texture`: contains the script TEXTURE_MEMORY.cu;
-  `measures`: contains the measures of global, shared and texture memory <b>(DO NOT DELETE `measures` FOLDER!!)</b>

## Common files

These files have been made to generate measures and extract means of them:

-  `cuda_execute.bash` --> bash script that runs .cu files 200 times
-  `cuda_means.py` --> python script that calculates means of the measures
-  `random_numbers.txt` --> text file contains random numbers to be sorted

## How to run

### Single program
Each version of this program is executed from the local GPU:
1.  `nvcc .\name_of_program.cu -o .\name_of_program` to compile the program;
2.  `./name_of_program` to execute the program.


### All programs
To run the bash file, after any change, <b>it is necessary previously to compile global, shared and texture version</b>.
1. `bash cuda_execute.bash`

So, to extract the mean of the measure and to print them on the terminal (paying attenction to the set correctly <b>memory_type_list</b> and <b>thread_per_block_list</b> parameters since they depend from the content of `measures` folder):

2. `python3 cuda_means.py`