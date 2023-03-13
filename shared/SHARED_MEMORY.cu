/*
* Course: High Performance Computing 2021/2022
*
* Lecturer: Francesco Moscato   fmoscato@unisa.it
*
* Group:
* Lamberti      Martina     0622701476  m.lamberti61@studenti.unisa.it
* Salvati       Vincenzo    0622701550  v.salvati10@studenti.unisa.it
* Silvitelli    Daniele     0622701504  d.silvitelli@studenti.unisa.it
* Sorrentino    Alex        0622701480  a.sorrentino120@studenti.unisa.it
*
* Copyright (C) 2021 - All Rights Reserved
*
* This file is part of EsameHPC.
*
* Contest-CUDA is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* Contest-CUDA is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with Contest-CUDA.  If not, see <http://www.gnu.org/licenses/>.
*/

/**
    @file SHARED_MEMORY.cu
*/

// PURPOSE OF THE FILE: Implementation of the RadixSort algorithm by using CUDA shared memory.

#include <stdio.h>
#include <time.h>

#define STARTTIME(id) \
    clock_t start_time_42_##id, end_time_42_##id; \
start_time_42_##id = clock()

#define ENDTIME(id, x) \
end_time_42_##id = clock(); \
x = ((double)(end_time_42_##id - start_time_42_##id)) / CLOCKS_PER_SEC

#define size_of_array 5120

#define thread_per_block 128

/*
* @brief Function to get the largest element from an array.
* @param array    pointer to the vector to be sorted.
*/
unsigned int getMax(unsigned int *array) {
    unsigned int max = array[0],
                 i;

    for (i = 1; i < size_of_array; i++)
        if (array[i] > max)
            max = array[i];

    return max;
}

/**
* @brief Function to execute localsort.
* @param array    pointer to the vector to be sorted.
* @param place    current digit which it is considered (units, tens, hundreds, ...).
*/
void localsort(unsigned int *array, unsigned int place) {
    unsigned int max = (array[0] / place) % 10;
    int i;
    unsigned int *output,
                 *count;

    output = (unsigned int *) malloc((size_of_array + 1) * sizeof(unsigned int));

    for (i = 1; i < size_of_array; i++)
        if (((array[i] / place) % 10) > max)
            max = array[i];

    count = (unsigned int *) malloc((max + 1) * sizeof(unsigned int));

    for (i = 0; i < max; ++i)
        count[i] = 0;
    for (i = 0; i < size_of_array; i++)
        count[(array[i] / place) % 10]++;
    for (i = 1; i < 10; i++)
        count[i] += count[i - 1];
    for (i = size_of_array - 1; i >= 0; i--) {
        output[count[(array[i] / place) % 10] - 1] = array[i];
        count[(array[i] / place) % 10]--;
    }
    for (i = 0; i < size_of_array; i++)
        array[i] = output[i];
}

/*
* @brief Function to perform RadixSort by using CPU.
* @param input_array         pointer to the vector to be sorted.
* @param output_array_CPU    pointer to the sorted vector.
* @param elapsed_from_CPU    elapsed time committed from CPU.
*/
void radix_sort_on_CPU(unsigned int *input_array, unsigned int *output_array_CPU, double *elapsed_from_CPU) {
    // Start timer
    STARTTIME(1);

    unsigned int max = getMax(input_array),
                 place,
                 i;

    for (place = 1; max / place > 0; place *= 10)
        localsort(input_array, place);

    // End timer
    ENDTIME(1, *elapsed_from_CPU);

    for (i = 0; i < size_of_array; i++)
        output_array_CPU[i] = input_array[i];
}

/*
* @brief Kernel to perform cumulative sum per block.
* @param input_array_GPU                  pointer to the vector to be sorted.
* @param cumulative_sum_array_per_block   pointer to the vector which contains cumulative sum for each thread starting from the beginning of the own block.
* @param single_bit                       pointer to the vector which contains the single bit of the numbers.
* @param bit                              the current bit of the iteration taken into account.
*/
__global__ void cumulative_sum_per_block(unsigned int *input_array_GPU, unsigned int *cumulative_sum_array_per_block, unsigned int *single_bit, unsigned int bit) {
    unsigned int dim_block = blockDim.x,
                 current_block = blockIdx.x,
                 start_block = dim_block * current_block,
                 current_thread = threadIdx.x,
                 index_thread = start_block + current_thread,
                 i;

    __shared__ unsigned int shared_single_bit[thread_per_block];
    __shared__ unsigned int shared_cumulative_sum_array_per_block[thread_per_block];

    shared_cumulative_sum_array_per_block[current_thread] = cumulative_sum_array_per_block[index_thread];
    shared_single_bit[current_thread] = ((input_array_GPU[index_thread] >> bit) & 1);

    __syncthreads();

    for (i = 0; i <= current_thread; i++)
        shared_cumulative_sum_array_per_block[current_thread] += shared_single_bit[i];

    single_bit[index_thread] = shared_single_bit[current_thread];
    cumulative_sum_array_per_block[index_thread] = shared_cumulative_sum_array_per_block[current_thread];
}

/*
* @brief Kernel to count previous ones for each thread.
* @param cumulative_sum_array_per_block    pointer to the vector which contains cumulative sum for each thread starting from the beginning of the own block.
* @param cumulative_sum_array_per_grid     pointer to the vector which contains cumulative sum for each thread starting from the beginning of the grid.
*/
__global__ void counting_ones_per_grid(unsigned int *cumulative_sum_array_per_block, unsigned int *cumulative_sum_array_per_grid) {
    unsigned int dim_block = blockDim.x,
                 current_block = blockIdx.x,
                 start_block = dim_block * current_block,
                 current_thread = threadIdx.x,
                 index_thread = start_block + current_thread,
                 i;

    __shared__ unsigned int shared_cumulative_sum_array_per_grid[thread_per_block];

    shared_cumulative_sum_array_per_grid[current_thread] = cumulative_sum_array_per_block[index_thread];

    for (i = 0; i < current_block; i++)
        shared_cumulative_sum_array_per_grid[current_thread] += cumulative_sum_array_per_block[(i * dim_block) + (dim_block - 1)];
    cumulative_sum_array_per_grid[index_thread] = shared_cumulative_sum_array_per_grid[current_thread];
}

/*
* @brief Kernel to perform sorting per bit.
* @param input_array_GPU                  pointer to the vector to be sorted.
* @param result_array                     pointer to the sorted vector.
* @param cumulative_sum_array_per_grid    pointer to the vector which contains cumulative sum for each thread starting from the beginning of the grid.
* @param single_bit                       pointer to the vector which contains the single bit considered from each thread.
*/
__global__ void sorting_per_bit(unsigned int *input_array_GPU, unsigned int *result_array, unsigned int *cumulative_sum_array_per_grid, unsigned int *single_bit) {
    unsigned int dim_block = blockDim.x,
                 current_block = blockIdx.x,
                 start_block = dim_block * current_block,
                 current_thread = threadIdx.x,
                 index_thread = start_block + current_thread;

    if (single_bit[index_thread])
        result_array[cumulative_sum_array_per_grid[index_thread] - 1 + size_of_array - cumulative_sum_array_per_grid[size_of_array - 1]] = input_array_GPU[index_thread];
    else
        result_array[index_thread - cumulative_sum_array_per_grid[index_thread]] = input_array_GPU[index_thread];
}

/*
* @brief Function to perform time from milliseconds to seconds.
* @param elapsed_in_ms    elapsed time in milliseconds.
*/
float perform_milliseconds_to_seconds(float elapsed_in_ms) {
    return elapsed_in_ms / 1000.f;
}

/*
* @brief Function to count flops.
* @param number_of_all_blocks    number of all grid's blocks into the GPU.
*/
unsigned int count_flops(unsigned int number_of_all_blocks) {
    unsigned int flops = 0,
                 i;

    for (i = 0; i < thread_per_block; i++)
        flops += i + 3;
    for (i = 0; i < number_of_all_blocks; i++)
        flops += 6;
    flops += 13;

    return size_of_array * 32 * flops;
}

/*
* @brief Function to perform Mflops.
* @param elapsed_in_ms           elapsed time in milliseconds.
* @param number_of_all_blocks    number of all grid's blocks into the GPU.
*/
float perform_Mflop_per_sec(float elapsed_in_ms, unsigned int number_of_all_blocks) {
    float flops = count_flops(number_of_all_blocks) / (1000.f * 1000.f);
    return (flops / perform_milliseconds_to_seconds(elapsed_in_ms));
}

/*
* @brief Function to check error from CUDA operation.
* @param cuda_error     CUDA error.
* @param show_result    print eventual error into the console.
*/
void check_error_from(cudaError_t cuda_error, bool show_result = true) {
    if (show_result)
        if (cuda_error != cudaSuccess) {
            fprintf(stderr, "%s\n", cudaGetErrorString(cuda_error));
            exit(1);
        }
}

/*
* @brief Function to perform RadixSort by using GPU.
* @param input_array             pointer to the vector to be sorted.
* @param output_array_GPU        pointer to the sorted vector.
* @param number_of_all_blocks    number of all grid's blocks into the GPU.
* @param mflop_per_sec           Mflops.
* @param elapsed_from_GPU        elapsed time committed from GPU.
*/
void radix_sort_on_GPU(unsigned int *input_array, unsigned int *output_array_GPU, unsigned int *number_of_all_blocks, float *mflop_per_sec, float *elapsed_from_GPU) {
    // Reset stack CUDA error
    check_error_from(cudaGetLastError(), false);

    // Init variables
    cudaEvent_t start,
                stop;

    float elapsed_in_ms;

    unsigned int size_of_array_in_bytes = size_of_array * sizeof(unsigned int),
                 bit;

    unsigned int *input_array_GPU,
                 *single_bit,
                 *cumulative_sum_array_per_block,
                 *cumulative_sum_array_per_grid,
                 *result_array;

    *number_of_all_blocks = ((size_of_array / thread_per_block) * thread_per_block < size_of_array) ? ((size_of_array / thread_per_block) + 1) : (size_of_array / thread_per_block);
    dim3 dim_grid(*number_of_all_blocks);
    dim3 dim_block(thread_per_block);

    check_error_from(cudaEventCreate(&start));
    check_error_from(cudaEventCreate(&stop));

    check_error_from(cudaMalloc(&input_array_GPU, size_of_array_in_bytes));
    check_error_from(cudaMalloc(&single_bit, size_of_array_in_bytes));
    check_error_from(cudaMalloc(&cumulative_sum_array_per_block, size_of_array_in_bytes));
    check_error_from(cudaMalloc(&cumulative_sum_array_per_grid, size_of_array_in_bytes));
    check_error_from(cudaMalloc(&result_array, size_of_array_in_bytes));

    check_error_from(cudaMemcpy(input_array_GPU, input_array, size_of_array_in_bytes, cudaMemcpyHostToDevice));

    // Start timer
    check_error_from(cudaEventRecord(start, 0));

    // Run kernels
    for (bit = 0; bit < 32; bit++) {
        // Reset "cumulative_sum_array_per_block"
        check_error_from(cudaMemset(cumulative_sum_array_per_block, 0, size_of_array_in_bytes));

        cumulative_sum_per_block<<<dim_grid, dim_block>>>(input_array_GPU,
                                                          cumulative_sum_array_per_block,
                                                          single_bit,
                                                          bit);
        cudaDeviceSynchronize();
        check_error_from(cudaGetLastError());

        counting_ones_per_grid<<<dim_grid, dim_block>>>(cumulative_sum_array_per_block, 
                                                        cumulative_sum_array_per_grid);
        cudaDeviceSynchronize();
        check_error_from(cudaGetLastError());

        sorting_per_bit<<<dim_grid, dim_block>>>(input_array_GPU,
                                                 result_array,
                                                 cumulative_sum_array_per_grid,
                                                 single_bit);
        cudaDeviceSynchronize();
        check_error_from(cudaGetLastError());

        // Set new "input_array_GPU" by sub ordered array "result_array"
        check_error_from(cudaMemcpy(input_array_GPU, result_array, size_of_array_in_bytes, cudaMemcpyDeviceToDevice));
    }

    // End timer
    check_error_from(cudaEventRecord(stop, 0));
    check_error_from(cudaEventSynchronize(stop));

    // Perform time
    check_error_from(cudaEventElapsedTime(&elapsed_in_ms, start, stop));
    *elapsed_from_GPU = perform_milliseconds_to_seconds(elapsed_in_ms);

    // Perform Mflops
    *mflop_per_sec = perform_Mflop_per_sec(elapsed_in_ms, *number_of_all_blocks);

    // Result GPU
    check_error_from(cudaMemcpy(output_array_GPU, result_array, size_of_array_in_bytes, cudaMemcpyDeviceToHost));

    // Free memory on GPU
    check_error_from(cudaEventDestroy(start));
    check_error_from(cudaEventDestroy(stop));
    check_error_from(cudaFree(input_array_GPU));
    check_error_from(cudaFree(single_bit));
    check_error_from(cudaFree(cumulative_sum_array_per_block));
    check_error_from(cudaFree(cumulative_sum_array_per_grid));
    check_error_from(cudaFree(result_array));
}

/*
* @brief Function to read an unsorted array from a file.
* @param array    pointer to the vector to be sorted.
*/
void read_input_from_file(unsigned int *array) {
    FILE *file = fopen("random_numbers.txt", "r");;
    char buffer[10];
    int i = 0;
    if (file == NULL) {
        perror("Error opening file");
        exit(1);
    }
    while (i < size_of_array) {
        fgets(buffer, 10, file);
        array[i++] = (unsigned int) atoi(buffer);
    }
    fclose(file);
}

/*
* @brief Function to check if file exists.
* @param filename    directory.
*/
bool file_exist(char *filename) {
    FILE *file;
    if (file = fopen(filename, "r")) {
        fclose(file);
        return true;
    }
    return false;
}

/*
* @brief Function to make a csv of the measures.
* @param number_of_all_blocks    number of all grid's blocks into the GPU.
* @param mflop_per_sec           Mflops.
* @param elapsed_from_GPU        elapsed time committed from GPU.
* @param elapsed_from_CPU        elapsed time committed from CPU.
*/
void make_csv_of(unsigned int number_of_all_blocks, float mflop_per_sec, float elapsed_from_GPU, double elapsed_from_CPU) {
    FILE *file;
    char thread_per_block_char[] = "";
    itoa(thread_per_block, thread_per_block_char, 10);

    char filename[] = "../measures/shared_measures_";
    strcat(filename, thread_per_block_char);
    strcat(filename, ".csv");

    if (!file_exist(filename)) {
        file = fopen(filename, "w");
        fprintf(file,"size_of_array, thread_per_block, number_of_all_blocks, Mflops, elapsed_from_GPU, elapsed_from_CPU\n");
    } else 
        file = fopen(filename, "a");

    fprintf(file, "%d, %d, %d, %f, %f, %lf\n", size_of_array, thread_per_block, number_of_all_blocks, mflop_per_sec, elapsed_from_GPU, elapsed_from_CPU);

    fclose(file);
}

/*
* @brief Test function to detect errors between CPU and GPU.
* @param output_array_CPU    pointer to the sorted vector by CPU.
* @param output_array_GPU    pointer to the sorted vector by GPU.
*/
unsigned int errors_in_comparison_between(unsigned int *output_array_CPU, unsigned int *output_array_GPU) {
    unsigned int number_of_errors = 0,
                 i;

    for (i = 0; i < size_of_array; i++) {
        if (output_array_CPU[i] != output_array_GPU[i]) {
            number_of_errors++;
            printf("(CPU) %d != %d (GPU)\tposition=%d\n", output_array_CPU[i], output_array_GPU[i], i);
        }
    }

    if (number_of_errors > 0)
        printf("\nNumber of errors: %d\n", number_of_errors);

    return number_of_errors;
}

/*
* @brief Test function to compare output between CPU and GPU.
* @param output_array_CPU    pointer to the sorted vector by CPU.
* @param output_array_GPU    pointer to the sorted vector by GPU.
*/
void compare_output_between_CPU_and_GPU(unsigned int *output_array_CPU, unsigned int *output_array_GPU) {
    if (!errors_in_comparison_between(output_array_CPU, output_array_GPU))
        printf("\nTEST PASSED!\n");
    else {
        printf("\nTEST FAILED!\n");
        exit(1);
    }
}

int main(int argc, char **argv) {
    // Init variables
    double elapsed_from_CPU = 0.0;

    float elapsed_from_GPU = 0.0,
          mflop_per_sec = 0.0;

    unsigned int size_of_array_in_bytes = size_of_array * sizeof(unsigned int),
                 number_of_all_blocks;

    unsigned int *input_array,
                 *output_array_CPU,
                 *output_array_GPU;

    input_array = (unsigned int *) malloc(size_of_array_in_bytes);
    output_array_CPU = (unsigned int *) malloc(size_of_array_in_bytes);
    output_array_GPU = (unsigned int *) malloc(size_of_array_in_bytes);

    read_input_from_file(input_array);

    // Perform RadixSort
    radix_sort_on_CPU(input_array, output_array_CPU, &elapsed_from_CPU);
    radix_sort_on_GPU(input_array, output_array_GPU, &number_of_all_blocks, &mflop_per_sec, &elapsed_from_GPU);
    
    // Results
    compare_output_between_CPU_and_GPU(output_array_CPU, output_array_GPU);
    make_csv_of(number_of_all_blocks, mflop_per_sec, elapsed_from_GPU, elapsed_from_CPU);
}