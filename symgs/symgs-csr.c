#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <cura_runtime.h>

double get_time() // function to get the time of day in seconds
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

// Reads a sparse matrix and represents it using CSR (Compressed Sparse Row) format
void read_matrix(int **row_ptr, int **col_ind, float **values, float **matrixDiagonal, const char *filename, int *num_rows, int *num_cols, int *num_vals)
{
    //verify if the file can be opened
    int err;
    FILE *file = fopen(filename, "r");
    if (file == NULL)
    {
        fprintf(stdout, "File cannot be opened!\n");
        exit(0);
    }
    // Get number of rows, columns, and non-zero values
    if(fscanf(file, "%d %d %d\n", num_rows, num_cols, num_vals)==EOF)
        printf("Error reading file");

    int *row_ptr_t, *col_ind_t;
    float *values_t, *matrixDiagonal_t;
    cudaMallocManaged(&row_ptr_t ,(*num_rows + 1) * sizeof(int));
    cudaMallocManaged(&col_ind_t ,(*num_vals * sizeof(int)));
    cudaMallocManaged(&values_t ,(*num_vals * sizeof(float)));
    cudaMallocManaged(&matrixDiagonal_t ,(*num_rows * sizeof(float)));

    // Collect occurances of each row for determining the indices of row_ptr
    int *row_occurances = (int *)malloc(*num_rows * sizeof(int));
    for (int i = 0; i < *num_rows; i++)
    {
        row_occurances[i] = 0;
    }

    int row, column;
    float value;
    while (fscanf(file, "%d %d %f\n", &row, &column, &value) != EOF)
    {
        // Subtract 1 from row and column indices to match C format
        row--;
        column--;
        row_occurances[row]++;
    }

    // Set row_ptr
    int index = 0;
    for (int i = 0; i < *num_rows; i++)
    {
        row_ptr_t[i] = index;
        index += row_occurances[i];
    }
    row_ptr_t[*num_rows] = *num_vals;
    free(row_occurances);

    // Set the file position to the beginning of the file
    rewind(file);

    // Read the file again, save column indices and values
    for (int i = 0; i < *num_vals; i++)
    {
        col_ind_t[i] = -1;
    }

    if(fscanf(file, "%d %d %d\n", num_rows, num_cols, num_vals)==EOF)
        printf("Error reading file");
    
    int i = 0, j = 0;
    while (fscanf(file, "%d %d %f\n", &row, &column, &value) != EOF)
    {
        row--;
        column--;

        // Find the correct index (i + row_ptr_t[row]) using both row information and an index i
        while (col_ind_t[i + row_ptr_t[row]] != -1)
        {
            i++;
        }
        col_ind_t[i + row_ptr_t[row]] = column;
        values_t[i + row_ptr_t[row]] = value;
        if (row == column)
        {
            matrixDiagonal_t[j] = value;
            j++;
        }
        i = 0;
    }
    fclose(file);
    *row_ptr = row_ptr_t;
    *col_ind = col_ind_t;
    *values = values_t;
    *matrixDiagonal = matrixDiagonal_t;
}

// CPU implementation of SYMGS using CSR, DO NOT CHANGE THIS
void symgs_csr_sw(const int *row_ptr, const int *col_ind, const float *values, const int num_rows, float *x, float *matrixDiagonal)
{

    // forward sweep
    for (int i = 0; i < num_rows; i++)
    {
        float sum = x[i];
        const int row_start = row_ptr[i];
        const int row_end = row_ptr[i + 1];
        float currentDiagonal = matrixDiagonal[i]; // Current diagonal value

        for (int j = row_start; j < row_end; j++)
        {
            sum -= values[j] * x[col_ind[j]];
        }

        sum += x[i] * currentDiagonal; // Remove diagonal contribution from previous loop

        x[i] = sum / currentDiagonal;
    }

    // backward sweep
    for (int i = num_rows - 1; i >= 0; i--)
    {
        float sum = x[i];
        const int row_start = row_ptr[i];
        const int row_end = row_ptr[i + 1];
        float currentDiagonal = matrixDiagonal[i]; // Current diagonal value

        for (int j = row_start; j < row_end; j++)
        {
            sum -= values[j] * x[col_ind[j]];
        }
        sum += x[i] * currentDiagonal; // Remove diagonal contribution from previous loop

        x[i] = sum / currentDiagonal;
    }
}

// GPU implementation of SYMGS
__global__ void symgsGPU(const int *row_ptr, const int *col_ind, const float *values, const int num_rows, float *x, float *matrixDiagonal)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    //forward sweep
    //cpu code that must be modified and optimized
    if (i<N)
    {
        float sum = x[i];
        const int row_start = row_ptr[i];
        const int row_end = row_ptr[i + 1];
        float currentDiagonal = matrixDiagonal[i]; // Current diagonal value

        for (int j = row_start; j < row_end; j++)
        {
            sum -= values[j] * x[col_ind[j]];
        }

        sum += x[i] * currentDiagonal; // Remove diagonal contribution from previous loop

        x[i] = sum / currentDiagonal;
    }
    //back sweep

}


int main(int argc, const char *argv[])
{

    if (argc != 2)
    {
        printf("Usage: ./exec matrix_file");
        return 0;
    }

    int *row_ptr, *col_ind, num_rows, num_cols, num_vals;
    float *values;
    float *matrixDiagonal;

    const char *filename = argv[1];

    double start_cpu, end_cpu, start_gpu, end_gpu;

    read_matrix(&row_ptr, &col_ind, &values, &matrixDiagonal, filename, &num_rows, &num_cols, &num_vals);
    float *x = (float *)malloc(num_rows * sizeof(float));

    // Generate a random vector
    srand(time(NULL));
    for (int i = 0; i < num_rows; i++)
    {
        x[i] = (rand() % 100) / (rand() % 100 + 1); // the number we use to divide cannot be 0, that's the reason of the +1
    }
    // Compute in sw
    start_cpu = get_time();
    symgs_csr_sw(row_ptr, col_ind, values, num_rows, x, matrixDiagonal);
    end_cpu = get_time();


    // Compute in GPU
    start_gpu = get_time();
    
    // inputs to move from RAM to VRAM (row_ptr, col_ind, values, num_rows, x, matrixDiagonal)
    // device=GPU
    int d_row_ptr[num_rows+1], d_col_ind[num_vals];
    float d_values[num_vals], d_matrixDiagonal[num_rows];
    cudaMemcpy(d_row_ptr, row_ptr, (num_rows+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_ind, col_ind, num_vals*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, values,num_vals*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_matrixDiagonal, matrixDiagonal, num_rows*sizeof(int), cudaMemcpyHostToDevice);
    
    // kernel invocation
    dim3 threadsPerBlock(N, N);
    dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);
    symgsGPU<<<threadsPerBlock, numBlocks>>>(d_row_ptr, d_col_ind, d_values, d_matrixDiagonal);
    
    //copy back data from VRAM to RAM
    cudaMemcpy(row_ptr, d_row_ptr, (num_rows+1)*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(col_ind,d_col_ind,  num_vals*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(values, d_values, num_vals*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(matrixDiagonal, d_matrixDiagonal, num_rows*sizeof(int), cudaMemcpyDeviceToHost);

    end_gpu = get_time();

    // Print time
    printf("SYMGS Time CPU: %.10lf\n", end_cpu - start_cpu);
    printf("SYMGS Time CPU: %.10lf\n", end_gpu - start_gpu);

    // Free
    cudaFree(row_ptr);
    cudaFree(col_ind);
    cudaFree(values);
    cudaFree(matrixDiagonal);

    return 0;
}