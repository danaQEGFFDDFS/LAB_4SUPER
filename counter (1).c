#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <strings.h>
#include <cuda_runtime.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); } //ошибки
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

__global__ void matrixCalc(int *a, int *b, int *res, int N) {//cчитываем колонку и столбец

    int col = blockIdx.x * blockDim.x + threadIdx.x;//при инициализации создаются сами,номер колонки номер столбца
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < N && row < N) {
        res[col + row * N] = a[col + row * N] * b[col];//
    }
}


int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("Введите все параметры при запуске программы [размер] [имя файла]");
        exit(0);
    }

    FILE *f;
    f = fopen(argv[2],"r");
    if(!f) {
        printf("ERROR: Error open %s in mode %s \n", argv[2], "r");
        exit(0);
    }
    printf("INFO: File %s in mode %s successfully opened\n", argv[2], "r");

    int resInRows = atoi(argv[1]);

    int** matrix;
    int* arr;
    int** res_matrix;
    int flag = 0;

    arr = (int*) malloc(resInRows * sizeof(int));

    matrix = (int**) malloc(resInRows * sizeof(int*));
    for (int j = 0; j < resInRows; j++) {
        matrix[j] = (int*) malloc(resInRows * sizeof(int));
    }

    res_matrix = (int**) malloc(resInRows * sizeof(int*));
    for (int j = 0; j < resInRows; j++) {
        res_matrix[j] = (int*) malloc(resInRows * sizeof(int));
    }

    int index_row = 0;
    int index_col = 0;

    char data[2];

    char *val;
    val = (char*)malloc(80);

    while(fgets(data, 2, f) != NULL) {

        if (flag == 0) {

            if (strcmp(data, " ") == 0) {
                matrix[index_col][index_row] = atoi(val);
                memset(val, 0, 80);
                index_row++;
                continue;
            }

            if (strcmp(data, "\n") == 0) {

                if (index_row == 0) {
                    flag = 1;
                } else {
                    matrix[index_col][index_row] = atoi(val);
                    index_row = 0;
                    index_col++;
                }
                memset(val, 0, 80);
                continue;
            }

            if (strcmp(data, "\t") == 0) {
                continue;
            }
            sprintf(val, "%s%s", val, data);
            continue;
        }

        if (strcmp(data, " ") == 0) {
            arr[index_row] = atoi(val);
            memset(val, 0, 80);
            index_row++;
            continue;
        }

        if (strcmp(data, "\n") == 0) {
            arr[index_row] = atoi(val);
            continue;
        }

        if (strcmp(data, "\t") == 0) {
            continue;
        }

        sprintf(val, "%s%s", val, data);

        bzero(data, 1);
    }

    fclose(f);
    f = fopen("res.txt","w+b");

    float res_time = 0;
    time_t start = clock();
    int data_counter = 0;

    cudaDeviceProp prop;//инициализазция
    cudaGetDeviceProperties(&prop, 0);

    int *a, *b, *res, *h_a, *h_res;

    size_t bytes = resInRows * resInRows * sizeof(int);
    size_t bytes_res = resInRows * sizeof(int);

    h_a = (int*) malloc(bytes);
    h_res = (int*) malloc(bytes);

    for (int i=0; i < resInRows; i++) {// конвертация матрицы в массив (была двумерная матрица стал массив)
        for (int j=0; j < resInRows; j++) {
            h_a[i*resInRows + j] = matrix[i][j];  // matrix to array
        }
    }

    gpuErrchk(cudaMalloc(reinterpret_cast<void **>(&a), bytes));//выделение памяти на видеокарте
    gpuErrchk(cudaMemcpyAsync(a, h_a, bytes, cudaMemcpyHostToDevice));// а это матрица,копирование массива из оперативки в память видеокарты
    gpuErrchk(cudaMalloc(reinterpret_cast<void **>(&b), bytes_res));
    gpuErrchk(cudaMemcpyAsync(b, arr, bytes_res, cudaMemcpyHostToDevice));// маленький массив

    gpuErrchk(cudaMalloc(reinterpret_cast<void **>(&res), bytes));// выделение памяти для рез. матрицы


    int block_size = 16;
    int grid_size = (resInRows + block_size) / block_size;// непросто выделить память. количестов блоков

    dim3 DimGrid(grid_size, grid_size,1);//создаем переменные грид и блок
    dim3 DimBlock(block_size, block_size,1);

    matrixCalc<<<DimGrid,DimBlock>>>(a, b, res, resInRows);//
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );//потоки синхронизирует

    gpuErrchk( cudaMemcpyAsync(h_res, res, bytes, cudaMemcpyDeviceToHost));// из памяти видеокарты в оперативную, гед h там операвтиная

    time_t end = clock();
    res_time = ((float)(end - start) / 1000000.0F ) * 1000;


    for (int i = 0; i < resInRows;i++) {
        for (int j = 0; j < resInRows - 1; j++) {
            fprintf(f, "%d ", h_res[i * resInRows + j]);
           // printf("%d ", h_res[i * resInRows + j]);
        }
        // array
        fprintf(f, "%d", h_res[i * resInRows + resInRows - 1]);
        //printf("%d", h_res[i * resInRows + resInRows - 1]);
        fprintf(f, "\n");
       // printf("\n");
    }
    printf("Count time: %f ms\n", res_time);
    printf("Count of digits: %d ms\n", data_counter);
    fprintf(f, "Count time: %f ms\n", res_time);
    fprintf(f, "Count of digits: %d ms\n", data_counter);
    fclose(f);
}