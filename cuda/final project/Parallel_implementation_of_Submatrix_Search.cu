#include <stdio.h>
#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <mpi.h>
#include <stddef.h>

#ifndef NUM_PICS
#define NUM_PICS 2
#endif

#define TAG_WORK 1
#define TAG_DONE 2

// Structure for Picture or Object
typedef struct {
    int id;           // ID of Picture/Object
    int size;         // Size (N for Picture, M for Object)
    int** data;       // 2D array for matrix
} Matrix;

// Structure for Result
typedef struct {
    int pic_id;     // Picture ID
    int obj_id;     // Object ID (-1 if no match)
    int i, j;       // Position (I,J) (-1,-1 if no match)
} Result;


// CUDA kernel for vector addition
__global__ void find_Object_in_picture(Matrix* picture, Matrix * object,int objsize, Result* results) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;  // row
    int j = blockIdx.x * blockDim.x + threadIdx.x;  // col

    for (int index =0; index< objsize; index++){
        // printf("index=%d objsize=%d (i,j)=(%d,%d) \n", index,object[index].size,i,j);
        if (i + object[index].size <= picture->size && j + object[index].size <= picture->size) {
            float sum = 0.0f;
           
            
            
             
            for (int oi = 0; oi < object[index].size; oi++) {
                for (int oj = 0; oj < object[index].size; oj++) {
                    int p = picture->data[i + oi][j + oj];
                    int o = object[index].data[oi][oj];
                    sum += abs((p - o) /p);
                }
            }
          
            if (sum < 0.1) { // Assuming a threshold of 0.1 for a match
                // for (int oi = 0; oi < object[index].size; oi++) {
                //     for (int oj = 0; oj < object[index].size; oj++) {
                //         printf("picture.data[%d][%d]=%d \n", i + oi, j + oj, picture->data[i + oi][j + oj]);
                //         printf("object.data[%d][%d]=%d \n", oi, oj, object[index].data[oi][oj]);
                //     }
                // }
                results->pic_id = picture->id;
                results->obj_id = object[index].id;
                results->i = i;
                results->j = j;
                // printf("picture.id ttt =%d (i,j)=(%d,%d) obj id=%d\n", results->pic_id,results->i,results->j, results->obj_id);
                break; // Exit after finding the first match
            }     
        }     
    }    
}

MPI_Datatype mpi_result_type;

void create_mpi_result_type() {
    int blocklengths[4] = {1, 1, 1, 1}; 
    MPI_Aint offsets[4];
    MPI_Datatype types[4] = {MPI_INT, MPI_INT, MPI_INT, MPI_INT};

    offsets[0] = offsetof(Result, pic_id);
    offsets[1] = offsetof(Result, obj_id);
    offsets[2] = offsetof(Result, i);
    offsets[3] = offsetof(Result, j);

    MPI_Type_create_struct(4, blocklengths, offsets, types, &mpi_result_type);
    MPI_Type_commit(&mpi_result_type);
}


cudaError_t allocateObjectsOnDevice(Matrix* h_objects, int num_objs, Matrix** d_objects, int*** d_rows_obj) {
    cudaError_t err;

    // Allocate array of Matrix structures on the device
    err = cudaMalloc((void**)d_objects, num_objs * sizeof(Matrix));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n",
                cudaGetErrorString(err));
        return err;
    }

    // Allocate array of row pointers for all objects
    err = cudaMalloc((void**)d_rows_obj, num_objs * sizeof(int*));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device row pointers array (error code %s)!\n",
                cudaGetErrorString(err));
        return err;
    }

    // Copy each object's data to device
    for (int i = 0; i < num_objs; i++) {
        Matrix temp = h_objects[i];
        int** d_data;

        // Allocate row pointers for the current Matrix
        err = cudaMalloc((void**)&d_data, temp.size * sizeof(int*));
        if (err != cudaSuccess) {
            fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n",
                    cudaGetErrorString(err));
            return err;
        }

        // Allocate and copy each row
        for (int j = 0; j < temp.size; j++) {
            int* d_row;
            err = cudaMalloc((void**)&d_row, temp.size * sizeof(int));
            if (err != cudaSuccess) {
                fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n",
                        cudaGetErrorString(err));
                return err;
            }

            err = cudaMemcpy(d_row, temp.data[j], temp.size * sizeof(int), cudaMemcpyHostToDevice);
            if (err != cudaSuccess) {
                fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n",
                        cudaGetErrorString(err));
                return err;
            }

            err = cudaMemcpy(&d_data[j], &d_row, sizeof(int*), cudaMemcpyHostToDevice);
            if (err != cudaSuccess) {
                fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n",
                        cudaGetErrorString(err));
                return err;
            }
        }

        // Update the Matrix structure with the device row pointers
        temp.data = d_data;
        err = cudaMemcpy(&(*d_objects)[i], &temp, sizeof(Matrix), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n",
                    cudaGetErrorString(err));
            return err;
        }

        // Store the row pointers array for this object
        err = cudaMemcpy(&(*d_rows_obj)[i], &d_data, sizeof(int*), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            fprintf(stderr, "Failed to copy row pointers array (error code %s)!\n",
                    cudaGetErrorString(err));
            return err;
        }
    }

    return cudaSuccess;
}


cudaError_t allocateMatrixOnDevice(Matrix h_mat, Matrix** d_picture, int*** d_rows) {
    cudaError_t err;

    // Allocate Matrix structure on the device  
    err = cudaMalloc((void**)d_picture, sizeof(Matrix));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate d_picture (error %s)!\n",
                cudaGetErrorString(err));
        return err;
    }
    // Allocate array of row pointers on the device
    err = cudaMalloc((void**)d_rows, h_mat.size * sizeof(int*));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate d_rows (error %s)!\n",
                cudaGetErrorString(err));
        return err;
    }
    // Allocate and copy each row to device
    for (int r = 0; r < h_mat.size; r++) {
        int* d_row;
        err = cudaMalloc((void**)&d_row, h_mat.size * sizeof(int));
        if (err != cudaSuccess) {
            fprintf(stderr, "Failed to allocate d_row (error %s)!\n",
                    cudaGetErrorString(err));
            return err;
        }

        err = cudaMemcpy(d_row, h_mat.data[r],
                        h_mat.size * sizeof(int),
                        cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            fprintf(stderr, "Failed to copy row (error %s)!\n",
                    cudaGetErrorString(err));
            return err;
        }

    
        err = cudaMemcpy(&(*d_rows)[r], &d_row, sizeof(int*),
                        cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            fprintf(stderr, "Failed to set row pointer (error %s)!\n",
                    cudaGetErrorString(err));
            return err;
        }
    }

    // Create a temporary Matrix struct to hold device pointers
    Matrix h_mat_dev = h_mat;
    h_mat_dev.data = *d_rows;

    err = cudaMemcpy(*d_picture, &h_mat_dev, sizeof(Matrix),
                    cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy Matrix struct (error %s)!\n",
                cudaGetErrorString(err));
        return err;
    }

    return cudaSuccess;
}

// Function to read a matrix from file
void read_matrix(FILE* fp, Matrix* mat) {
    if (fscanf(fp, "%d", &mat->id) != 1) {
        fprintf(stderr, "Error reading matrix ID\n");
        exit(1);
    }
    if (fscanf(fp, "%d", &mat->size) != 1) {
        fprintf(stderr, "Error reading matrix size\n");
        exit(1);
    }
    // Allocate 2D array
    mat->data = (int**)malloc(mat->size * sizeof(int*));
    if (!mat->data) {
        fprintf(stderr, "Memory allocation failed for matrix rows\n");
        exit(1);
    }
    for (int i = 0; i < mat->size; i++) {
        mat->data[i] = (int*)malloc(mat->size * sizeof(int));
        if (!mat->data[i]) {
            fprintf(stderr, "Memory allocation failed for matrix row %d\n", i);
            exit(1);
        }
    }
    // Read matrix elements
    for (int i = 0; i < mat->size; i++) {
        for (int j = 0; j < mat->size; j++) {
            if (fscanf(fp, "%d", &mat->data[i][j]) != 1) {
                fprintf(stderr, "Error reading matrix element at (%d,%d)\n", i, j);
                for (int k = 0; k <= i; k++) {
                    free(mat->data[k]);
                }
                free(mat->data);
                exit(1);
            }
        }
    }
}

int load_input(const char* filename, double* threshold, int* num_pics, Matrix** pictures, int* num_objs, Matrix** objects) {
    FILE* fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "Cannot open %s\n", filename);
        return 1;
    }

    // Read threshold
    if (fscanf(fp, "%lf", threshold) != 1) {
        fprintf(stderr, "Error reading threshold\n");
        fclose(fp);
        return 1;
    }

    // Read number of pictures
    if (fscanf(fp, "%d", num_pics) != 1) {
        fprintf(stderr, "Error reading number of pictures\n");
        fclose(fp);
        return 1;
    }

    // Allocate and read pictures
    *pictures = (Matrix*)malloc(*num_pics * sizeof(Matrix));
    if (!*pictures) {
        fprintf(stderr, "Memory allocation failed for pictures\n");
        fclose(fp);
        return 1;
    }
    for (int p = 0; p < *num_pics; p++) {
        read_matrix(fp, &(*pictures)[p]);
    }

    // Read number of objects
    if (fscanf(fp, "%d", num_objs) != 1) {
        fprintf(stderr, "Error reading number of objects\n");
        fclose(fp);
        for (int p = 0; p < *num_pics; p++) {
            for (int i = 0; i < (*pictures)[p].size; i++) {
                free((*pictures)[p].data[i]);
            }
            free((*pictures)[p].data);
        }
        free(*pictures);
        return 1;
    }

    // Allocate and read objects
    *objects = (Matrix*)malloc(*num_objs * sizeof(Matrix));
    if (!*objects) {
        fprintf(stderr, "Memory allocation failed for objects\n");
        fclose(fp);
        for (int p = 0; p < *num_pics; p++) {
            for (int i = 0; i < (*pictures)[p].size; i++) {
                free((*pictures)[p].data[i]);
            }
            free((*pictures)[p].data);
        }
        free(*pictures);
        return 1;
    }
    for (int o = 0; o < *num_objs; o++) {
        read_matrix(fp, &(*objects)[o]);
    }
    fclose(fp);
    return 0;
}


cudaError_t allocateAllOnDevice(
    Matrix* h_pictures, 
    Matrix* h_objects, 
    int* num_objs, 
    Result* results,
    Matrix** d_picture, 
    int*** d_rows, 
    Matrix** d_objects, 
    int*** d_rows_obj, 
    int** d_num_objs, 
    Result** d_results,
    int num_chosen_pics
) {
    cudaError_t err;

    // Allocate device memory for one picture
    Matrix h_mat = h_pictures[num_chosen_pics];
    err = allocateMatrixOnDevice(h_mat, d_picture, d_rows);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate matrix on device (error %s)!\n", cudaGetErrorString(err));
        return err;
    }

    // Allocate and copy objects to device
    err = allocateObjectsOnDevice(h_objects, *num_objs, d_objects, d_rows_obj);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate objects on device (error %s)!\n", cudaGetErrorString(err));
        return err;
    }

    // allocate and copy num_objs on device
    err = cudaMalloc((void **)d_num_objs, sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device vector num_objs (error code %s)!\n",
                cudaGetErrorString(err));
        return err;
    }
    err = cudaMemcpy(*d_num_objs, num_objs, sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy num_objs from host to device (error code %s)!\n",
                cudaGetErrorString(err));
        return err;
    }

    // Allocate device memory for results
    err = cudaMalloc((void**)d_results, sizeof(Result));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device results (error %s)!\n",
                cudaGetErrorString(err));
        return err;
    }

    err = cudaMemcpy(*d_results, results, sizeof(Result), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device vector results (error code %s)!\n",
                cudaGetErrorString(err));
        return err;
    }

    return cudaSuccess;
}


int main(int argc, char* argv[]) {

    int rank, size;
    MPI_Status status;
    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Create MPI datatype for Result struct
    create_mpi_result_type();

    cudaError_t err = cudaSuccess;
    double threshold;
    int num_pics;
    Matrix* h_pictures;
    int* num_objs=(int *)malloc(sizeof(int));
    Matrix* h_objects;
    // Load input from file
    if (load_input("input.txt", &threshold, &num_pics, &h_pictures, num_objs, &h_objects) != 0) {
        return 1;
    }

    int num_chosen_pics = -1; 
    
    
    // Result* results = (Result*)malloc(num_pics * sizeof(Result));
    // if (!results) {
    //     fprintf(stderr, "Memory allocation failed for results\n");
    //     return 1;
    // }
    // for (int p = 0; p < num_pics; p++) {
    //     results[p].pic_id = h_pictures[p].id;
    //     results[p].obj_id = -1; // Default to -1 (no match)
    //     results[p].i = -1;
    //     results[p].j = -1;
    // }
    if (rank == 0)
    {
        // Master
        int active_workers = size - 1;

        while (active_workers > 0) {
            Result* final_results=(Result*)malloc(num_pics * sizeof(Result));
            
            // int worker_rank;
            // printf("Master waiting for results...\n");
            MPI_Recv(final_results,1, mpi_result_type,MPI_ANY_SOURCE,MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            // if(final_results->pic_id==-1){
            //     free(final_results);
                
            // }
            // printf("Master received results from worker %d\n", status.MPI_SOURCE);
            // printf("Master received results: Picture ID: %d, Object ID: %d found at position (%d, %d)\n",
                // final_results->pic_id, final_results->obj_id, final_results->i, final_results->j);
            // int src = status.MPI_SOURCE; 
            // MPI_Recv(&worker_rank, 1, MPI_INT, src, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            if (final_results != NULL && final_results->pic_id != -1) {
                if (final_results->obj_id != -1) {
                    printf("Picture %d found Object %d in position (%d,%d)\n",
                        final_results->pic_id, final_results->obj_id, final_results->i, final_results->j);
                } else {
                    printf("Picture %d, No object were found\n", final_results->pic_id);
                }
                free(final_results);
            }
            
            num_chosen_pics++;
            if (num_chosen_pics < num_pics) {
                // Send row index to worker
                MPI_Send(&num_chosen_pics, 1, MPI_INT, status.MPI_SOURCE, TAG_WORK, MPI_COMM_WORLD);
            } else {
                // No more work, send termination signal
                int terminate = -1;
                MPI_Send(&terminate, 1, MPI_INT, status.MPI_SOURCE, TAG_DONE, MPI_COMM_WORLD);
                active_workers--;
            }
        }

    }
    else
    {
        while (1) 
        {
            // Worker
            // Request work from master
            Result* results = (Result*)malloc(sizeof(Result));
            results->pic_id = -1;
            results->obj_id = -1; // Default to -1 (no match)
            results->i = -1;
            results->j = -1;
            if (num_chosen_pics == -1)
            {
                // printf("Worker %d requesting work from master\n", rank);
                // printf("sending empty result to master\n");
                MPI_Send(results, 1, mpi_result_type, 0, TAG_WORK, MPI_COMM_WORLD);
            }
            // printf("Worker %d waiting for work from master\n", rank);
            // Receive index picture that i wanna to work  from master
            MPI_Recv(&num_chosen_pics, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            if (status.MPI_TAG == TAG_DONE) {
                // printf("Worker %d received termination signal from master\n", rank);
                free(results);
                break; // Exit the loop if termination signal is received
            }
            // printf("Worker %d received work: picture index %d\n", rank, num_chosen_pics);
            if (num_pics < num_chosen_pics) {
                fprintf(stderr, "Not enough pictures in input\n");
                free(results);
                break;
            }
            results->pic_id = h_pictures[num_chosen_pics].id;
            // Allocate all necessary data on device
            Matrix* d_picture;
            int** d_rows;
            Matrix *d_objects = NULL;
            int** d_rows_obj ;
            int *d_num_objs = NULL;
            Result* d_results ;
            err = allocateAllOnDevice(h_pictures, h_objects, num_objs, results,&d_picture, &d_rows, &d_objects, &d_rows_obj, &d_num_objs, &d_results,num_chosen_pics);
            if (err != cudaSuccess) {
                fprintf(stderr, "Failed to allocate all on device (error %s)!\n", cudaGetErrorString(err));
                exit(EXIT_FAILURE);
            }
            
            // Launch the Vector Add CUDA Kernel
            dim3 threadsPerBlock(16, 16);
            dim3 numBlocks((h_pictures[num_chosen_pics].size + threadsPerBlock.x - 1) / threadsPerBlock.x,
                        (h_pictures[num_chosen_pics].size + threadsPerBlock.y - 1) / threadsPerBlock.y);
        
            find_Object_in_picture<<<numBlocks, threadsPerBlock>>>(d_picture, d_objects, *num_objs, d_results);
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                fprintf(stderr, "Kernel launch failed (error %s)!\n",
                        cudaGetErrorString(err));
                exit(EXIT_FAILURE);
            }
            
          
            // Copy the device result vector to the host result vector
            err = cudaMemcpy(results, d_results,  sizeof(Result), cudaMemcpyDeviceToHost);
            if (err != cudaSuccess) {
                fprintf(stderr, "Failed to copy Matrix struct (error %s)!\n",
                        cudaGetErrorString(err));
                exit(EXIT_FAILURE);
            }
            // printf("Worker %d finished processing picture index %d\n", rank, num_chosen_pics);
            // printf("worker %d sending results to master\n", rank);
            // printf("Result: Picture ID: %d, Object ID: %d found at position (%d, %d)\n",
            //     results->pic_id, results->obj_id, results->i, results->j);
            MPI_Send(results, 1, mpi_result_type, 0, TAG_WORK, MPI_COMM_WORLD);
            
            cudaFree(d_picture);
            cudaFree(d_objects);
            cudaFree(d_num_objs);
            cudaFree(d_results);
            free(results);
              

        }

    }    







    
    // Cleanup
    for (int p = 0; p < num_pics; p++) {
        for (int i = 0; i < h_pictures[p].size; i++) {
            free(h_pictures[p].data[i]);
        }
        free(h_pictures[p].data);
    }
    free(h_pictures);
    for (int o = 0; o < *num_objs; o++) {
        for (int i = 0; i < h_objects[o].size; i++) {
            free(h_objects[o].data[i]);
        }
        free(h_objects[o].data);
    }
    free(h_objects);
    free(num_objs);
    
    MPI_Finalize();
    return 0;
}