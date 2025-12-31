#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>



void split_string(const char *src, char *odd, char *even) {
    int oddi = 0, eveni = 0;
    for (int i = 0; src[i] != '\0'; i++) {
        if (i % 2 == 0)
        {
            even[eveni++] = src[i];
        }
        else {
            odd[oddi++] = src[i];
        } 
    }
    odd[oddi] = '\0';
    even[eveni] = '\0';
}

int main(int argc, char **argv) {
    int rank, size, K, N, MaxIterations;
    char *my_string = NULL;
    char *SubString = NULL;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int maxString_len ;
    // Step 1: Read input on process 0
    if (rank == 0) {
        FILE *fp = fopen("data.txt", "r");
        fscanf(fp, "%d %d %d", &K, &N, &MaxIterations);
        maxString_len = 2 * N;
        SubString = (char *)malloc(maxString_len);
        fscanf(fp, "%s", SubString);
        int total = K * K;
        if (total != size) {
            printf("Error: Run with exactly %d processes.\n", total);
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
        char temp[maxString_len];
        for (int i = 0; i < total; i++) {
            MPI_Send(&MaxIterations, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            MPI_Send(&K, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            MPI_Send(&maxString_len, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            MPI_Send(SubString, maxString_len + 1, MPI_CHAR, i,0, MPI_COMM_WORLD);
            fscanf(fp, "%s", temp);
            if (i == 0) {
                my_string = strdup(temp);
            } else {
                MPI_Send(temp, strlen(temp) + 1, MPI_CHAR, i, 0, MPI_COMM_WORLD);
            }
        }
        fclose(fp);
    } else {
        MPI_Recv(&MaxIterations, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&K, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&maxString_len, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        SubString = (char *)malloc(maxString_len + 1);
        MPI_Recv(SubString, maxString_len + 1, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        my_string = (char *)malloc(maxString_len+1);
        MPI_Recv(my_string, maxString_len+1, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // Create Cartesian communicator
    
    MPI_Comm cart_comm;
    int dims[2] = {K, K}, periods[2] = {0, 0}, coords[2];
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &cart_comm);
    MPI_Cart_coords(cart_comm, rank, 2, coords);
    int left, right, up, down;
    MPI_Cart_shift(cart_comm, 1, -1, &right, &left);
    MPI_Cart_shift(cart_comm, 0, -1, &down, &up);

    char odd[maxString_len], even[maxString_len];
    char recv_right[maxString_len], recv_down[maxString_len];

    for (int iter = 0; iter < MaxIterations; iter++) {
        int found = (strstr(my_string, SubString) != NULL);
        
        int global_found;
        MPI_Allreduce(&found, &global_found, 1, MPI_INT, MPI_LOR, cart_comm);

        if (global_found) {
            if (rank == 0) {
                for (int i = 0; i < size; i++) {
                    if (i == 0) {
                        printf("Process 0: %s\n", my_string);
                    } else {
                        char buf[maxString_len];
                        MPI_Recv(buf, maxString_len, MPI_CHAR, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        printf("Process %d: %s\n", i, buf);
                    }
                }
            } else {
                MPI_Send(my_string, strlen(my_string)+1, MPI_CHAR, 0, 1, MPI_COMM_WORLD);
            }
            break;
        }
        split_string(my_string, odd, even);
        // send and receive  
        if (right != MPI_PROC_NULL && left != MPI_PROC_NULL) 
        {
            MPI_Sendrecv(odd, strlen(odd)+1, MPI_CHAR, left, 0,
                        recv_right, strlen(odd)+1, MPI_CHAR, right, 0, cart_comm, MPI_STATUS_IGNORE);
        } 
        else if (left != MPI_PROC_NULL )
        {
            MPI_Send(odd, strlen(odd)+1, MPI_CHAR, left, 0, cart_comm);
            recv_right[0] = '\0'; // Initialize to empty string
        }
        else if (right != MPI_PROC_NULL)
        {
            MPI_Recv(recv_right, strlen(odd)+1, MPI_CHAR, right, 0, cart_comm, MPI_STATUS_IGNORE);
        }
        if (up != MPI_PROC_NULL && down != MPI_PROC_NULL) 
        {
            MPI_Sendrecv(even, strlen(even)+1, MPI_CHAR, up, 0,
                        recv_down, strlen(even)+1, MPI_CHAR, down, 0, cart_comm, MPI_STATUS_IGNORE);
        } 
        else if (up != MPI_PROC_NULL)
        {
            MPI_Send(even, strlen(even)+1, MPI_CHAR, up, 0, cart_comm);
            recv_down[0] = '\0'; // Initialize to empty string
        }
        else if (down != MPI_PROC_NULL)
        {
            MPI_Recv(recv_down, strlen(even)+1, MPI_CHAR, down, 0, cart_comm, MPI_STATUS_IGNORE);
        }

        // build new string
        snprintf(my_string, maxString_len+1, "%s%s", recv_right, recv_down);
        printf("Process %d: %s\n", rank, my_string);
        // check for SubString
        
    }

    if (rank == 0 && strstr(my_string, SubString) == NULL) {
        printf("The string was not found\n");
    }
    free(SubString);
    free(my_string);
    MPI_Finalize();
    return 0;
}
