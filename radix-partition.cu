#include <assert.h>
#include <stdio.h>
#include <math.h>

#define RAND_RANGE(N) ((double)rand()/((double)RAND_MAX + 1)*(N))

//data generator
void dataGenerator(long long* data, long long count, int first, int step)
{
	assert(data != NULL);

	for(long long i = 0; i < count; ++i)
		data[i] = first + i * step;
	srand(time(NULL));
    for(long long i = count-1; i>0; i--) //knuth shuffle
    {
        long long j = RAND_RANGE(i);
        long long k_tmp = data[i];
        data[i] = data[j];
        data[j] = k_tmp;
    }
}

/* This function embeds PTX code of CUDA to extract bit field from x. 
   "start" is the starting bit position relative to the LSB. 
   "nbits" is the bit field length.
   It returns the extracted bit field as an unsigned integer.
*/
__device__ uint bfe(uint x, uint start, uint nbits)
{
    uint bits;
    asm("bfe.u32 %0, %1, %2, %3;" : "=r"(bits) : "r"(x), "r"(start), "r"(nbits));
   // printf("%d\n", bits);
    return bits;
}

//Feel free to change the names of the kernels or define more kernels below if necessary

//define the histogram kernel here
__global__ void histogram(long long* r_d, int* partition_d, int partitionSize, long long rSize)
{
    extern __shared__ int R[];

    int blkDim = blockDim.x;
    int thdId = threadIdx.x;
    int blkId = blockIdx.x;

    int *sharedPartitionArray = (int *)&R[partitionSize];

    long long tid = blkDim * blkId + thdId;
    

    for(int i = thdId; i <  partitionSize; i = i + blkDim)
    {
        sharedPartitionArray[i] = 0;
         
    }

    __syncthreads();
    if(tid < rSize)
    {
        long long key = r_d[tid];
        uint  nbits =ceil(log2((float(partitionSize)))); 
        uint h = bfe(key, 0, nbits);
        //printf("%d\n", h);
        //int h = key % partitionSize;
        atomicAdd(&(sharedPartitionArray[h]), 1); 
       // printf("%d\n", sharedPartitionArray[h]);      
    }
    __syncthreads();

    for(int i = thdId; i <  partitionSize; i = i + blkDim)
   {
       atomicAdd(&(partition_d[i]), (sharedPartitionArray[i]));
   }

}

//define the prefix scan kernel here
//implement it yourself or borrow the code from CUDA samples
__global__ void prefixScan(int* partition_d, int* prefixScan_d, int partitionSize)
{
    extern __shared__ int R[];

    int blkDim = blockDim.x;
    int thdId = threadIdx.x;
    int blkId = blockIdx.x;

    int *sharedPartitionArray = (int *)&R[partitionSize];
    int stride = 1;
    int span = 2*thdId;
    long long tid = blkDim * blkId + thdId;
    if(tid < partitionSize/2 ){

        sharedPartitionArray[span] = partition_d[span];
        sharedPartitionArray[span+1] = partition_d[span+1];

        for(int i = partitionSize >> 1; i > 0; i >>= 1)
        {
            __syncthreads();
            if(thdId < i)
            {
                int a = stride * (span+1) - 1;
                int b = stride * (span+2) - 1;
                 atomicAdd(&(sharedPartitionArray[b]), (sharedPartitionArray[a]));
                 // sharedPartitionArray[b] += sharedPartitionArray[a];
            }
            stride *= 2;
        }

        if(thdId == 0)
        {
            sharedPartitionArray[partitionSize - 1] = 0;
        }

        for(int i = 1; i < partitionSize; i *= 2)
        {
            stride >>= 1;
            __syncthreads();

            if(thdId < i)
            {
                int a = stride * (span+1)-1;
                int b = stride * (span+2)-1;

                int t = sharedPartitionArray[a];
                sharedPartitionArray[a] = sharedPartitionArray[b];
                atomicAdd(&(sharedPartitionArray[b]), t);
                // sharedPartitionArray[b] += t;
            }
        }

        __syncthreads();

        span = 2*thdId;
        prefixScan_d[span] = sharedPartitionArray[span];
        prefixScan_d[span+1] = sharedPartitionArray[span+1];
}
}

//define the reorder kernel here
__global__ void Reorder(long long* r_d, int* prefixScan_d, int partitionSize, long long rSize, long long* finalOutput)
{
    int blkDim = blockDim.x;
    int thdId = threadIdx.x;
    int blkId = blockIdx.x;

    long long tid = blkDim * blkId + thdId;



    if(tid < rSize)
    {
        long long key = r_d[tid];
        uint  nbits =ceil(log2((float(partitionSize)))); 
        uint h = bfe(key, 0, nbits);
        int offset = atomicAdd(&(prefixScan_d[h]), 1); 
        // printf("%d %lld\n", key,offset);
        finalOutput[offset] = key;        
     }

}

int main(int argc, char const *argv[])
{
    long long rSize = atoi(argv[1]);
    int partitionSize = atoi(argv[2]);
    
    long long* r_h; //input array
    long long* r_d; //

    int* partition_h = (int *)malloc(sizeof(int)*partitionSize);
    int* partition_d = (int *)malloc(sizeof(int)*partitionSize);


    cudaMallocHost((void**)&r_h, sizeof(long long)*rSize); //use pinned memory in host so it copies to GPU faster
    cudaMalloc((void**)&r_d, sizeof(long long)*rSize);

    // cudaMallocHost((void**)&r_h, sizeof(int)*rSize); //use pinned memory in host so it copies to GPU faster
    // cudaMalloc((void**)&r_d, sizeof(int)*rSize);

    int no_of_blocks = ceil(rSize/(float)128);
 
    
    dataGenerator(r_h, rSize, 0, 1);
    cudaMemcpy(r_d, r_h, sizeof(long long)*rSize, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&partition_d, sizeof(int) * partitionSize);
    cudaMemcpy(partition_d, partition_h, sizeof(int) * partitionSize, cudaMemcpyHostToDevice);

    int* prefixScan_h = (int *)malloc(sizeof(int)*partitionSize);
    int* prefixScan_d;

    cudaMalloc((void**)&prefixScan_d, sizeof(int) * partitionSize);
    long long* finalOutput_h = (long long *)malloc(sizeof(long long)*rSize);
    long long* finalOutput;

    cudaMalloc((void**)&finalOutput, sizeof(long long) *rSize);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    histogram<<<no_of_blocks, 128, sizeof(int) * partitionSize>>>(r_d, partition_d, partitionSize, rSize);

    no_of_blocks = ceil(partitionSize/(float)32);
 
    prefixScan<<<no_of_blocks, 32, sizeof(int) * partitionSize>>>(partition_d, prefixScan_d, partitionSize);
    cudaMemcpy(prefixScan_h, prefixScan_d, sizeof(int)*partitionSize, cudaMemcpyDeviceToHost);


    no_of_blocks = ceil(rSize/(float)32);


    Reorder<<<no_of_blocks, 32>>>(r_d, prefixScan_d, partitionSize, rSize, finalOutput);

    cudaEventRecord(stop, 0); 
    cudaEventSynchronize(stop);
    float elapsedTime; 
    cudaEventElapsedTime(&elapsedTime, start, stop);
    
    cudaEventDestroy(start); 
    cudaEventDestroy(stop);

    cudaMemcpy(partition_h, partition_d, sizeof(int)*partitionSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(finalOutput_h, finalOutput, sizeof(long long)*rSize, cudaMemcpyDeviceToHost);

    printf("Histogram Array : ");
    for(int i = 0; i < partitionSize; i++)
    {
        printf("%d ", partition_h[i]);
    }
    printf("\n\n");     
    printf("Prefix Scan Array : ");
    for(int i = 0; i < partitionSize; i++)
    {
        printf("%lld ", prefixScan_h[i]);
    }
    printf("\n\n");
    printf("Radix Partition Result : ");
    for(long long i = 0; i < rSize; i++)
    {
        printf("%lld ", finalOutput_h[i]);
    }
    printf("\n\n");
    printf( "******** Total Running Time of Kernel: %0.5f ms *******\n", elapsedTime );
    /* your code */


    cudaFreeHost(r_h);
    cudaFree(r_d);
    free(partition_h);
    cudaFree(partition_d);
    cudaFree(prefixScan_d);
    free(prefixScan_h);
    return 0;
}
