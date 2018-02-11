#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <cuda.h>

#define I 5000
#define J 2

#define BLOCKSIZEx 512

/*-Global variables-*/
struct timeval startwtime, endwtime;
double seq_time;

/**---Host function declarations---**/
int Blocks(int x, int b){ return ((x % b) != 0) ? (x / b + 1) : (x / b); }
void getData(double *, double *);									// get dataset
void printTable(double *);

/***-----Device function declarations-----***/
__device__ void sort(double *, int);									// Insertion sort
__device__ double KNN(double *, int, int);								// KNN algorithm for updating the bandwidth
__device__ double gaussian(double distance, double bandwidth){
  return exp(-distance / (2 * pow(bandwidth, 2)));
}
__global__ void kernel(int *, int *, double *, double *, double *);

/****-------------Main programm-------------****/
int main(int argc, char** argv)
{
  int i, l, k = 0, intsize = sizeof(int), doubsize = sizeof(double);					// k = neighbors for KNN algorithm
  int conv = 0;
  double *d_x, *d_y, *d_m;
  double *x, *y, *m;
  int *d_c, *d_k;
  while ((k < 1 || k > I - 1) && (conv <= 0))
  {
    printf("Give number of neighbors( > 1 & < elements-1), used for calculating bandwidth :\n");
    scanf("%d", &k);
    printf("And also the number of iterations for convergence (> 0): \n");
    scanf("%d", &conv);
  }
  x = (double *)malloc((I * J) * doubsize);
  y = (double *)malloc((I * J) * doubsize);
  m = (double *)malloc((I * J) * doubsize);

  getData(x, y);
  //printTable(x);
  printf("\n");

  cudaMalloc(&d_c, intsize);
  cudaMalloc(&d_k, intsize);
  cudaMalloc(&d_x, (I*J)*doubsize);
  cudaMalloc(&d_y, (I*J)*doubsize);
  cudaMalloc(&d_m, (I*J)*doubsize);
  cudaMemcpy(d_c, &conv, intsize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_k, &k, intsize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_x, x, (I*J)*doubsize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, (I*J)*doubsize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_m, m, (I*J)*doubsize, cudaMemcpyHostToDevice);

  dim3 gridSize(Blocks(I*J, BLOCKSIZEx));
  dim3 blockSize(BLOCKSIZEx);

  gettimeofday (&startwtime, NULL);

  kernel<<<gridSize,blockSize>>>(d_c, d_k, d_x, d_y, d_m);
  cudaDeviceSynchronize();

  gettimeofday (&endwtime, NULL);
  seq_time = (double)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6
	+ endwtime.tv_sec - startwtime.tv_sec);

  cudaMemcpy(m, d_m, (I*J)*doubsize, cudaMemcpyDeviceToHost);
  for(i = 0; i < I; i++){
    printf("\nMean for [%d] element is : ", i);
    for(l = 0; l < J; l++) printf("%lf ", m[i*J+l]);
  }

  printf("\n\nKernel clock time = %f\n", seq_time);

  free(x); free(y);
  cudaFree(d_x); cudaFree(d_y); cudaFree(d_m);

  return 0;
}


/*****---------Host functions---------*****/
void getData(double *x, double *y) {
  int i, j;
  FILE *file = fopen("dataset.txt", "r");
  for (i = 0; i < I; i++) {
    for (j = 0; j < J; j++)
    {
      fscanf(file, "%lf", &x[i*J+j]);
      y[i*J+j] = x[i*J+j];
    }
  }
  fclose(file);
}

void printTable(double *x)
{
  int i, j;
  for (i = 0; i < I; i++) {
    printf("\t");
    for (j = 0; j < J; j++) printf("%lf ", x[i*J+j]);
  }
}

/******---------Device functions---------******/
__device__ void sort(double *dist, int n){
  int i, tmp, z;
  for(i = 1; i < n; ++i){
    tmp = dist[i];
    z = i;
    while(z > 0 && tmp < dist[z - 1]) {
      dist[z] = dist[z - 1];
      --z;
    }
    dist[z] = tmp;
  }
}

__device__ double KNN(double *X, int indexi, int n){
  int j, l, cnt = 1;                         								// initialize counters for every i element
  double distance, *dist;
  dist = (double *)malloc(n * sizeof(double));
  for (j = 0; j < I; j = j + 2)
  {
    distance = 0;                           								// initialize dist sum for every j
    if (j == indexi) continue;                    							// distance = 0, duh
    for (l = 0; l < J; l++)	distance += pow(X[indexi+l] - X[j+l], 2);
    distance = sqrt(distance);
    if (cnt <= n)
    {
      dist[cnt - 1] = distance;
      if (cnt == n) sort(dist, n);
	cnt++;
    }
    else
    {
      if (dist[cnt - 2] > distance)
      {
	dist[cnt - 2] = distance;
	sort(dist, n);
      }
    }
  }
  return dist[n-1];
}

__global__ void kernel(int *conv, int *k, double *X, double *Y, double *M){
  int indexi = blockIdx.x*blockDim.x + threadIdx.x * 2;							// index for every i element
  int realIndex = blockIdx.x*blockDim.x + threadIdx.x;
  if(indexi < I*J){

    int j, l, z, c = *conv, n = *k;									// iterators
    double sum1, sum2, distance, MeanshRange, bandwidth = 0, ynew[2], yprevious[2];
    MeanshRange = 1000000; sum1 = 0; sum2 = 0;

    for(z = 0; z < J; z++) ynew[z] = 0;									// reset ynew[] for every i
    bandwidth = KNN(X, indexi, n);									// bandwidth = distance between i and k neighbor

    for(z = 0; z < c; z++) {
      for(j = 0; j < I; j = j + 2) {
	sum1 = 0; distance = 0;
	for(l = 0; l < J; l++) distance += pow(Y[indexi+l] - X[j+l], 2);
	distance = sqrt(distance);
	if (distance <= bandwidth);
	{
	  sum1 = gaussian(distance, bandwidth);
	  sum2 += gaussian(distance, bandwidth);
	  for(l = 0; l < J; l++) ynew[l] += sum1 * X[j+l];	     
	}
      }
      MeanshRange = 0;
      for(l = 0; l < J; l++) {
	yprevious[l] = Y[indexi+l];
	ynew[l] = ynew[l] / sum2;
	Y[indexi+l] = ynew[l];
	MeanshRange += pow(ynew[l] - yprevious[l], 2);
      }
      MeanshRange = sqrt(MeanshRange);
    }

    for(l = 0; l < J; l++) M[indexi+l] = ynew[l];
  }
}
