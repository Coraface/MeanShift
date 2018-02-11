#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define I 5
#define J 2

/*-Global variables-*/
double x[I][J], y[I][J];
int k;																							// k = neighbors for KNN algorithm

/**--Data structure--*/
typedef struct data {
	double dist;                                  									// squared distance of two elements
	int neighbor;                                  									// position of 2nd element
}Data;

/***---Function declarations---***/
double gaussian(double distance, double bandwidth);
void getData();																		// get dataset
int KNN(int i, struct data *);														// KNN algorithm for updating the bandwidth
int compareFunc(const void* a, const void* b);  									// compare function for qsort, ascending
void printTable();

/****------Main programm------****/
int main(int argc, char** argv)
{

	int conv = -1;
	while ((k < 1 || k > I - 1) && (conv < 0))
	{
		printf("Give number of neighbors( > 1 & < elements-1), used for calculating bandwidth :\n");
		scanf("%d", &k);
		printf("And also the number of iterations for convergence (> 0): \n");
		scanf("%d", &conv);
	}

	int i, j, l, z;																	// iterators
	double *d_x, *d_y;
	double sum1, sum2, distance, meanshift, previousmeansh, bandwidth = 0, ynew[2], yprevious[2];
	Data *array = (Data*)malloc(k * sizeof(Data));

	getData();
	//printTable();

	for (i = 0; i < I; i++) {
		meanshift = 1000000; previousmeansh = 0; sum1 = 0; sum2 = 0;
		for (z = 0; z < J; z++) ynew[z] = 0;										// reset ynew[] for every i
		bandwidth = KNN(i, array);													// bandwidth = distance between i and k neighbor
		for(z = 0; z < conv; z++) {
			for (j = 0; j < I; j++) {
				sum1 = 0; distance = 0;
				for (l = 0; l < J; l++)	distance += pow(y[i][l] - x[j][l], 2);
				distance = sqrt(distance);
				if (distance <= bandwidth);
				{
					sum1 = gaussian(distance, bandwidth);
					sum2 += gaussian(distance, bandwidth);
					for (l = 0; l < J; l++)
						ynew[l] += sum1 * x[j][l];
				}
			}
			previousmeansh = meanshift;
			meanshift = 0;
			for (l = 0; l < J; l++) {
				yprevious[l] = y[i][l];
				ynew[l] = ynew[l] / sum2;
				y[i][l] = ynew[l];
				meanshift += pow(ynew[l] - yprevious[l], 2);
			}
			meanshift = sqrt(meanshift);
		}
		printf("Mean for %d element is : (%lf, %lf)\n", i, y[i][0], y[i][1]);
	}

	return 0;
}


/****---------Functions---------****/

double gaussian(double distance, double bandwidth) {
	return exp(-distance / (2 * pow(bandwidth, 2)));
}

void getData() {
	int i, j;
	FILE *file = fopen("dataset.txt", "r");
	for (i = 0; i < I; i++) {
		for (j = 0; j < J; j++)
		{
			fscanf(file, "%lf", &x[i][j]);
			y[i][j] = x[i][j];
		}
	}
	fclose(file);
}

int KNN(int i, Data *array) {
	int j, l, cnt;
	double distance;
	cnt = 1;                         												// initialize counters for every i element
	for (j = 0; j < I; j++)
	{
		distance = 0;                           									// initialize dist sum for every j
		if (j == i) continue;                    									// distance = 0, duh
		for (l = 0; l < J; l++)	distance += pow(y[i][l] - x[j][l], 2);
		distance = sqrt(distance);
		if (cnt <= k)
		{
			array[cnt - 1].dist = distance;
			array[cnt - 1].neighbor = j;
			if (cnt == k) qsort(array, k, sizeof(Data), &compareFunc);
			cnt++;
		}
		else
		{
			if (array[cnt - 2].dist > distance)
			{
				array[cnt - 2].dist = distance;
				array[cnt - 2].neighbor = j;
				qsort(array, k, sizeof(Data), &compareFunc);
			}
		}
	}
	return array[k - 1].dist;
}

int compareFunc(const void* a, const void* b) {
	struct data *a1 = (struct data *)a;
	struct data *a2 = (struct data *)b;
	if ((*a1).dist > (*a2).dist) return 1;
	else if ((*a1).dist < (*a2).dist) return -1;
	else return 0;
}

void printTable()
{
	int i, j;
	for (i = 0; i < I; i++) {
		printf("\t");
		for (j = 0; j < J; j++)  printf("%lf ", x[i][j]);
	}
}
