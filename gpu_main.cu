// GPU-powered Genetic Algorithm code
// Version 1.0
// Matthew Smith, Swinburne University of Technology

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <curand.h>
#include <curand_kernel.h>
#include "config.h"

float *h_x;		// 1D Array holding genetic information for NO_VAR parameters and NO_KIDS children
float *d_x;		//            As above, for storage and computation on device
float *d_x_new;		//	 New copy of the above data produced while mating.

float *h_fitness;	// Fitness for each child
float *h_fitness_new;
float *d_fitness;	// As above, stored on the GPU
float *d_fitness_new;	// New fitness

float x_min[] = {-5.0, -5.0, -5.0};	// Array of length NO_VAR holding initial minimum values for each parameter
float x_max[] = {5.0, 5.0, 5.0};	// As above, except this holds the maximum values

float *d_fitness_history;
float *h_fitness_history;
float *d_x_history;
float *h_x_history;

__device__ int d_BestKid;		// Store the best kid
__device__ float d_BestFitness;		// Store the best fitness
__device__ int d_Step;			// Store the current step

curandState *d_state;			// CuRAND state

int h_BestKid;				// Hold the best kid on the host
float h_BestFitness; 			// Host the best fitness too

float *d_Info;
float *h_Info;


/* ---------------------------------

	Device and GPU functions

-----------------------------------*/

__device__ float MaxFrac(float input) {

	if (input > 0.5) {
		return input;
	} else {
		return (1.0-input);
	}
}

__device__ float ComputeFitnessGPU(float *x, int Z) {

	// Ackley Function for N inputs
	int i;
	float sum1 = 0.0, sum2 = 0.0;
	for (i = 0; i < Z; i++) {
		sum1 = sum1 + x[i]*x[i];
		sum2 = sum2 + cosf(2.0*PI_F*x[i]);
	}

	return (-20.0*expf(-0.2*sqrtf(0.5*sum1)) - expf(0.5*sum2) + expf(1) + 20.0);

	// This was used for debugging
	//return d_BestKid;
}

__global__ void ComputeNewGenerationGPU(float *fitness, float *fitness_new, float *x, float *x_new, curandState *state, int N) {

	// Compute the new fitness
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int var;
	int ParentA, ParentB, ParentC;
	float trial_x[NO_VAR];
	float trial_fitness;
	float GF, RN;
	if (i < N) {

		ParentA = d_BestKid; // The fittest
		// Need to select 2 parents. The 3rd
		ParentB = -1; ParentC = -1;
		while ((ParentB == ParentC) || (ParentA == ParentB) || (ParentA == ParentC)) {
			// Choose two parents randomly
			ParentB = (int)(N*curand_uniform(&state[i]));
			ParentC = (int)(N*curand_uniform(&state[i]));
		}

		// Compute the new generation's value of x, and its fitness
		for (var = 0; var < NO_VAR; var++) {
			// Compute value of GF
			GF = curand_uniform(&state[i]);
			GF = FR*MaxFrac(GF);
			RN = curand_normal(&state[i]);

			// Compute new value of x
			trial_x[var] = GF*x[ParentA*NO_VAR + var] + (1.0-GF)*x[ParentB*NO_VAR + var]
				       + RN*SIG*(x[ParentB*NO_VAR + var]-x[ParentC*NO_VAR + var]);
		}

		// Now we have our X array, pass it on to compute our fitness
		trial_fitness = ComputeFitnessGPU(trial_x, NO_VAR);
		fitness_new[i] = trial_fitness;

		if (fabs(fitness_new[i]) < fabs(fitness[i])) {
			// Update
			for (var = 0; var < NO_VAR; var++) {
				x_new[i*NO_VAR + var] = trial_x[var];
			}
			// Fitness_new is already updated
		} else {
			// Dont use the new values
			fitness_new[i] = fitness[i];
                        // Update
                        for (var = 0; var < NO_VAR; var++) {
                                x_new[i*NO_VAR + var] = x[i*NO_VAR + var];
                        }
		}

	}
}

__global__ void SetupGPU(float *x, float *fitness, int N, int Z) {

	// Prepare the computation on the GPU
	// First step is to compute the fitness for each child
	// N = No. kids
	// Z = No. var
	int child = threadIdx.x + blockIdx.x*blockDim.x;
	float x_local[NO_VAR];
	int i;
	if (child < N) {
		// Copy X info into local mem
		for (i = 0; i < NO_VAR; i++) {
			x_local[i] = x[child*NO_VAR + i];
		}
		// Compute the fitness
		fitness[child] = ComputeFitnessGPU(x_local, Z);
	}

	// Set up our step count - currently on 0
	d_Step = 0;
}

__global__ void SetupRfGPU(curandState *state, int N) {

	int thread = threadIdx.x + blockDim.x*blockIdx.x;
	if (thread < N) {
		curand_init(1234, thread, 0, &state[thread]);
	}
}


__global__ void FindBestGPU(float *fitness, float *Info, float *fitness_history, int N) {

	int thread = threadIdx.x + blockIdx.x*blockDim.x;
	__shared__ float best_fitness[MAX_TPB];
	__shared__ int best_index[MAX_TPB];
	int pass = 0;
	int i;

	/*
	// This creates an array 1/4 the size of the original array.
	// The first passneeds to compare 4 values

	if (thread < N) {

		best_index[thread] = thread; // Say this is the best for now
		best_fitness[thread] = fitness[thread];

		for (i = 0; i < 3; i++) {
			if (fitness[thread + i*MAX_TPB] < best_fitness[thread]) {
				// This is a better choice
				best_fitness[thread] = fitness[thread + i*MAX_TPB];
				best_index[thread] = thread + i*MAX_TPB;
			}
		}

		// Now we have a list MAX_TPG long. We may commence reduction
		__syncthreads();

		if (thread == 0) {
			// Use a single thread for now
			for (i = 0; i < MAX_TPB; i++) {
				if (fitness[i] < best_fitness[thread]) {
					best_fitness[thread] = fitness[i];
					best_index[thread] = i;
				}
			}
			d_BestKid = best_index[thread];
			d_BestFitness = best_fitness[thread];
			Info[0] = d_BestKid;
			Info[1] = d_BestFitness;
		}
	}
	*/


	if (thread == 0) {
		best_fitness[0] = 10000.0;
		best_index[0] = -1;
		for (i = 0; i < N; i++) {
			// Find the best
			if (fabs(fitness[i]) < fabs(best_fitness[0])) {
				// Save it
				best_fitness[0] = fitness[i];
				best_index[0] = i;
			}
		}
		d_BestKid = best_index[0];
		d_BestFitness = best_fitness[0];
		Info[0] = best_index[0];
		Info[1] = best_fitness[0];

		// Update the history
	        fitness_history[d_Step] = d_BestFitness;
        	d_Step++;

	}
}

/* -----------------------------------
	CPU and Wrapping functions
--------------------------------------*/


void Allocate_Memory() {

	size_t size;
	cudaError_t error;

        if (DEBUG) printf("Allocating memory for %d kids and %d variables\n", NO_KIDS, NO_VAR);


	// Allocate memory on Host
	size = NO_KIDS*NO_VAR*sizeof(float);
	h_x = (float*)malloc(size);
	size = NO_KIDS*sizeof(float);
	h_fitness = (float*)malloc(size);
	h_fitness_new = (float*)malloc(size);
	size = 2*sizeof(float);
	h_Info = (float*)malloc(size);
	size = NO_GEN*sizeof(float);
	h_fitness_history = (float*)malloc(size);

	// Allocate memory on device
	size = NO_KIDS*NO_VAR*sizeof(float);
	error = cudaMalloc((void**)&d_x, size);
	if (DEBUG) printf("Memory Allocation of d_x - Error = %s\n", cudaGetErrorString(error));
        error = cudaMalloc((void**)&d_x_new, size);
        if (DEBUG) printf("Memory Allocation of d_x_new - Error = %s\n", cudaGetErrorString(error));

	size = NO_KIDS*sizeof(float);
	error = cudaMalloc((void**)&d_fitness, size);
        if (DEBUG) printf("Memory Allocation of d_fitness - Error = %s\n", cudaGetErrorString(error));
	size = 2*sizeof(float);
	error = cudaMalloc((void**)&d_Info, size);
	size = NO_KIDS*sizeof(float);
	error = cudaMalloc((void**)&d_fitness_new, size);

	size = NO_KIDS*sizeof(curandState);
	error = cudaMalloc((void**)&d_state, size);

	size = NO_GEN*sizeof(float);
	error = cudaMalloc((void**)&d_fitness_history, size);
}

float ComputeFitness(float *x, int N) {
        // Ackley Function for N inputs
        int i;
        float sum1 = 0.0, sum2 = 0.0;
        for (i = 0; i < N; i++) {
                sum1 = sum1 + x[i]*x[i];
                sum2 = sum2 + cosf(2.0*PI_F*x[i]);
        }

        return (-20.0*expf(-0.2*sqrtf(0.5*sum1)) - expf(0.5*sum2) + expf(1) + 20.0);
}

void ComputeNewGenCall() {

	if (DEBUG) printf("Computing New Generation\n");
	// Compute the next generation properties
	ComputeNewGenerationGPU<<<BPG,TPB>>>(d_fitness, d_fitness_new, d_x, d_x_new, d_state, NO_KIDS);

}


void UpdateNextGeneration() {

	size_t size;

	if (DEBUG) printf("Updating next generation\n");

	// Copy fitness
	size = NO_KIDS*sizeof(float);
	cudaMemcpy(d_fitness, d_fitness_new, size, cudaMemcpyDeviceToDevice);

	// Copy variables
	size = NO_KIDS*NO_VAR*sizeof(float);
	cudaMemcpy(d_x, d_x_new, size, cudaMemcpyDeviceToDevice);

}

void FindBestGPUCall() {
	// Using 512 threads on a single block (single SM) to find the best kid.
	// Very lazy.
	cudaError_t error;
	if (DEBUG) printf("Finding best child in current generation\n");
	FindBestGPU<<<1,MAX_TPB>>>(d_fitness, d_Info, d_fitness_history, NO_KIDS);

	// Copy values across for debugging
	cudaMemcpy(&h_BestKid, &d_BestKid, sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(&h_BestFitness, &d_BestFitness, sizeof(float), cudaMemcpyDeviceToHost);
	error = cudaMemcpy(h_Info, d_Info, 2*sizeof(float), cudaMemcpyDeviceToHost);
	printf("Primary: Index  = %d, Fitness = %g\n", h_BestKid, h_BestFitness);
	printf("CUDA Error on Copy: %s\n", cudaGetErrorString(error));
	printf("Backup: Index = %g, Fitness = %g\n", h_Info[0], h_Info[1]);

}

void SetupRfGPUCall() {

	if (DEBUG) printf("Initializing CuRAND states\n");
	SetupRfGPU<<<BPG,TPB>>>(d_state, NO_KIDS);

}

void SetupGPUCall() {

	if (DEBUG) printf("Launching SetupGPU kernel on device with %d threads per block across %d blocks\n", TPB, BPG);
	SetupGPU<<<BPG,TPB>>>(d_x, d_fitness, NO_KIDS, NO_VAR);

}

void Free_Memory() {
	cudaError_t error;
	// Free host memory
	free(h_x); free(h_fitness); free(h_Info); free(h_fitness_new); free(h_fitness_history);
	// Free GPU memory
	error = cudaFree(d_x);
	if (DEBUG) printf("Deallocation of d_x: Error = %s\n",	cudaGetErrorString(error));
        error = cudaFree(d_x_new);
        if (DEBUG) printf("Deallocation of d_x_new: Error = %s\n",  cudaGetErrorString(error));
	error = cudaFree(d_fitness);
        if (DEBUG) printf("Deallocation of d_fitness: Error = %s\n",  cudaGetErrorString(error));
	error = cudaFree(d_Info);
	error = cudaFree(d_fitness_new);

	error = cudaFree(d_state);

	error = cudaFree(d_fitness_history);

}

void Init() {
	// Set up the initial values of x for each variable and each child
	int seed = time(NULL);
	int i, j, index;
	int size[2];
	FILE *pFile;

	// Initialize the seed for the random number generator
	srand(seed);

	// Check size of x_min and x_max
	size[0] = sizeof(x_min);
	size[1] = sizeof(x_max);
	if (size[0] != size[1]) {
		printf("WARNING: Size mismatch in the x_min() and x_max() variables.\n");
	} else {
		if (size[0]/4 != NO_VAR) {
			printf("WARNING: Size mismatch between x_min/x_max and NO_VAR parameter.\n");
		} else {
			// Continue
			printf("Size of x_min() and x_max() arrays = OK\n");
		}
	}

	// Create our values for each child
	for (i = 0; i < NO_KIDS; i++) {
		for (j = 0; j < NO_VAR; j++) {
			index = i*NO_VAR + j;
			h_x[index] = x_min[j] + (x_max[j]-x_min[j])*Randf();
			// if (DEBUG) printf("x[%d, %d] = %g\n", i, j, h_x[index]);
		}
	}

	pFile = fopen("Init_Data.txt", "w");
	if (pFile == NULL) {
		printf("Error opening Init_Data.txt file.\n");
	} else {
		index = 0;
		for (i = 0; i < NO_KIDS; i++) {
			for (j = 0; j < NO_VAR; j++) {
				fprintf(pFile, "%g", h_x[index]);
				index++;
				if (j == (NO_VAR-1)) {
					fprintf(pFile, "\n");
				} else {
					fprintf(pFile, "\t");
				}
			}
		}
		fclose(pFile);
	}

}

float Randf() {
	// Random Number generator
	// This is a very low quality generator
	return (float)rand()/(float)RAND_MAX;
}

float RandNf() {
	// Generate normally distributed random  number
	// using the Marsaglia Polar Method
	// This produces 2 values for any given Randf() call,
	// which are stored in static values between calls.

	float U1, U2, W, mult;
	static float X1, X2;
	static int call = 0;

	if (call == 1) {
		call = !call;
		return X2;
	}
	do {
		U1 = -1 + Randf()*2.0;
		U2 = -1 + Randf()*2.0;
		W = U1*U1 + U2*U2;
	} while (W >= 1 || W == 0);

	mult = sqrt((-2.0*log(W))/W);
	X1 = U1*mult;
	X2 = U2*mult;

	call = !call;

	return X1;
}

void SaveFitness() {

	FILE *pFile;
	int i;
	pFile = fopen("Fitness.txt", "w");

	if (pFile == NULL) {
		printf("Error: Could not open Fitness.txt for writing.\n");
	} else {
		for (i = 0; i < NO_KIDS; i++) {
			fprintf(pFile, "%d\t%g\t%g\n", i, h_fitness[i], h_fitness_new[i]);
		}
		fclose(pFile);
	}

	// Save the history
	pFile = fopen("History.txt", "w");
	if (pFile == NULL) {
		printf("Error: Could not open History.txt for writing\n");
	} else {
		for (i = 0; i < NO_GEN; i++) {
			fprintf(pFile, "%d\t%g\n", i, h_fitness_history[i]);
		}
		fclose(pFile);
	}

}

void SendToGPU() {

	size_t size = NO_KIDS*NO_VAR*sizeof(float);
	cudaError_t error;
	error = cudaMemcpy(d_x, h_x, size, cudaMemcpyHostToDevice);
	if (DEBUG) printf("Memcpy to device: h_x: Error = %s\n",
	cudaGetErrorString(error));
}

void SendToHost() {

        size_t size = NO_KIDS*sizeof(float);
	int BestIndex, i;
	float BestFitness;
	int var;
        cudaError_t error;
        error = cudaMemcpy(h_fitness, d_fitness, size, cudaMemcpyDeviceToHost);
        if (DEBUG) printf("Memcpy to host: d_fitness: Error = %s\n", cudaGetErrorString(error));

	error = cudaMemcpy(h_fitness_new, d_fitness_new, size, cudaMemcpyDeviceToHost);
	if (DEBUG) printf("Memcpy to host: d_fitness_new: Error = %s\n", cudaGetErrorString(error));

	// Copy the values of X across too
	size = NO_KIDS*NO_VAR*sizeof(float);
        error = cudaMemcpy(h_x, d_x, size, cudaMemcpyDeviceToHost);
        if (DEBUG) printf("Memcpy to host: d_x: Error = %s\n", cudaGetErrorString(error));

	// Copy the history across
	size = NO_GEN*sizeof(float);
	error = cudaMemcpy(h_fitness_history, d_fitness_history, size, cudaMemcpyDeviceToHost);

	// Debug time - find best
	BestFitness = 10000.0;
	BestIndex = -1;

	for (i = 0; i < NO_KIDS; i++) {
		if (h_fitness[i] < BestFitness) {
			BestFitness = h_fitness[i];
			BestIndex = i;
		}
	}
	printf("Best Index = %d, Best Fitness = %g\n", BestIndex, BestFitness);
	printf("X Values for best = ");
	for (var = 0; var < NO_VAR; var++) {
		printf("%g,\t", h_x[BestIndex*NO_VAR + var]);
	}
}