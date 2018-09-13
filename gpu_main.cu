// GPU-powered Genetic Algorithm code
// Version 1.1
// Matthew Smith, Swinburne University of Technology

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
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

float x_min[] = {-2.0, -2.0};	// Array of length NO_VAR holding initial minimum values for each parameter
float x_max[] = {2.0, 2.0};	// As above, except this holds the maximum values

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

float *d_Info;				// Used for double checking
float *h_Info;

__device__ int d_Test;			// Our current ensemble (test)

// Create redundant timers
cudaEvent_t start, stop;
int msec;
struct timeval start2, stop2, result;

// Data to hold ensemble information
float *d_avg_fitness;		// 1D array of length NO_GEN*NO_TESTS
float *h_avg_fitness;		// Same, only stored on the host

/* ---------------------------------

	Device and GPU functions

-----------------------------------*/

__global__ void Increment_Step() {

	// All we are doing is incrementing the step
	d_Test++;

}

__device__ float MaxFrac(float input) {

	if (input > 0.5) {
		return input;
	} else {
		return (1.0-input);
	}
}

__device__ float ComputeFitnessGPU(float *x, int Z) {

	/*
	// Ackley Function for N inputs
	int i;
	float sum1 = 0.0, sum2 = 0.0;
	for (i = 0; i < Z; i++) {
		sum1 = sum1 + x[i]*x[i];
		sum2 = sum2 + cosf(2.0*PI_F*x[i]);
	}

	return (-20.0*expf(-0.2*sqrtf(0.5*sum1)) - expf(0.5*sum2) + expf(1) + 20.0);
	*/
	// Goldstein-Price function (two variables)
	return (1.0 + (x[0]+x[1]+1.0)*(x[0]+x[1]+1.0)*(19.0-14.0*x[0]+3.0*x[0]*x[0]-14.0*x[1]+6.0*x[0]*x[1]+3.0*x[1]*x[1]))*
		(30.0 + (2.0*x[0]-3.0*x[1])*(2.0*x[0]-3.0*x[1])*(18.0-32.0*x[0]+12.0*x[0]*x[0]+48.0*x[1]-36.0*x[0]*x[1]+27.0*x[1]*x[1]));




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

	if (thread == 0) {
		// We can also initialize the ensemble count here
		d_Test = 0;
	}
}


__global__ void FindBestGPU(float *fitness, float *Info, float *fitness_history, float *x_history, float *x, float *avg_fitness, int N) {

	// This kernel is designed to run with a single block of MAX_TPB threads.
	int thread = threadIdx.x;
	__shared__ float best_fitness[MAX_TPB];
	__shared__ int best_index[MAX_TPB];
	int pass = 0;
	int i;
	int PASS_PER_THREAD = NO_KIDS/MAX_TPB;
	int REM = NO_KIDS%MAX_TPB;

	if (REM != 0) {
		// Feeling lazy at the moment. Use a single thread
		if (thread == 0) {
			best_fitness[0] = 10000.0;
			best_index[0] = -1;
			for (i = 0; i < NO_KIDS; i++) {
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
	} else {
		// Each thread has access to PASS_PER_THREAD  elements
		best_fitness[thread] = 100000.0;
		best_index[thread] = -1;

		for (i = 0; i < PASS_PER_THREAD; i++) {
			if (fabs(fitness[thread + i*MAX_TPB]) < fabs(best_fitness[thread])) {
				best_fitness[thread] = fitness[thread + i*MAX_TPB];
				best_index[thread] = thread + i*MAX_TPB;
			}
		}

		// We have reduced this problem down to one with MAX_TPB elements now.
		// We could use parallel reduction here.

		/*
		__syncthreads();
		if (thread == 0) {
			for (i = 0; i < MAX_TPB; i++) {
				if (fabs(best_fitness[i]) < fabs(best_fitness[thread])) {
					best_fitness[0] = best_fitness[i];
					best_index[0] = best_index[i];
				}
			}
			d_BestKid = best_index[0];
			d_BestFitness = best_fitness[0];
			Info[0] = best_index[0];
			Info[1] = best_fitness[0];

		}
		*/


		__syncthreads();

		for (int stride = blockDim.x/2; stride > 0; stride = stride/2) {
			if (thread < stride) {
				if (fabs(best_fitness[thread+stride]) < fabs(best_fitness[thread])) {
 					// store it
					best_fitness[thread] = best_fitness[thread + stride];
					best_index[thread] = best_index[thread + stride];
				}
			}
			__syncthreads();
		}

		// We have found the best one
		if (thread == 0) {
			d_BestKid = best_index[0];
			d_BestFitness = best_fitness[0];
			Info[0] = best_index[0];
			Info[1] = best_fitness[0];

			// Need also to save the history
			fitness_history[d_Step] = d_BestFitness;

			// Also need to save this as part of the larger test ensemble
			avg_fitness[d_Test*NO_GEN + d_Step] = d_BestFitness;

			// Save the X values
			for (i = 0; i < NO_VAR; i++) {
				x_history[d_Step*NO_VAR + i] = x[d_BestKid*NO_VAR + i];
			}
			d_Step++;
		}
	}
}

/* -----------------------------------
	CPU and Wrapping functions
--------------------------------------*/

void IncrementTestCall() {

	Increment_Step<<<1,1>>>();

}

void SendTestData() {

	// Copy the ensemble test data back to the host
	cudaError_t error;
	size_t size;
	FILE *pFile;
	int index;
	size = NO_GEN*NO_TESTS*sizeof(float);
	error = cudaMemcpy(h_avg_fitness, d_avg_fitness, size, cudaMemcpyDeviceToHost);

	// May as well save the test data too
	pFile = fopen("TestHist.txt", "w");
	if (pFile == NULL) {
		printf("Error opening TestHist.txt for writing\n");
	} else {
		index = 0;
		for (int i = 0; i < NO_TESTS; i++) {
			for (int j = 0; j < NO_GEN; j++) {
				fprintf(pFile, "%g", h_avg_fitness[index]);
				if (j == (NO_GEN-1)) {
					fprintf(pFile, "\n");
				} else {
					fprintf(pFile, "\t");
				}
				index++;
			}
		}
	}



}


void Allocate_Memory() {

	size_t size;
	cudaError_t error;

        if (DEBUG) printf("  Allocating memory for %d kids and %d variables\n", NO_KIDS, NO_VAR);


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
	size = NO_GEN*NO_VAR*sizeof(float);
	h_x_history = (float*)malloc(size);
	size = NO_GEN*NO_TESTS*sizeof(float);
	h_avg_fitness = (float*)malloc(size);

	// Allocate memory on device
	size = NO_KIDS*NO_VAR*sizeof(float);
	error = cudaMalloc((void**)&d_x, size);
	if (DEBUG) printf("  -Memory Allocation of d_x - Error = %s\n", cudaGetErrorString(error));
        error = cudaMalloc((void**)&d_x_new, size);
        if (DEBUG) printf("  -Memory Allocation of d_x_new - Error = %s\n", cudaGetErrorString(error));

	size = NO_KIDS*sizeof(float);
	error = cudaMalloc((void**)&d_fitness, size);
        if (DEBUG) printf("  -Memory Allocation of d_fitness - Error = %s\n", cudaGetErrorString(error));

	size = 2*sizeof(float);
	error = cudaMalloc((void**)&d_Info, size);
        if (DEBUG) printf("  -Memory Allocation of d_Info - Error = %s\n", cudaGetErrorString(error));

	size = NO_KIDS*sizeof(float);
	error = cudaMalloc((void**)&d_fitness_new, size);
        if (DEBUG) printf("  -Memory Allocation of d_fitness_new - Error = %s\n", cudaGetErrorString(error));


	size = NO_KIDS*sizeof(curandState);
	error = cudaMalloc((void**)&d_state, size);
        if (DEBUG) printf("  -Memory Allocation of d_state - Error = %s\n", cudaGetErrorString(error));


	size = NO_GEN*sizeof(float);
	error = cudaMalloc((void**)&d_fitness_history, size);
        if (DEBUG) printf("  -Memory Allocation of d_fitness_history - Error = %s\n", cudaGetErrorString(error));

	size = NO_GEN*NO_VAR*sizeof(float);
	error = cudaMalloc((void**)&d_x_history, size);
        if (DEBUG) printf("  -Memory Allocation of d_x_history - Error = %s\n", cudaGetErrorString(error));

	size = NO_GEN*NO_TESTS*sizeof(float);
	error = cudaMalloc((void**)&d_avg_fitness, size);
        if (DEBUG) printf("  -Memory Allocation of d_avg_fitness - Error = %s\n", cudaGetErrorString(error));


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

	if (DEBUG) printf("  Computing New Generation\n");
	// Compute the next generation properties
	ComputeNewGenerationGPU<<<BPG,TPB>>>(d_fitness, d_fitness_new, d_x, d_x_new, d_state, NO_KIDS);

}


void UpdateNextGeneration() {

	size_t size;

	if (DEBUG) printf("  Updating next generation\n");

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
	if (DEBUG) printf("  Finding best child in current generation\n");
	FindBestGPU<<<1,MAX_TPB>>>(d_fitness, d_Info, d_fitness_history, d_x_history, d_x, d_avg_fitness, NO_KIDS);

	// Copy values across for debugging
	//cudaMemcpy(&h_BestKid, &d_BestKid, sizeof(int), cudaMemcpyDeviceToHost);
	//cudaMemcpy(&h_BestFitness, &d_BestFitness, sizeof(float), cudaMemcpyDeviceToHost);
	if (DEBUG) {
		error = cudaMemcpy(h_Info, d_Info, 2*sizeof(float), cudaMemcpyDeviceToHost);
		//printf("  -Primary: Index  = %d, Fitness = %g\n", h_BestKid, h_BestFitness);
		//printf("  -CUDA Error on Copy: %s\n", cudaGetErrorString(error));
		printf("  -Best Kid: Index = %g, Fitness = %g\n", h_Info[0], h_Info[1]);
	}

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

	printf("   Freeing memory\n");

	// Free host memory
	free(h_x); free(h_fitness); free(h_Info); free(h_fitness_new); free(h_fitness_history); free(h_avg_fitness);
	// Free GPU memory
	error = cudaFree(d_x);
	if (DEBUG) printf("  -Deallocation of d_x: Error = %s\n",	cudaGetErrorString(error));
        error = cudaFree(d_x_new);
        if (DEBUG) printf("  -Deallocation of d_x_new: Error = %s\n",  cudaGetErrorString(error));
	error = cudaFree(d_fitness);
        if (DEBUG) printf("  -Deallocation of d_fitness: Error = %s\n",  cudaGetErrorString(error));
	error = cudaFree(d_Info);
	if (DEBUG) printf("  -Deallocation of d_Info: Error = %s\n",  cudaGetErrorString(error));
	error = cudaFree(d_fitness_new);
        if (DEBUG) printf("  -Deallocation of d_fitness_new: Error = %s\n",  cudaGetErrorString(error));
	error = cudaFree(d_state);
        if (DEBUG) printf("  -Deallocation of d_state: Error = %s\n",  cudaGetErrorString(error));
	error = cudaFree(d_fitness_history);
        if (DEBUG) printf("  -Deallocation of d_fitness_history: Error = %s\n",  cudaGetErrorString(error));
	error = cudaFree(d_x_history);
        if (DEBUG) printf("  -Deallocation of d_x_history: Error = %s\n",  cudaGetErrorString(error));
	error = cudaFree(d_avg_fitness);
        if (DEBUG) printf("  -Deallocation of d_avg_history: Error = %s\n",  cudaGetErrorString(error));


}

void TicGPU() {
	// Simple cuda event timer
	cudaEventRecord(start);
	// Use the wall clock as well
	gettimeofday(&start2, NULL);

}

void TocGPU() {
	// Snycronize
	cudaDeviceSynchronize();

	// Record end
	cudaEventRecord(stop);
	// CPU clock too
	gettimeofday(&stop2, NULL);
	// Report the time
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("CUDA Elapsed time = %g microseconds\n", 1000.0*milliseconds);
	timersub(&stop2, &start2, &result);
	printf("CPU Elapsed time = %ld microseconds\n", result.tv_usec);
}



void Init() {
	// Set up the initial values of x for each variable and each child
	int seed = time(NULL);
	int i, j, index;
	int size[2];
	FILE *pFile;

	// Initialize the seed for the random number generator
	srand(seed);

	// Create the cuda timer events
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

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
	int i, j;
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
			fprintf(pFile, "%d\t", i);
			for (j = 0; j < NO_VAR; j++) {
				fprintf(pFile, "%g\t", h_x_history[i*NO_VAR + j]);
			}
			fprintf(pFile, "%g\n", i, h_fitness_history[i]);
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
	if (DEBUG) printf("   Sending data from GPU to Host\n");
        error = cudaMemcpy(h_fitness, d_fitness, size, cudaMemcpyDeviceToHost);
        if (DEBUG) printf("  -Memcpy to host: d_fitness: Error = %s\n", cudaGetErrorString(error));

	error = cudaMemcpy(h_fitness_new, d_fitness_new, size, cudaMemcpyDeviceToHost);
	if (DEBUG) printf("  -Memcpy to host: d_fitness_new: Error = %s\n", cudaGetErrorString(error));

	// Copy the values of X across too
	size = NO_KIDS*NO_VAR*sizeof(float);
        error = cudaMemcpy(h_x, d_x, size, cudaMemcpyDeviceToHost);
        if (DEBUG) printf("  -Memcpy to host: d_x: Error = %s\n", cudaGetErrorString(error));

	// Copy the history across
	size = NO_GEN*sizeof(float);
	error = cudaMemcpy(h_fitness_history, d_fitness_history, size, cudaMemcpyDeviceToHost);
        if (DEBUG) printf("  -Memcpy to host: d_fitness_history: Error = %s\n", cudaGetErrorString(error));


	size = NO_GEN*NO_VAR*sizeof(float);
	error = cudaMemcpy(h_x_history, d_x_history, size, cudaMemcpyDeviceToHost);
        if (DEBUG) printf("  -Memcpy to host: d_x_history: Error = %s\n", cudaGetErrorString(error));


	// Debug time - find best
	BestFitness = 10000.0;
	BestIndex = -1;

	for (i = 0; i < NO_KIDS; i++) {
		if (fabs(h_fitness[i]) < fabs(BestFitness)) {
			BestFitness = h_fitness[i];
			BestIndex = i;
		}
	}
	printf("Best Index = %d, Best Fitness = %g\n", BestIndex, BestFitness);
	printf("X Values for best = ");
	for (var = 0; var < NO_VAR; var++) {
		printf("%g, ", h_x[BestIndex*NO_VAR + var]);
	}
	printf("\n");

	// Debugging - copy info
}
