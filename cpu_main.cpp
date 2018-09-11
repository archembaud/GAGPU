/*

GPU Accelerated Genetic Algorithm
Data


*/

#include <stdio.h>
#include "config.h"

int main() {

	// Allocate memory
	Allocate_Memory();

	// Initialize the problem
	Init();

	// Setup the RF generator on GPU
	SetupRfGPUCall();

	// Send it to the device
	SendToGPU();

	// Test / Debug
	FindBestGPUCall();

	// Compute the fitness
	SetupGPUCall();

	for (int gen = 0; gen < NO_GEN; gen++) {
		if (DEBUG) printf("=== Computing next generation\n");
		// Real Test
		FindBestGPUCall();

		// Compute New Generation
		ComputeNewGenCall();

		// Update generation
		UpdateNextGeneration();
	}

	// Send information to the host
	SendToHost();

	// Save the fitness to file
	SaveFitness();

	// Free Memory
	Free_Memory();

	return 0;
}
