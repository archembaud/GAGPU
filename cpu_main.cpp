/*

GPU Accelerated Genetic Algorithm
Data


*/

#include <stdio.h>
#include "config.h"

int main() {

	// Allocate memory
	Allocate_Memory();

	// Setup the random number generator on the GPU
	SetupRfGPUCall();

	// Start our testing
	for (int test = 0; test < NO_TESTS; test++) {

		printf("*** Ensemble Average %d of %d ***\n", test, NO_TESTS);

		// Initialize the problem
		Init();

		// Send it to the device
		SendToGPU();

		// Compute the fitness
		SetupGPUCall();

		// Start the clocl
		TicGPU();

		for (int gen = 0; gen < NO_GEN; gen++) {
			if (DEBUG) printf("=== Computing generation %d of %d ===\n", gen+1, NO_GEN);
			// Real Test
			FindBestGPUCall();

			// Compute New Generation
			ComputeNewGenCall();

			// Update generation
			UpdateNextGeneration();
		}

		// Stop the clock
		TocGPU();

		// Add the the device step counter
		IncrementTestCall();

		// Send information to the host
		SendToHost();

	}

	// Send Test Data
	SendTestData();

	// Save the fitness to file
	SaveFitness();

	// Free Memory
	Free_Memory();

	return 0;
}
