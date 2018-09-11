# GAGPU
A Parallel Genetic Algorithm using CUDA for GPU Computing

## Summary

The algorithm is outlined in the paper below:

Smith, M.R., Kuo, F.-A., Hsieh, C.-W., Yu, J.-P., Wu, J.-S. and Ferguson, A., Rapid optimization of blast wave mitigation strategies using Quiet Direct Simulation and Genetic Algorithm, Computer Physics Communications, 181[6], 2010.

A brief summary of the approach used is as follows: consider a parameter X which contributes in some unknown way to a fitness function F(X). A population of N children each store their own value of X - unique to each child. The N children together make up a single generation of children - the characteristics of each child (i.e. the values of X) are used in the creation of new generation of children, loosely based around the ideas of genetics.

In the case where we wish to maximize the value of the fitness, one of these children will have a higher value than the others - we shall call this X1. When creating new children, we choose another parent randomly from the population - X2 - and a third parent is chosen for the sake of measuring the populations’ current variance (X3). When combined, the value of the new child’s XC is:

      XC = (GF)*X1 + (1-GF)*X2 + SIGMA*RN*(X2-X3)
      
where GF is:

      GF = FR*MAX(RF, 1-RF)
      
where Rf is a uniformly generated random fraction (between 0 and 1) and RN is a normally distributed random variable from a normal distribution with a mean of 0 and a variance of 1. The value of FR - normally equal to one - may be used to accelerate convergence by increasing its value, while sigma - attached to the mutation term - may be increased to increase stability of convergence.
The fitness is then computed for each child; if it is found to be an improvement over that parent previously occupying its space in memory, we choose its value of X and associated fitness and save them as part of the next generation information. Otherwise, the parent information is passed on. This process is repeated across all individuals from generation to generation, the result being generations of children which have (eventually) higher levels of fitness with lower in-generation variance.

## Key Code Variables
A brief description of key variables and constants is found below. NOTE: In general,variables beginning with d_ are stored in GPU memory while h_ are stored in host (CPU) memory. UPPER_CASE constants are #defined values which are replaced at compile time.

### config.h

- NO_GEN	No. of Genetic Algorithm generations to run. Increase if problem is not converged.
- TPB		Threads Per Block executed within each Streaming Multiprocessor on the GPU
- BGP		The number of thread blocks to use on the device. 
- NO_KIDS	Number of children in each generation of the GA. Currently set such that each 
streaming processor in use on the GPU controls a single child.
- NO_VAR	Number of parameters required to be solved in our optimization problem.
- DEBUG	Set to 1 to turn on CUDA and debugging messages during runtime.
- FR	Over-convergence parameter on crossover computation. Standard value is 1.0; can be increased to accelerate convergence, which also creates dangers.
- SIG	Mutation parameter, larger values encourage larger mutations and prevent capture in local minima while also slowing convergence. 
gpu_main.cu
- d_x	One dimensional arrays holding our solution parameters. (d_x for GPU, h_x for host)
- d_x_new	New (next generation) values of x. Computed and stored between generations.
- d_fitness	Current fitness as computed according to the values 
- d_fitness_new	New (next generation) values of fitness, computed according to x_new.
- x_min, m_max	Two arrays of length NO_VAR which contain the minimum and maximum values of x for each variable. These are only used during initialization; they are not used to enforce the ranges of x during computation.
- d_fitness_history		History of the best fitness, one dimensional array NO_GEN in length.
- d_x_history	History of the best values of x, one dimensional array of length NO_GEN*NO_VAR.
- d_BestKid	Index (i.e. name) of current best child in its generation. Integer.
- d_BestFitness	The fitness of the current best child in its generation.
- d_state	cuRAND random number generator state; each thread requires its own state.

## Function Descriptions

### Host Functions / Wrapping Functions

- Allocate_Memory()	Allocate memory on host and device. Perform prior to computation.
- Free_Memory()		Free memory on host and device. Last step prior to exit() of program.
- ComputeFitness(float *x, int N)	Compute fitness on CPU using CPU held data.
- ComputeNewGenCall()		Call GPU kernel (ComputeNewGenerationGPU) which computes the 
              new fitness and new values of x.
- UpdateNextGeneration()	Overwrites values of _x and _fitness on device with _x_new and  
              _fitness_new.
- FindBestGPUCall()		Calls GPU kernel (FindBestGPU) which uses a single block with 
              MAX_TPB threads to find the current best child.
- SetupRfGPUCall()		Calls GPU kernel (SetupRFGPU) which initializes the random number 
              generator engine state for each thread. 
- SetupGPUCall()		Calls GPU kernel (SetupGPU)  which primarily computes the fitness 
              for each child in the starting (0th) generation.
- Init()		Creates the initial values of x (solution parameters) for each child on 
              the host, and saves them to file.
- Randf()	Generate a random fraction between 0 and 1. 
- RandNf()	Generate a normally distributed random variable from a normal distribution with a mean of 0 and a variance of 1.
- SaveFitness()	Saves the fitness data for each child after NO_GEN generations. Also saves to file the history data for analysis of its convergence.

### GPU Kernels / Device Functions

- MaxFrac()	Device function (single thread). Generates the maximum of A or (1-A). 
- ComputeFitnessGPU()	Device kernel (multi-thread). Computes the fitness for each thread using the parameters contained in d_x.
	Note: This is the part of the code which primarily needs to be modified in order to customize this code to specific applications. 
	Note: The code currently solves minimization problems.
- ComputeNewGenerationGPU()	Device kernel (multi-threaded). Compute the next generation value for each thread (i.e. child) in the current generation.
	Note: We assume here that a lower fitness value is optimal, i.e. minimization. In the event maximization is required, the line of code which reads:
		if (fabs(fitness_new[i] < fabs(fitness[i])) {
	should be changed to read:
		if (fabs(fitness_new[i]) > fabs(fitness[i])) {
- SetupGPU()		Device kernel (multi-threaded). Computes the fitness for each child using 
 			d_x,  and initializes the on-device generation counter (d_Step).
- SetupRfGPU()	Device kernel (multi-threaded). Initializes the CUDA random number generator state for each thread.
- FindBestGPU()	Device kernel (multi-threaded). Compute the best child in the current generation using info held in d_fitness. Also saves the fitness history for each generation and increments the d_Step variable.

## Contact Information
Name: Dr. Matthew Smith

Address: Center for Astrophysics and Supercomputing (CAS), Swinburne University of Technology.

Email: msmith@astro.swin.edu.au




