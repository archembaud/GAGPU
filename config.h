#define NO_GEN 30
#define TPB 64
#define BPG 60
#define NO_KIDS (TPB*BPG)
#define NO_VAR 2
#define DEBUG 0
#define PI_F 3.141592654
#define MAX_TPB 960 // Used in reduction, should be NO_KIDS/4
#define FR 1.0
#define SIG 1.5

void Allocate_Memory();
void Free_Memory();
void Init();
float Randf();
float RandNf();
void SaveFitness();
void SendToGPU();
void SendToHost();
void SetupGPUCall();
void FindBestGPUCall();
void SetupRfGPUCall();
void ComputeNewGenCall();
void UpdateNextGeneration();
void TicGPU();
void TocGPU();
