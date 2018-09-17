#Loading data and plotting routines
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import numpy as np


# Try loading data
data = np.loadtxt("TestHist.txt", delimiter="\t")

#Specific Info about the test case
no_gen = 30
no_tests = 1000
solution = 0.0   # Ackley Function Global Minimum

#Setup our average
avg = np.zeros([no_gen])

#Compute the average
for i in range(no_tests):
	for j in range(no_gen):
		avg[j] = avg[j] + (data[i][j]-solution)/no_tests


#Set up variable for printing
gen = np.arange(0, no_gen, 1)

#Plot data
fig, ax =  plt.subplots()
#ax.plot(gen,avg)
ax.semilogy(gen,avg)

ax.set(xlabel='Generation',ylabel='Error', title='Average convergence')
ax.grid()

fig.savefig("Test.png")
plt.show()


