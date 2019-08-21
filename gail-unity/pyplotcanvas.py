import matplotlib.pyplot as plt
import numpy as np 
from matplotlib import rcParams

#font = {'fontname':'Serif'}

x = np.arange(0,50)
y = np.random.rand(50)*10

#plt.axis([0, 50, 0, 10])

rcParams['font.family'] = 'sans-serif'
rcParams['font.serif'] = ['Tahoma']

#np.savetxt('xdata.txt', x, delimiter=',')   # X is an array
np.savetxt('ydata.txt', y, delimiter=',')   # Y is an array

#x = np.loadtxt('xdata.txt')
#y = np.loadtxt('ydata.txt')
y = np.loadtxt('GAILTrainingRewards.txt')
x = np.arange(len(y))

for i in range(len(x)):
		print(x[i]," | ",y[i])

#equal scaling
#plt.axis('equal')
#plt.gca().set_aspect('equal', adjustable='box')

plt.plot(x, y)

#plt.xlabel('Training Time',**font)
plt.xlabel('Step (Testing)')
plt.ylabel('Cumulative Reward')

plt.title('Test Graph')

plt.savefig('graph.png')

plt.show()