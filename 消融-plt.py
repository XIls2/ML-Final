import matplotlib.pyplot as plt
import numpy as np

x=['top-1 acc','top-5 acc','Best top-1 acc','Mean top-1 acc']
# y1=[17.65,63.71,18.70,16.96]
# y2=[70.14,94.55,70.14,61.23]
# y3=[84.90,99.00,84.90,79.39]
# y4=[90.44,99.61,90.44,87.71]

y1=[56.49,93.59,56.49,53.83]
y2=[84.90,99.00,84.90,79.39]
y3=[81.27,98.56,81.27,72.99]
y4=[65.81,96.28,65.81,58.97]

# y1=[84.90,99.00,84.90,79.39]
# y2=[77.91,98.41,77.91,69.54]
# y3=[48.40,90.63,48.40,30.98]
# y4=[10.00,50.00,10.00,10.00]

plt.plot(x,y1,marker = '<',color = 'dodgerblue', alpha=0.5, linewidth=1, label='learning-rate:0.003')
plt.plot(x,y2,marker = '>',color = 'darkorange', alpha=0.5, linewidth=1, label='learning-rate:0.03')
plt.plot(x,y3,marker = 'v',color = 'green', alpha=0.5, linewidth=1, label='learning-rate:0.3')
plt.plot(x,y4,marker = '^',color = 'red', alpha=0.5, linewidth=1, label='learning-rate:0.9')

plt.legend(['learning-rate:0.003','learning-rate:0.03','learning-rate:0.3','learning-rate:0.9'],loc='best')

plt.show()