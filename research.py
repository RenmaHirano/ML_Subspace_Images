import csv
import pprint
import numpy as np

filename = 'np_save.csv'
theta_array = np.loadtxt(filename,  delimiter=',')

is_correct_array = np.empty(282)

sum = 0
for i in range(282):
    if(np.argmax(theta_array[i][:]) // 3 == i // 6 ):
        is_correct_array[i] = 1
        sum += 1
    else:
        is_correct_array[i] = 0
            
correct_rate = sum / 282
print(correct_rate)
