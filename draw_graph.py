# coding: utf-8
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import json

file_names = [
    '3MLP_100_12class',
    '3MLP_12class',
    '3MLP_200_12class',
    'CNN_12class'
]

data = {}

for name in file_names:
	f = open("result/" + name)
	data[name] = json.load(f)
	f.close()

markers = {
    '3MLP_100_12class': "o",
    '3MLP_12class': "v",
    '3MLP_200_12class': "8",
    'CNN_12class': "1"
}

for name in data:
	plt.plot(
	list(map(lambda x:x["epoch"], data[name])), 
	list(map(lambda x:x["main/accuracy"], data[name])),
	marker=markers[name], markevery=100, label=name )

#plt.plot(data[:,0], data[:,1])
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.legend(loc="lower right")
plt.savefig('result/graph.png', bbox_inches='tight')
