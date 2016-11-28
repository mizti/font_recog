# coding: utf-8
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import json

file_names = [
  '3MLP_12class_log',
  '3MLP_6class_log'
]

data = {}

for name in file_names:
	f = open("result" + name)
	data["name"] = json.load(f)
	f.close()


plt.plot(data[:,0], data[:,1])
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.savefig('result/3MLP.png', bbox_inches='tight')
