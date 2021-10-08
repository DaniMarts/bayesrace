
__author__ = 'Danilo Martins'
__email__ = 'mrtdan014@myuct.ac.za'

import matplotlib.pyplot as plt
import numpy as np
import csv

x, y = [], []

with open("map4.csv") as file:
	data = list(csv.reader(file, delimiter=","))
	for row in data:
		x.append(np.float(row[0]))
		y.append(np.float(row[1]))

plt.plot(x, y)
plt.show()