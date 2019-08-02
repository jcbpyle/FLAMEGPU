# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 16:48:27 2019

@author: James
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

def same_pc():
	original_experiment = open("iterations/simulation_results.csv","r")
	original_results = original_experiment.readlines()
	original_experiment.close()
	reproduced_experiment = open("reproduced_iterations/simulation_results.csv","r")
	reproduced_results = reproduced_experiment.readlines()
	reproduced_experiment.close()
	#print([x.rstrip().split(",") for x in original_results])
	print([x.rstrip().split(",") for x in original_results if not(x=="\n")])
	#print([x.rstrip().split(",")[1] for x in reproduced_results])
	for i in range(len(original_results[0].rstrip().split(","))):
		if not (i==0 or i==9):
			if not (i==4 or i==5 or i==11 or i==12):
				orig = [int(x.rstrip().split(",")[i]) for x in original_results if not(x=="\n")]
				repro = [int(x.rstrip().split(",")[i]) for x in reproduced_results if not(x=="\n")]
				minp = min(min(orig),min(repro))
				maxp = max(max(orig),max(repro))
				plt.figure(figsize=(10, 10)) 
				plt.plot(orig,repro, 'bo')
				plt.plot(range(minp,maxp),range(minp,maxp))
				plt.title("temp title")
				plt.show()
				plt.close()
			else:
				orig = [float(x.rstrip().split(",")[i]) for x in original_results if not(x=="\n")]
				repro = [float(x.rstrip().split(",")[i]) for x in reproduced_results if not(x=="\n")]
#				minp = int(min(min(orig),min(repro)))
#				maxp = int(max(max(orig),max(repro)))
#				plt.figure(figsize=(10, 10)) 
#				plt.plot(orig,repro, 'bo')
#				plt.plot(range(minp,maxp),range(minp,maxp))
#				plt.title("temp title")
#				plt.show()
#				plt.close()
			
#	parameter1 = [[int(x.rstrip().split(",")[1]) for x in original_results if not(x=="\n")],[int(x.rstrip().split(",")[1]) for x in reproduced_results if not(x=="\n")]]
#	parameter2 = [[int(x.rstrip().split(",")[2]) for x in original_results if not(x=="\n")],[int(x.rstrip().split(",")[2]) for x in reproduced_results if not(x=="\n")]]
#	parameter3 = [[int(x.rstrip().split(",")[3]) for x in original_results if not(x=="\n")],[int(x.rstrip().split(",")[3]) for x in reproduced_results if not(x=="\n")]]
#	parameter4 = [[float(x.rstrip().split(",")[4]) for x in original_results if not(x=="\n")],[float(x.rstrip().split(",")[4]) for x in reproduced_results if not(x=="\n")]]
#	parameter5 = [[float(x.rstrip().split(",")[5]) for x in original_results if not(x=="\n")],[float(x.rstrip().split(",")[5]) for x in reproduced_results if not(x=="\n")]]
#	parameter6 = [[int(x.rstrip().split(",")[6]) for x in original_results if not(x=="\n")],[int(x.rstrip().split(",")[6]) for x in reproduced_results if not(x=="\n")]]
#	parameter7 = [[int(x.rstrip().split(",")[7]) for x in original_results if not(x=="\n")],[int(x.rstrip().split(",")[7]) for x in reproduced_results if not(x=="\n")]]
#	parameter8 = [[int(x.rstrip().split(",")[8]) for x in original_results if not(x=="\n")],[int(x.rstrip().split(",")[8]) for x in reproduced_results if not(x=="\n")]]
#	fitness1 = [[int(x.rstrip().split(",")[10]) for x in original_results if not(x=="\n")],[int(x.rstrip().split(",")[10]) for x in reproduced_results if not(x=="\n")]]
#	fitness2 = [[float(x.rstrip().split(",")[11]) for x in original_results if not(x=="\n")],[float(x.rstrip().split(",")[11]) for x in reproduced_results if not(x=="\n")]]
#	fitness3 = [[float(x.rstrip().split(",")[12]) for x in original_results if not(x=="\n")],[float(x.rstrip().split(",")[12]) for x in reproduced_results if not(x=="\n")]]
	return

same_pc()