
# Copyright 2019 The University of Sheffield
# Author: James Pyle
# Contact: jcbpyle1@sheffield.ac.uk
# Template experiment script file for FLAME GPU agent-based model
#
# University of Sheffield retain all intellectual property and 
# proprietary rights in and to this software and related documentation. 
# Any use, reproduction, disclosure, or distribution of this software 
# and related documentation without an express license agreement from
# University of Sheffield is strictly prohibited.
#
# For terms of licence agreement please attached licence or view licence 
# on www.flamegpu.com website.
#

import os
import threading
import sys
import queue
import random
import datetime

import pycuda.driver as cuda
import pycuda.autoinit

BASE_DIRECTORY = os.getcwd()

#Initial state file creation.
def initial_state_creation_test1(file_name,agent_information):
	SAVE_DIRECTORY = BASE_DIRECTORY+"../../"+"/"
	SAVE_DIRECTORY = BASE_DIRECTORY+"/"
	initial_state_file = open(SAVE_DIRECTORY+str(file_name)+".xml","w")
	initial_state_file.write("<states>\n<itno>0</itno>\n<environment>\n")
	
	initial_state_file.write("</environment>\n")
	for i in range(len(agent_information)):
		ind = [x[0] for x in agent_information[i]].index("initial_population")
		num_agents = int(random.uniform(agent_information[i][ind][1],agent_information[i][ind][2]))
		agent_id = 0
		agent_name = agent_information[i][0]
		for j in range(num_agents):
			initial_state_file.write("<xagent>\n")
			initial_state_file.write("<name>"+str(agent_name)+"</name>\n")
			initial_state_file.write("<id>"+str(agent_id)+"</id>\n")
			for k in agent_information[i]:
				if not (k[0]=="initial_population" or k==agent_name):
					if type(k[1])==type(int()):
						initial_state_file.write("<"+str(k[0])+">"+str(int(random.uniform(k[1],k[2])))+"</"+str(k[0])+">\n")
					elif type(k[1])==type(float()):
						initial_state_file.write("<"+str(k[0])+">"+str(random.uniform(k[1],k[2]))+"</"+str(k[0])+">\n")
			initial_state_file.write("</xagent>\n")
			agent_id += 1
	initial_state_file.write("</states>")
	return

base_agent_information = []

initial_state_creation("",base_agent_information)
