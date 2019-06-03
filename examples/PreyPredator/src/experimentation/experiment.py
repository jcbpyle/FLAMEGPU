
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
import sys
import threading
import queue
import random
import datetime
import pycuda.driver as cuda
import pycuda.autoinit

BASE_DIRECTORY = os.getcwd()

#Yoooooooo

#Initial state file creation.
def initial_state_creation(file_name,agent_information):
	SAVE_DIRECTORY = BASE_DIRECTORY+"../..//iterations"+"/"
	SAVE_DIRECTORY = BASE_DIRECTORY+"/"
	initial_state_file = open(SAVE_DIRECTORY+str(file_name)+".xml","w")
	initial_state_file.write("<states>\n<itno>0</itno>\n<environment>\n")
	initial_state_file.write("<INTERACTION_DISTANCE_TEST_VARIABLE>"+str(0.12345)+"</INTERACTION_DISTANCE_TEST_VARIABLE>\n")
	initial_state_file.write("<REPRODUCE_PREY_PROB>"+str(random.uniform(0,0.25))+"</REPRODUCE_PREY_PROB>\n")
	initial_state_file.write("<REPRODUCE_PREDATOR_PROB>"+str(random.uniform(0,0.25))+"</REPRODUCE_PREDATOR_PROB>\n")
	initial_state_file.write("<GAIN_FROM_FOOD_PREY>"+str(int(random.uniform(0,500)))+"</GAIN_FROM_FOOD_PREY>\n")
	initial_state_file.write("<GAIN_FROM_FOOD_PREDATOR>"+str(int(random.uniform(0,500)))+"</GAIN_FROM_FOOD_PREDATOR>\n")
	initial_state_file.write("<GRASS_REGROW_CYCLES>"+str(int(random.uniform(0,500)))+"</GRASS_REGROW_CYCLES>\n")
	
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

base_agent_information = [
["prey",["initial_population",0,5000],["x",-1.0,1.0],["y",-1.0,1.0],["type",1,1],["fx",-1.0,1.0],["fy",-1.0,1.0],["steer_x",0.0,0.0],["steer_y",0.0,0.0],["life",1,50],],
["predator",["initial_population",0,5000],["x",-1.0,1.0],["y",-1.0,1.0],["type",1,1],["fx",-1.0,1.0],["fy",-1.0,1.0],["steer_x",0.0,0.0],["steer_y",0.0,0.0],["life",1,50],],
["grass",["initial_population",0,5000],["x",-1.0,1.0],["y",-1.0,1.0],["type",2,2],["dead_cycles",0,0],["available",1,1],],]

initial_state_creation("0",base_agent_information)
