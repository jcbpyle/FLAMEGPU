
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

import os
import pycuda.driver as cuda
import pycuda.autoinit

BASE_DIRECTORY = os.getcwd()
GPUS_AVAILABLE = cuda.Device(0).count()

#InitialStates

#Initial state file creation for experiment predprey.
def initial_state_creation_predprey(file_name,agent_information):
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

#Agent data stored in list of lists
base_agent_information = [
["prey",["initial_population",0,5000],["x",-1.0,1.0],["y",-1.0,1.0],["type",1,1],["fx",-1.0,1.0],["fy",-1.0,1.0],["steer_x",0.0,0.0],["steer_y",0.0,0.0],["life",1,50],],
["predator",["initial_population",0,5000],["x",-1.0,1.0],["y",-1.0,1.0],["type",1,1],["fx",-1.0,1.0],["fy",-1.0,1.0],["steer_x",0.0,0.0],["steer_y",0.0,0.0],["life",1,50],],
["grass",["initial_population",0,5000],["x",-1.0,1.0],["y",-1.0,1.0],["type",2,2],["dead_cycles",0,0],["available",1,1],],]

#Create initial state
#initial_state_creation_predprey("0",base_agent_information)

#ExperimentSet

############## testing_initial_state ############

############## testing_batch_simulation ############

#Run for desired number of repeats
#for i in range(10):
	#initial_state_creation_predprey(file_name,base_agent_information)
	#Run simulation
	#os.system(simulation_command)
	#Parse results
	#results_file = open("../..//iterations","r")
	#results = results_file.readlines()
	#results_file.close()

############## testing_ga_experiment ############
mu = int(100)
LAMBDA = int(10)
max_time = int(60)
max_generations = int(10)
mutation = float(0.25)
crossover = float(0.5)

def fitness_function(primary,secondary,tertiary,placeholder=None):
	fitness = None

	#Model executable
	#executable = ""
	#simulation_command = ""
	#if os.name=='nt':
	#	executable = "../../../../../bin/x64/Release_Console//PreyPredator_api_test.exe"
	#	simulation_command = executable+" ../..//iterations/0.xml 1000"
	#else:
	#	executable = "./../../../../../bin/x64/Release_Console//PreyPredator_api_test"
	#	simulation_command = executable+" ../..//iterations/0.xml 1000"

	
	#Initial state creator
	
	#Run for desired number of repeats
	#for i in range(1):
		#initial_state_creation_predprey(file_name,base_agent_information)
		#Run simulation
		#os.system(simulation_command)
		#Parse results
		#results_file = open("../..//iterations","r")
		#results = results_file.readlines()
		#results_file.close()
	

	
	return fitness 

def run_ga(mu,lamb,gen,time,start,evals,placeholder=None):
	global curr_pop, 
	population = None

	#Model executable
	#executable = ""
	#simulation_command = ""
	#if os.name=='nt':
	#	executable = "../../../../../bin/x64/Release_Console//PreyPredator_api_test.exe"
	#	simulation_command = executable+" ../..//iterations/0.xml 1000"
	#else:
	#	executable = "./../../../../../bin/x64/Release_Console//PreyPredator_api_test"
	#	simulation_command = executable+" ../..//iterations/0.xml 1000"

	
	#Initial state creator
	
	#Run for desired number of repeats
	#for i in range(1):
		#initial_state_creation_predprey(file_name,base_agent_information)
		#Run simulation
		#os.system(simulation_command)
		#Parse results
		#results_file = open("../..//iterations","r")
		#results = results_file.readlines()
		#results_file.close()
	

	
	return population 

############## testing_surrogate_experiment ############
hidden_layers = tuple(100,100)
error = float(1e-9)
max_time = int(60)
max_training_generations = int(2000)
mutation = float(0.25)
crossover = float(0.5)

def fitness_function(primary,secondary,tertiary,placeholder=None):
	fitness = None

	#Model executable
	#executable = ""
	#simulation_command = ""
	#if os.name=='nt':
	#	executable = "../../../../../bin/x64/Release_Console//PreyPredator_api_test.exe"
	#	simulation_command = executable+" ../..//.xml 1000"
	#else:
	#	executable = "./../../../../../bin/x64/Release_Console//PreyPredator_api_test"
	#	simulation_command = executable+" ../..//.xml 1000"

	
	#Run simulation
	#os.system(simulation_command)
	#Parse results
	#results_file = open("../../","r")
	#results = results_file.readlines()
	#results_file.close()
	

	
	return fitness 
