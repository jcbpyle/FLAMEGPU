
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

import threading
import sys
import queue
import datetime

import os
import random
import pycuda.driver as cuda
import pycuda.autoinit

BASE_DIRECTORY = os.getcwd()+"/"
PROJECT_DIRECTORY = BASE_DIRECTORY+"../../"
GPUS_AVAILABLE = cuda.Device(0).count()

#InitialStates


initial_state_files = []

initial_state_files += ["/iterations/0.xml"]

#Initial state file creation.
def initial_state(save_location,file_name,global_information,agent_information):
	SAVE_DIRECTORY = PROJECT_DIRECTORY+"iterations"+"/"
	initial_state_file = open(SAVE_DIRECTORY+str(file_name)+".xml","w")
	initial_state_file.write("<states>\n<itno>0</itno>\n<environment>\n")
	for g in range(len(global_information)):
		initial_state_file.write("<"+str(global_information[g][0])+">"+str(gloabl_information[g][1])+"</"+str(global_information[g][0])+">\n")
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

#Generate initial states based on defined ranges/lists/values for all global and agent population variables for experiment predprey.
def generate_initial_states_predprey():
	global_data = []
	agent_data = []
	INTERACTION_DISTANCE_TEST_VARIABLE = 0.12345
	LIST_TEST_VARIABLE = [1,2,3,4,5]
	LIST_TEST_VARIABLE_2 = [1,2,3,4,5]
	LIST_TEST_VARIABLE_3 = random.choice([1,2,3,4,5,10],2)
	DISTRIBUTION_TEST_VARIABLE = random.uniform(1,10)
	DISTRIBUTION_TEST_VARIABLE_2 = random.sample(1,10,5)
	DISTRIBUTION_TEST_VARIABLE_3 = random.randRange(1,10,2)
	REPRODUCE_PREY_PROB = [random.uniform(0,0.25) for i in range(1)]
	REPRODUCE_PREDATOR_PROB = [random.uniform(0,0.25) for i in range(1)]
	GAIN_FROM_FOOD_PREY = [random.uniform(0,500) for i in range(1)]
	GAIN_FROM_FOOD_PREDATOR = [random.uniform(0,500) for i in range(1)]
	GRASS_REGROW_CYCLES = [random.uniform(0,500) for i in range(1)]
	
	return global_data,agent_data

#Agent data stored in list of lists
base_agent_information = []

#Create initial state
#initial_state_creation_predprey("",base_agent_information)

#ExperimentSet

############## testing_initial_State ############

############## testing_batch_simulation ############

#Run for desired number of repeats
#for i in range(10):
	#initial_state_creation_(file_name,base_agent_information)
	#Run simulation
	#os.system(simulation_command)
	#Parse results
	#results_file = open("../../","r")
	#results = results_file.readlines()
	#results_file.close()

############## testing_ga_experiment ############
mu = int(100)
LAMBDA = int(10)
Max_time = int(60)
Max_generations = int(10)
mutation = float(0.25)
crossover = float(0.5)

def fitness_function(primary,secondary,tertiary):
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

def run_ga(mu,lamb,gen,time,start,evals):
	global curr_pop
	Population = None

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
	
	
	return Population 

############## testing_surrogate_experiment ############
hidden_layers = tuple(100,100)
error = float(1e-9)
Max_time = int(60)
Max_training_generations = int(2000)
mutation = float(0.25)
crossover = float(0.5)

def fitness_function(primary,secondary,tertiary):
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
