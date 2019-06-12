
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

import sys

import os
import random
import itertools
import pycuda.driver as cuda
import pycuda.autoinit

BASE_DIRECTORY = os.getcwd()+"/"
PROJECT_DIRECTORY = BASE_DIRECTORY+"../../"
GPUS_AVAILABLE = cuda.Device(0).count()

#InitialStates


initial_state_files = []

initial_state_files += ["/example_iterations/example.xml"]

#Initial state generator function to be created by the user
def initial_state_generator_function_example_user_generator_function():

	return

#Initial state file creation.
def initial_state(save_location,file_name,global_information,agent_information):
	SAVE_DIRECTORY = PROJECT_DIRECTORY+"example_iterations"+"/"
	if not os.path.exists(SAVE_DIRECTORY+str(save_location)):
		os.mkdir(SAVE_DIRECTORY+str(save_location))
	print(SAVE_DIRECTORY,save_location,file_name)
	initial_state_file = open(SAVE_DIRECTORY+str(save_location)+"/"+str(file_name)+".xml","w")
	initial_state_file.write("<states>\n<itno>0</itno>\n<environment>\n")
	if len(global_information)>0:
		for g in range(len(global_information)):
			initial_state_file.write("<"+str(global_information[g][0])+">"+str(gloabl_information[g][1])+"</"+str(global_information[g][0])+">\n")
	initial_state_file.write("</environment>\n")
	if len(agent_information)>0:
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

#Generate initial states based on defined ranges/lists/values for all global and agent population variables for experiment example_generator.
def generate_initial_states_example_generator():
	global_data = []
	agent_data = []
	
	print("global_data",global_data)
	print()
	global_data.sort(key=len)
	agent_data.sort(key=len)
	prefix = "seed"
	file_name = str(prefix)+"/0"
	parameter_count = 0
	if len(global_data)>0:
		constructed_data = [x for y in global_data for x in y[1]]
		for current,others in itertools.combinations(global_data,2):
			print(current,others)
			for i in current[1]:
				for j in others[1]:
					print("outer loop parameter",i,current)
					print("inner loop parameter",j,others)
					current_global = []
					current_agent = []
					#initial_state(str(prefix),"0",current_global,current_agent)
					prefix = prefix+"seed"
					print(prefix)
	return global_data,agent_data

generate_initial_states_example_generator()
#Agent data stored in list of lists
base_agent_information = []

#Create initial state
#initial_state_creation_example_generator("",base_agent_information)

#Generate initial states based on defined ranges/lists/values for all global and agent population variables for experiment example_generator_1param_list.
def generate_initial_states_example_generator_1param_list():
	global_data = []
	agent_data = []
	global_data += [["LIST_TEST_VARIABLE", [1,2,3,4,5]]]
	
	print("global_data",global_data)
	print()
	global_data.sort(key=len)
	agent_data.sort(key=len)
	prefix = "1"
	file_name = str(prefix)+"/0"
	parameter_count = 0
	if len(global_data)>0:
		constructed_data = [x for y in global_data for x in y[1]]
		for current,others in itertools.combinations(global_data,2):
			print(current,others)
			for i in current[1]:
				for j in others[1]:
					print("outer loop parameter",i,current)
					print("inner loop parameter",j,others)
					current_global = []
					current_agent = []
					#initial_state(str(prefix),"0",current_global,current_agent)
					prefix = prefix+"1"
					print(prefix)
	return global_data,agent_data

generate_initial_states_example_generator_1param_list()
#Agent data stored in list of lists
base_agent_information = []

#Create initial state
#initial_state_creation_example_generator_1param_list("",base_agent_information)

#Generate initial states based on defined ranges/lists/values for all global and agent population variables for experiment example_generator_1param_range.
def generate_initial_states_example_generator_1param_range():
	global_data = []
	agent_data = []
	global_data += [["RANGE_TEST_VARIABLE", range(1,5,1)]]
	
	print("global_data",global_data)
	print()
	global_data.sort(key=len)
	agent_data.sort(key=len)
	prefix = "1"
	file_name = str(prefix)+"/0"
	parameter_count = 0
	if len(global_data)>0:
		constructed_data = [x for y in global_data for x in y[1]]
		for current,others in itertools.combinations(global_data,2):
			print(current,others)
			for i in current[1]:
				for j in others[1]:
					print("outer loop parameter",i,current)
					print("inner loop parameter",j,others)
					current_global = []
					current_agent = []
					#initial_state(str(prefix),"0",current_global,current_agent)
					prefix = prefix+"1"
					print(prefix)
	return global_data,agent_data

generate_initial_states_example_generator_1param_range()
#Agent data stored in list of lists
base_agent_information = []

#Create initial state
#initial_state_creation_example_generator_1param_range("",base_agent_information)

#Generate initial states based on defined ranges/lists/values for all global and agent population variables for experiment example_generator_1param_randomrange.
def generate_initial_states_example_generator_1param_randomrange():
	global_data = []
	agent_data = []
	global_data += [["RANGE_CHOOSE_TEST_VARIABLE", [(random.uniform(1,5)) for i in range(2)]]]
	
	print("global_data",global_data)
	print()
	global_data.sort(key=len)
	agent_data.sort(key=len)
	prefix = "1"
	file_name = str(prefix)+"/0"
	parameter_count = 0
	if len(global_data)>0:
		constructed_data = [x for y in global_data for x in y[1]]
		for current,others in itertools.combinations(global_data,2):
			print(current,others)
			for i in current[1]:
				for j in others[1]:
					print("outer loop parameter",i,current)
					print("inner loop parameter",j,others)
					current_global = []
					current_agent = []
					#initial_state(str(prefix),"0",current_global,current_agent)
					prefix = prefix+"1"
					print(prefix)
	return global_data,agent_data

generate_initial_states_example_generator_1param_randomrange()
#Agent data stored in list of lists
base_agent_information = []

#Create initial state
#initial_state_creation_example_generator_1param_randomrange("",base_agent_information)

#Generate initial states based on defined ranges/lists/values for all global and agent population variables for experiment example_generator_1param_randomlist.
def generate_initial_states_example_generator_1param_randomlist():
	global_data = []
	agent_data = []
	global_data += [["LIST_CHOOSE_TEST_VARIABLE", random.choices([1,2,3,4,5],k=2)]]
	
	print("global_data",global_data)
	print()
	global_data.sort(key=len)
	agent_data.sort(key=len)
	prefix = "1"
	file_name = str(prefix)+"/0"
	parameter_count = 0
	if len(global_data)>0:
		constructed_data = [x for y in global_data for x in y[1]]
		for current,others in itertools.combinations(global_data,2):
			print(current,others)
			for i in current[1]:
				for j in others[1]:
					print("outer loop parameter",i,current)
					print("inner loop parameter",j,others)
					current_global = []
					current_agent = []
					#initial_state(str(prefix),"0",current_global,current_agent)
					prefix = prefix+"1"
					print(prefix)
	return global_data,agent_data

generate_initial_states_example_generator_1param_randomlist()
#Agent data stored in list of lists
base_agent_information = []

#Create initial state
#initial_state_creation_example_generator_1param_randomlist("",base_agent_information)

#Generate initial states based on defined ranges/lists/values for all global and agent population variables for experiment example_generator_2param_list.
def generate_initial_states_example_generator_2param_list():
	global_data = []
	agent_data = []
	global_data += [["LIST_TEST_VARIABLE", [1,2,3,4,5]]]
	global_data += [["SECOND_LIST_TEST_VARIABLE", [99,98,97,96,95]]]
	
	print("global_data",global_data)
	print()
	global_data.sort(key=len)
	agent_data.sort(key=len)
	prefix = "1"
	file_name = str(prefix)+"/0"
	parameter_count = 0
	if len(global_data)>0:
		constructed_data = [x for y in global_data for x in y[1]]
		for current,others in itertools.combinations(global_data,2):
			print(current,others)
			for i in current[1]:
				for j in others[1]:
					print("outer loop parameter",i,current)
					print("inner loop parameter",j,others)
					current_global = []
					current_agent = []
					#initial_state(str(prefix),"0",current_global,current_agent)
					prefix = prefix+"1"
					print(prefix)
	return global_data,agent_data

generate_initial_states_example_generator_2param_list()
#Agent data stored in list of lists
base_agent_information = []

#Create initial state
#initial_state_creation_example_generator_2param_list("",base_agent_information)

#ExperimentSet

##############  ############
