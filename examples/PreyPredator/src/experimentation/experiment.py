
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
	agent_data += [["prey",["initial_population",[int(random.uniform(0,5000)) for i in range(1)]],
					["x",[float(random.uniform(-1.0,1.0)) for i in range(1)]],
					["y",[float(random.uniform(-1.0,1.0)) for i in range(1)]],
					["type",[1]],
					["fx",[float(random.uniform(-1.0,1.0)) for i in range(1)]],
					["fy",[float(random.uniform(-1.0,1.0)) for i in range(1)]],
					["steer_x",[0.0]],
					["steer_y",[0.0]],
					["life",range(1,50,10)]]]
	agent_data += [["predator",["initial_population",[int(random.uniform(0,5000)) for i in range(1)]],
					["x",[float(random.uniform(-1.0,1.0)) for i in range(1)]],
					["y",[float(random.uniform(-1.0,1.0)) for i in range(1)]],
					["type",[1]],
					["fx",[float(random.uniform(-1.0,1.0)) for i in range(1)]],
					["fy",[float(random.uniform(-1.0,1.0)) for i in range(1)]],
					["steer_x",[0.0]],
					["steer_y",[0.0]],
					["life",range(1,50,10)]]]
	
	global_data.sort(key=len)
	agent_data.sort(key=len)
	
	prefix_components = []
	prefix_components += [["LIST_TEST_VARIABLE",[x[1][0] for x in global_data if x[0]=="LIST_TEST_VARIABLE"][0] if len(global_data)>0 else "N/A"]]
	prefix_components += [["test_var", "seed"]]
	
	prefix_strings = [str(y) for x in prefix_components for y in x]
	prefix = "_".join(prefix_strings)
	
	parameter_count = 0
	testing_agent = []
	if len(global_data)>0:
		global_parameter_names = [x[0] for x in global_data]
		global_combinations = list(itertools.product(*[x[1] for x in global_data]))
		testing = list(zip([global_parameter_names for x in range(len(global_combinations))],global_combinations))
	if len(agent_data)>0:
		print(agent_data)
		print("ad0",agent_data[0][0],"ad1",agent_data[0][1])
		print([[y[0] for y in x[1:]] for x in agent_data])
		agent_parameter_names = [[x[0],[y[0] for y in x[1:]]] for x in agent_data]
		print("apn",agent_parameter_names)
		print([y[1] for x in agent_data for y in x[1:]])
		agent_combinations = list(itertools.product(*[y[1] for x in agent_data for y in x[1:]]))
		print("combinations",agent_combinations)
		testing_agent = list(zip([[x for i in range(len(agent_parameter_names)) for x in agent_parameter_names[i][1]] for x in range(len(agent_combinations))],agent_combinations))
		print("testing",testing_agent)
	for var_names,var_values in testing_agent:
		print("potential combination",var_names,var_values)
		
		current_global_data = [x if len(x[1])==0 else [x[0],var_values[var_names.index(x[0])]] for x in global_data]
		print("current global",current_global_data)
		print("param values",[[y if len(y[1])==1 else [y[0],var_values[var_names.index(y[0])]]] for x in agent_data for y in x[1:]])
		current_agent_data = [[x[0],[y if len(y[1])==1 else [y[0],var_values[var_names.index(y[0])]] for x in agent_data for y in x[1:]]] for x in agent_data]
		print("current agent data",current_agent_data)
		
		#initial_state(str(prefix),"0",current_global_data,current_agent_data)
		print("prefix components",prefix_components)

		prefix_components = [x if (not x[0] in var_names) else [x[0],var_values[var_names.index(x[0])]] for x in prefix_components]
		prefix_components = [x if not x[0]=="test_var" else [x[0],x[1]+"seed"] for x in prefix_components]
 
		prefix_strings = [str(y) for x in prefix_components for y in x]
		prefix = "_".join(prefix_strings)
		print(prefix)
		
	return global_data,agent_data

generate_initial_states_example_generator()
#Agent data stored in list of lists
base_agent_information = [
["prey",["initial_population",0,5000],["x",-1.0,1.0],["y",-1.0,1.0],["type",1,1],["fx",-1.0,1.0],["fy",-1.0,1.0],["steer_x",0.0,0.0],["steer_y",0.0,0.0],["life",1,50],],
["predator",["initial_population",0,5000],["x",-1.0,1.0],["y",-1.0,1.0],["type",1,1],["fx",-1.0,1.0],["fy",-1.0,1.0],["steer_x",0.0,0.0],["steer_y",0.0,0.0],["life",1,50],],]

#Create initial state
#initial_state_creation_example_generator("",base_agent_information)

#Generate initial states based on defined ranges/lists/values for all global and agent population variables for experiment example_generator_1param_list.
def generate_initial_states_example_generator_1param_list():
	global_data = []
	agent_data = []
	global_data += [["LIST_TEST_VARIABLE", [1,2,3,4,5]]]
	
	global_data.sort(key=len)
	agent_data.sort(key=len)
	
	prefix_components = []
	prefix_components += [["LIST_TEST_VARIABLE",[x[1][0] for x in global_data if x[0]=="LIST_TEST_VARIABLE"][0] if len(global_data)>0 else "N/A"]]
	prefix_components += [["test_var", 0]]
	
	prefix_strings = [str(y) for x in prefix_components for y in x]
	prefix = "_".join(prefix_strings)
	
	parameter_count = 0
	testing_agent = []
	if len(global_data)>0:
		global_parameter_names = [x[0] for x in global_data]
		global_combinations = list(itertools.product(*[x[1] for x in global_data]))
		testing = list(zip([global_parameter_names for x in range(len(global_combinations))],global_combinations))
	if len(agent_data)>0:
		print(agent_data)
		print("ad0",agent_data[0][0],"ad1",agent_data[0][1])
		print([[y[0] for y in x[1:]] for x in agent_data])
		agent_parameter_names = [[x[0],[y[0] for y in x[1:]]] for x in agent_data]
		print("apn",agent_parameter_names)
		print([y[1] for x in agent_data for y in x[1:]])
		agent_combinations = list(itertools.product(*[y[1] for x in agent_data for y in x[1:]]))
		print("combinations",agent_combinations)
		testing_agent = list(zip([[x for i in range(len(agent_parameter_names)) for x in agent_parameter_names[i][1]] for x in range(len(agent_combinations))],agent_combinations))
		print("testing",testing_agent)
	for var_names,var_values in testing_agent:
		print("potential combination",var_names,var_values)
		
		current_global_data = [x if len(x[1])==0 else [x[0],var_values[var_names.index(x[0])]] for x in global_data]
		print("current global",current_global_data)
		print("param values",[[y if len(y[1])==1 else [y[0],var_values[var_names.index(y[0])]]] for x in agent_data for y in x[1:]])
		current_agent_data = [[x[0],[y if len(y[1])==1 else [y[0],var_values[var_names.index(y[0])]] for x in agent_data for y in x[1:]]] for x in agent_data]
		print("current agent data",current_agent_data)
		
		#initial_state(str(prefix),"0",current_global_data,current_agent_data)
		print("prefix components",prefix_components)

		prefix_components = [x if (not x[0] in var_names) else [x[0],var_values[var_names.index(x[0])]] for x in prefix_components]
		prefix_components = [x if not x[0]=="test_var" else [x[0],x[1]+int(1)] for x in prefix_components]
 
		prefix_strings = [str(y) for x in prefix_components for y in x]
		prefix = "_".join(prefix_strings)
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
	
	global_data.sort(key=len)
	agent_data.sort(key=len)
	
	prefix_components = []
	prefix_components += [["test_var", 0]]
	
	prefix_strings = [str(y) for x in prefix_components for y in x]
	prefix = "_".join(prefix_strings)
	
	parameter_count = 0
	testing_agent = []
	if len(global_data)>0:
		global_parameter_names = [x[0] for x in global_data]
		global_combinations = list(itertools.product(*[x[1] for x in global_data]))
		testing = list(zip([global_parameter_names for x in range(len(global_combinations))],global_combinations))
	if len(agent_data)>0:
		print(agent_data)
		print("ad0",agent_data[0][0],"ad1",agent_data[0][1])
		print([[y[0] for y in x[1:]] for x in agent_data])
		agent_parameter_names = [[x[0],[y[0] for y in x[1:]]] for x in agent_data]
		print("apn",agent_parameter_names)
		print([y[1] for x in agent_data for y in x[1:]])
		agent_combinations = list(itertools.product(*[y[1] for x in agent_data for y in x[1:]]))
		print("combinations",agent_combinations)
		testing_agent = list(zip([[x for i in range(len(agent_parameter_names)) for x in agent_parameter_names[i][1]] for x in range(len(agent_combinations))],agent_combinations))
		print("testing",testing_agent)
	for var_names,var_values in testing_agent:
		print("potential combination",var_names,var_values)
		
		current_global_data = [x if len(x[1])==0 else [x[0],var_values[var_names.index(x[0])]] for x in global_data]
		print("current global",current_global_data)
		print("param values",[[y if len(y[1])==1 else [y[0],var_values[var_names.index(y[0])]]] for x in agent_data for y in x[1:]])
		current_agent_data = [[x[0],[y if len(y[1])==1 else [y[0],var_values[var_names.index(y[0])]] for x in agent_data for y in x[1:]]] for x in agent_data]
		print("current agent data",current_agent_data)
		
		#initial_state(str(prefix),"0",current_global_data,current_agent_data)
		print("prefix components",prefix_components)

		prefix_components = [x if (not x[0] in var_names) else [x[0],var_values[var_names.index(x[0])]] for x in prefix_components]
		prefix_components = [x if not x[0]=="test_var" else [x[0],x[1]+int(1)] for x in prefix_components]
 
		prefix_strings = [str(y) for x in prefix_components for y in x]
		prefix = "_".join(prefix_strings)
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
	
	global_data.sort(key=len)
	agent_data.sort(key=len)
	
	prefix_components = []
	prefix_components += [["test_var", 0]]
	
	prefix_strings = [str(y) for x in prefix_components for y in x]
	prefix = "_".join(prefix_strings)
	
	parameter_count = 0
	testing_agent = []
	if len(global_data)>0:
		global_parameter_names = [x[0] for x in global_data]
		global_combinations = list(itertools.product(*[x[1] for x in global_data]))
		testing = list(zip([global_parameter_names for x in range(len(global_combinations))],global_combinations))
	if len(agent_data)>0:
		print(agent_data)
		print("ad0",agent_data[0][0],"ad1",agent_data[0][1])
		print([[y[0] for y in x[1:]] for x in agent_data])
		agent_parameter_names = [[x[0],[y[0] for y in x[1:]]] for x in agent_data]
		print("apn",agent_parameter_names)
		print([y[1] for x in agent_data for y in x[1:]])
		agent_combinations = list(itertools.product(*[y[1] for x in agent_data for y in x[1:]]))
		print("combinations",agent_combinations)
		testing_agent = list(zip([[x for i in range(len(agent_parameter_names)) for x in agent_parameter_names[i][1]] for x in range(len(agent_combinations))],agent_combinations))
		print("testing",testing_agent)
	for var_names,var_values in testing_agent:
		print("potential combination",var_names,var_values)
		
		current_global_data = [x if len(x[1])==0 else [x[0],var_values[var_names.index(x[0])]] for x in global_data]
		print("current global",current_global_data)
		print("param values",[[y if len(y[1])==1 else [y[0],var_values[var_names.index(y[0])]]] for x in agent_data for y in x[1:]])
		current_agent_data = [[x[0],[y if len(y[1])==1 else [y[0],var_values[var_names.index(y[0])]] for x in agent_data for y in x[1:]]] for x in agent_data]
		print("current agent data",current_agent_data)
		
		#initial_state(str(prefix),"0",current_global_data,current_agent_data)
		print("prefix components",prefix_components)

		prefix_components = [x if (not x[0] in var_names) else [x[0],var_values[var_names.index(x[0])]] for x in prefix_components]
		prefix_components = [x if not x[0]=="test_var" else [x[0],x[1]+int(1)] for x in prefix_components]
 
		prefix_strings = [str(y) for x in prefix_components for y in x]
		prefix = "_".join(prefix_strings)
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
	
	global_data.sort(key=len)
	agent_data.sort(key=len)
	
	prefix_components = []
	prefix_components += [["test_var", 0]]
	
	prefix_strings = [str(y) for x in prefix_components for y in x]
	prefix = "_".join(prefix_strings)
	
	parameter_count = 0
	testing_agent = []
	if len(global_data)>0:
		global_parameter_names = [x[0] for x in global_data]
		global_combinations = list(itertools.product(*[x[1] for x in global_data]))
		testing = list(zip([global_parameter_names for x in range(len(global_combinations))],global_combinations))
	if len(agent_data)>0:
		print(agent_data)
		print("ad0",agent_data[0][0],"ad1",agent_data[0][1])
		print([[y[0] for y in x[1:]] for x in agent_data])
		agent_parameter_names = [[x[0],[y[0] for y in x[1:]]] for x in agent_data]
		print("apn",agent_parameter_names)
		print([y[1] for x in agent_data for y in x[1:]])
		agent_combinations = list(itertools.product(*[y[1] for x in agent_data for y in x[1:]]))
		print("combinations",agent_combinations)
		testing_agent = list(zip([[x for i in range(len(agent_parameter_names)) for x in agent_parameter_names[i][1]] for x in range(len(agent_combinations))],agent_combinations))
		print("testing",testing_agent)
	for var_names,var_values in testing_agent:
		print("potential combination",var_names,var_values)
		
		current_global_data = [x if len(x[1])==0 else [x[0],var_values[var_names.index(x[0])]] for x in global_data]
		print("current global",current_global_data)
		print("param values",[[y if len(y[1])==1 else [y[0],var_values[var_names.index(y[0])]]] for x in agent_data for y in x[1:]])
		current_agent_data = [[x[0],[y if len(y[1])==1 else [y[0],var_values[var_names.index(y[0])]] for x in agent_data for y in x[1:]]] for x in agent_data]
		print("current agent data",current_agent_data)
		
		#initial_state(str(prefix),"0",current_global_data,current_agent_data)
		print("prefix components",prefix_components)

		prefix_components = [x if (not x[0] in var_names) else [x[0],var_values[var_names.index(x[0])]] for x in prefix_components]
		prefix_components = [x if not x[0]=="test_var" else [x[0],x[1]+int(1)] for x in prefix_components]
 
		prefix_strings = [str(y) for x in prefix_components for y in x]
		prefix = "_".join(prefix_strings)
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
	global_data += [["THIRD_LIST_TEST_VARIABLE", [51,62,73,84]]]
	
	global_data.sort(key=len)
	agent_data.sort(key=len)
	
	prefix_components = []
	prefix_components += [["LIST_TEST_VARIABLE",[x[1][0] for x in global_data if x[0]=="LIST_TEST_VARIABLE"][0] if len(global_data)>0 else "N/A"]]
	prefix_components += [["test_var", 0]]
	
	prefix_strings = [str(y) for x in prefix_components for y in x]
	prefix = "_".join(prefix_strings)
	
	parameter_count = 0
	testing_agent = []
	if len(global_data)>0:
		global_parameter_names = [x[0] for x in global_data]
		global_combinations = list(itertools.product(*[x[1] for x in global_data]))
		testing = list(zip([global_parameter_names for x in range(len(global_combinations))],global_combinations))
	if len(agent_data)>0:
		print(agent_data)
		print("ad0",agent_data[0][0],"ad1",agent_data[0][1])
		print([[y[0] for y in x[1:]] for x in agent_data])
		agent_parameter_names = [[x[0],[y[0] for y in x[1:]]] for x in agent_data]
		print("apn",agent_parameter_names)
		print([y[1] for x in agent_data for y in x[1:]])
		agent_combinations = list(itertools.product(*[y[1] for x in agent_data for y in x[1:]]))
		print("combinations",agent_combinations)
		testing_agent = list(zip([[x for i in range(len(agent_parameter_names)) for x in agent_parameter_names[i][1]] for x in range(len(agent_combinations))],agent_combinations))
		print("testing",testing_agent)
	for var_names,var_values in testing_agent:
		print("potential combination",var_names,var_values)
		
		current_global_data = [x if len(x[1])==0 else [x[0],var_values[var_names.index(x[0])]] for x in global_data]
		print("current global",current_global_data)
		print("param values",[[y if len(y[1])==1 else [y[0],var_values[var_names.index(y[0])]]] for x in agent_data for y in x[1:]])
		current_agent_data = [[x[0],[y if len(y[1])==1 else [y[0],var_values[var_names.index(y[0])]] for x in agent_data for y in x[1:]]] for x in agent_data]
		print("current agent data",current_agent_data)
		
		#initial_state(str(prefix),"0",current_global_data,current_agent_data)
		print("prefix components",prefix_components)

		prefix_components = [x if (not x[0] in var_names) else [x[0],var_values[var_names.index(x[0])]] for x in prefix_components]
		prefix_components = [x if not x[0]=="test_var" else [x[0],x[1]+int(1)] for x in prefix_components]
 
		prefix_strings = [str(y) for x in prefix_components for y in x]
		prefix = "_".join(prefix_strings)
		print(prefix)
		
	return global_data,agent_data

generate_initial_states_example_generator_2param_list()
#Agent data stored in list of lists
base_agent_information = []

#Create initial state
#initial_state_creation_example_generator_2param_list("",base_agent_information)

#ExperimentSet

##############  ############
