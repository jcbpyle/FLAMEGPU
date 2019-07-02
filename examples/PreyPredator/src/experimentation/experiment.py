
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
import itertools
import pycuda.driver as cuda
import pycuda.autoinit

BASE_DIRECTORY = os.getcwd()+"/"
PROJECT_DIRECTORY = BASE_DIRECTORY+"../../"
GPUS_AVAILABLE = cuda.Device(0).count()
OS_NAME = os.name

#InitialStates

initial_state_files = []

initial_state_files += ["/iterations/0.xml"]

#Generate initial states based on defined ranges/lists/values for all global and agent population variables for experiment predprey.
def generate_initial_states_predprey():
	global_data = []
	agent_data = []
	vary_per_agent = []
	
	global_data = {"INTERACTION_DISTANCE_TEST_VARIABLE":[0.12345],"LIST_TEST_VARIABLE":[1,2,3,4,5],"LIST_TEST_VARIABLE_2":[1,2,3,4,5],"LIST_TEST_VARIABLE_3":random.choices([1,2,3,4,5,10],k=2),"DISTRIBUTION_TEST_VARIABLE":[random.uniform(1,10)],"DISTRIBUTION_TEST_VARIABLE_2":random.sample([1,10,5],k=2),"DISTRIBUTION_TEST_VARIABLE_3":[random.randrange(1,10,2)],"REPRODUCE_PREY_PROB":[float(random.uniform(0,0.25)) for i in range(1)],"REPRODUCE_PREDATOR_PROB":[float(random.uniform(0,0.25)) for i in range(1)],"GAIN_FROM_FOOD_PREY":[int(random.uniform(0,500)) for i in range(1)],"GAIN_FROM_FOOD_PREDATOR":[int(random.uniform(0,500)) for i in range(1)],"GRASS_REGROW_CYCLES":[int(random.uniform(0,500)) for i in range(1)]}
	prey = {"initial_population":[int(random.uniform(0,5000)) for i in range(1)], "type":[1],"steer_x":[0.0],"steer_y":[0.0],"life":[int(random.uniform(1,50)) for i in range(1)]}
	prey_vary_per_agent = {"x":[-1.0,1.0,"uniform"],"y":[-1.0,1.0,"uniform"],"fx":[-1.0,1.0,"uniform"],"fy":[-1.0,1.0,"uniform"],}
	predator = {"initial_population":[int(random.uniform(0,5000)) for i in range(1)], "type":[1],"steer_x":[0.0],"steer_y":[0.0],}
	predator_vary_per_agent = {"x":[-1.0,1.0,"uniform"],"y":[-1.0,1.0,"uniform"],"fx":[-1.0,1.0,"uniform"],"fy":[-1.0,1.0,"uniform"],"life":[1,100,"uniform"],}
	grass = {"initial_population":[int(random.uniform(0,5000)) for i in range(1)], "type":[2],"dead_cycles":[0],"available":[1]}
	grass_vary_per_agent = {"x":[-1.0,1.0,"uniform"],"y":[-1.0,1.0,"uniform"],}
	
	agent_data = {"prey":prey,"predator":predator,"grass":grass}
	
	vary_per_agent = {"prey":prey_vary_per_agent,"predator":predator_vary_per_agent,"grass":grass_vary_per_agent}

	
	prefix_components = []
	prefix_components += [["LIST_TEST_VARIABLE",global_data["LIST_TEST_VARIABLE"][0] if len(global_data)>0 else "NA"]]
	prefix_components += [["LIST_TEST_VARIABLE_2",global_data["LIST_TEST_VARIABLE_2"][0] if len(global_data)>0 else "NA"]]
	prefix_components += [["LIST_TEST_VARIABLE_3",global_data["LIST_TEST_VARIABLE_3"][0] if len(global_data)>0 else "NA"]]
	prefix_components += [["DISTRIBUTION_TEST_VARIABLE_2",global_data["DISTRIBUTION_TEST_VARIABLE_2"][0] if len(global_data)>0 else "NA"]]
	
	prefix_strings = [str(y) for x in prefix_components for y in x]
	prefix = "_".join(prefix_strings)
	
	if len(global_data)>0:
		global_names = [x for x in global_data]
		unnamed_global_combinations = list(itertools.product(*[y for x,y in global_data.items()]))
		global_combinations = list(zip([global_names for x in range(len(unnamed_global_combinations))],unnamed_global_combinations))
	if len(agent_data)>0:
		agent_names = [x for x in agent_data]
		unnamed_agent_combinations = list(itertools.product(*[z for x,y in agent_data.items() for w,z in y.items()]))
		loc = 0
		agent_combinations = [[] for x in range(len(unnamed_agent_combinations))]
		for an in agent_names:
			num_vars = loc+len(agent_data[an])
			var_names = [x for x in agent_data[an]]
			sublists = [x[loc:num_vars] for x in unnamed_agent_combinations]
			named_combinations = list(zip([var_names for x in range(len(sublists))],sublists))
			for i in range(len(named_combinations)):
				temp_list = [an]
				temp_list += [[named_combinations[i][0][x],[named_combinations[i][1][x]]] for x in range(len(named_combinations[i][0]))]
				agent_combinations[i] += [temp_list]
			loc = num_vars
	if len(global_combinations)>0 and len(agent_combinations)>0:
		for g in global_combinations:
			for a in agent_combinations:
				current_agent_data = [agent+[[x[0],x[1]] for x in vary_per_agent[agent[0]].items()] for agent in a]
				
				initial_state(str(prefix),"0",g,current_agent_data)
				prefix_components = [x if not x[0] in g[0] else [x[0],g[1][g[0].index(x[0])]] for x in prefix_components]
				 
				prefix_strings = [str(y) for x in prefix_components for y in x]
				prefix = "_".join(prefix_strings)
				
	elif len(global_combinations)>0:
		for g in global_combinations:
			current_agent_data = [agent+[[x[0],x[1]] for x in vary_per_agent[agent[0]].items()] for agent in agent_data]
			
			initial_state(str(prefix),"0",g,current_agent_data)
			prefix_components = [x if not x[0] in g[0] else [x[0],g[1][g[0].index(x[0])]] for x in prefix_components]
			 
			prefix_strings = [str(y) for x in prefix_components for y in x]
			prefix = "_".join(prefix_strings)
			
	elif len(agent_combinations)>0:
		for a in agent_combinations:
			current_agent_data = [agent+[[x[0],x[1]] for x in vary_per_agent[agent[0]].items()] for agent in a]
			
			initial_state(str(prefix),"0",global_data,current_agent_data)
			prefix_components = [x if not x[0] in a else [x[0],a.index(x[0])[1]] for x in prefix_components]
			 
			prefix_strings = [str(y) for x in prefix_components for y in x]
			prefix = "_".join(prefix_strings)
			
	else:
		print("No global or agent variations specified for experimentation\n")
	return global_data,agent_data

#Initial state file creation.
def initial_state(save_location,file_name,global_information,agent_information):
	if not os.path.exists(PROJECT_DIRECTORY+"iterations"):
		os.mkdir(PROJECT_DIRECTORY+"iterations")
	SAVE_DIRECTORY = PROJECT_DIRECTORY+"iterations"+"/"
	if not os.path.exists(SAVE_DIRECTORY+str(save_location)):
		os.mkdir(SAVE_DIRECTORY+str(save_location))
	print("Creating initial state in",SAVE_DIRECTORY,save_location,"/",file_name,"\n")
	initial_state_file = open(SAVE_DIRECTORY+str(save_location)+"/"+str(file_name)+".xml","w")
	initial_state_file.write("<states>\n<itno>0</itno>\n<environment>\n")
	if len(global_information)>0:
		for g in range(len(global_information[0])):
			initial_state_file.write("<"+str(global_information[0][g])+">"+str(global_information[1][g])+"</"+str(global_information[0][g])+">\n")
	initial_state_file.write("</environment>\n")
	if len(agent_information)>0:
		for i in range(len(agent_information)):
			try:
				ind = [x[0] for x in agent_information[i]].index("initial_population")
			except:
				ind = 0
			num_agents = int(agent_information[i][ind][1][0])
			agent_id = 0
			agent_name = agent_information[i][0]
			for j in range(num_agents):
				initial_state_file.write("<xagent>\n")
				initial_state_file.write("<name>"+str(agent_name)+"</name>\n")
				initial_state_file.write("<id>"+str(agent_id)+"</id>\n")
				for k in agent_information[i]:
					if not (k[0]=="initial_population" or k==agent_name):
						if len(k[1])>1:
							if len(k[1])==3:
								random_method = getattr(random, k[1][2])
								initial_state_file.write("<"+str(k[0])+">"+str(random_method(k[1][0],k[1][1]))+"</"+str(k[0])+">\n")
							else:
								initial_state_file.write("<"+str(k[0])+">"+str(random.uniform(k[1][0],k[1][1]))+"</"+str(k[0])+">\n")
						elif type(k[1][0])==type(int()):
							initial_state_file.write("<"+str(k[0])+">"+str(int(k[1][0]))+"</"+str(k[0])+">\n")
						elif type(k[1][0])==type(float()):
							initial_state_file.write("<"+str(k[0])+">"+str(float(k[1][0]))+"</"+str(k[0])+">\n")
						
				initial_state_file.write("</xagent>\n")
				agent_id += 1
	initial_state_file.write("</states>")
	return

#ExperimentSet

############## testing_initial_State ############

############## testing_batch_simulation ############

#Run for desired number of repeats
#for i in range(10):
	#initial_state_creation_(file_name,base_agent_information)
	#Run simulation
	#os.system(simulation_command)
	#Parse results
	#results_file = open("../../"+INSERT_FILE_DIRECTORY_AND_NAME_HERE,"r")
	#results = results_file.readlines()
	#results_file.close()

############## testing_ga_experiment ############

def fitness_function(primary,secondary,tertiary):
	fitness = None
	
	##Model executable
	#executable = ""
	#simulation_command = ""
	#initial_states = [x[0] for x in os.walk("../../")][1:]
	#for i in initial_states:
		#current_initial_state = i+"/0.xml"
		#if OS_NAME=='nt':
			#executable = PROJECT_DIRECTORY+"../../../bin/x64/Release_Console//PreyPredator_api_test.exe"
			#simulation_command = executable+" "+current_initial_state+" 1000"
		#else:
			#executable = "./"+PROJECT_DIRECTORY+"../../../bin/x64/Release_Console//PreyPredator_api_test"
			#simulation_command = executable+" "+current_initial_state+" 1000"

		
		##Run simulation
		#os.system(simulation_command)

		##Parse results
		#results_file = open(PROJECT_DIRECTORY+"/"+current_initial_state_loc+"/","r")
		#results = results_file.readlines()
		#results_file.close()
		
	return fitness

def setup_ga(mu):
	start_time = None
	population = None
	
	##Model executable
	#executable = ""
	#simulation_command = ""
	#initial_states = [x[0] for x in os.walk("../../")][1:]
	#for i in initial_states:
		#current_initial_state = i+"/0.xml"
		#if OS_NAME=='nt':
			#executable = PROJECT_DIRECTORY+"../../../bin/x64/Release_Console//PreyPredator_api_test.exe"
			#simulation_command = executable+" "+current_initial_state+" 1000"
		#else:
			#executable = "./"+PROJECT_DIRECTORY+"../../../bin/x64/Release_Console//PreyPredator_api_test"
			#simulation_command = executable+" "+current_initial_state+" 1000"

		
		##Run simulation
		#os.system(simulation_command)

		##Parse results
		#results_file = open(PROJECT_DIRECTORY+"/"+current_initial_state_loc+"/","r")
		#results = results_file.readlines()
		#results_file.close()
		
	return start_time, population

def run_ga(mu,LAMBDA,max_generations,max_time,start_time,crossover,mutation):
	global curr_pop
	Population = None
	
	##Model executable
	#executable = ""
	#simulation_command = ""
	#initial_states = [x[0] for x in os.walk("../../")][1:]
	#for i in initial_states:
		#current_initial_state = i+"/0.xml"
		#if OS_NAME=='nt':
			#executable = PROJECT_DIRECTORY+"../../../bin/x64/Release_Console//PreyPredator_api_test.exe"
			#simulation_command = executable+" "+current_initial_state+" 1000"
		#else:
			#executable = "./"+PROJECT_DIRECTORY+"../../../bin/x64/Release_Console//PreyPredator_api_test"
			#simulation_command = executable+" "+current_initial_state+" 1000"

		
		##Run simulation
		#os.system(simulation_command)

		##Parse results
		#results_file = open(PROJECT_DIRECTORY+"/"+current_initial_state_loc+"/","r")
		#results = results_file.readlines()
		#results_file.close()
		
	return Population

############## testing_surrogate_experiment ############

def train_surrogate(hidden_layers,error,max_time,max_training_generations):
	surrogate_accuracy = None
	
	##Model executable
	#executable = ""
	#simulation_command = ""
	#initial_states = [x[0] for x in os.walk("../../")][1:]
	#for i in initial_states:
		#current_initial_state = i+"/0.xml"
		#if OS_NAME=='nt':
			#executable = PROJECT_DIRECTORY+"../../../bin/x64/Release_Console//PreyPredator_api_test.exe"
			#simulation_command = executable+" "+current_initial_state+" 1000"
		#else:
			#executable = "./"+PROJECT_DIRECTORY+"../../../bin/x64/Release_Console//PreyPredator_api_test"
			#simulation_command = executable+" "+current_initial_state+" 1000"

		
		##Run simulation
		#os.system(simulation_command)

		##Parse results
		#results_file = open(PROJECT_DIRECTORY+"/"+current_initial_state_loc+"/","r")
		#results = results_file.readlines()
		#results_file.close()
		
	return surrogate_accuracy

def main():
	
	#Initial state creation function
	#initial_state(save_directory, initial_state_file_name, initial_state_global_data_list, initial_state_agent_data_list)

	#Generation functions (will automatically call initial state generation function)
	
	generate_initial_states_predprey()
	
	#Experiment Set user defined functions
	mu = int(100)
	LAMBDA = int(10)
	max_time = int(60)
	max_generations = int(10)
	mutation = float(0.25)
	crossover = float(0.5)
	
	#fitness = fitness_function(primary,secondary,tertiary)
	
	#start_time, population = setup_ga(mu)
	
	#Population = run_ga(mu,LAMBDA,max_generations,max_time,start_time,crossover,mutation)
	hidden_layers = (100,100)
	error = float(1e-9)
	max_time = int(60)
	max_training_generations = int(2000)
	
	#surrogate_accuracy = train_surrogate(hidden_layers,error,max_time,max_training_generations)
	
	return

if __name__ == "__main__":
	main()
