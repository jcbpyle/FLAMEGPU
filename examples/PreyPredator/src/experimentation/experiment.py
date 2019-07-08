
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
	
	global_data = {"REPRODUCE_PREY_PROB":[float(random.uniform(0,0.25)) for i in range(1)],"REPRODUCE_PREDATOR_PROB":[float(random.uniform(0,0.25)) for i in range(1)],"GAIN_FROM_FOOD_PREY":[int(random.uniform(0,500)) for i in range(1)],"GAIN_FROM_FOOD_PREDATOR":[int(random.uniform(0,500)) for i in range(1)],"GRASS_REGROW_CYCLES":[int(random.uniform(0,500)) for i in range(1)]}
	prey = {"initial_population":[int(random.uniform(1,1000)) for i in range(1)], "type":[1],"steer_x":[0.0],"steer_y":[0.0],}
	prey_vary_per_agent = {"x":[-1.0,1.0,"uniform",float],"y":[-1.0,1.0,"uniform",float],"fx":[-1.0,1.0,"uniform",float],"fy":[-1.0,1.0,"uniform",float],"life":[50,500,"uniform",int],}
	predator = {"initial_population":[0], "type":[1],"steer_x":[0.0],"steer_y":[0.0],}
	predator_vary_per_agent = {"x":[-1.0,1.0,"uniform",float],"y":[-1.0,1.0,"uniform",float],"fx":[-1.0,1.0,"uniform",float],"fy":[-1.0,1.0,"uniform",float],"life":[1,100,"uniform",int],}
	grass = {"initial_population":[int(random.uniform(1000,2000)) for i in range(1)], "type":[2],"dead_cycles":[0],"available":[1]}
	grass_vary_per_agent = {"x":[-1.0,1.0,"uniform",float],"y":[-1.0,1.0,"uniform",float],}
	
	agent_data = {"prey":prey,"predator":predator,"grass":grass}
	
	vary_per_agent = {"prey":prey_vary_per_agent,"predator":predator_vary_per_agent,"grass":grass_vary_per_agent}

	
	prefix_components = []
	
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
				initial_state("","0",g,current_agent_data)
	elif len(global_combinations)>0:
		for g in global_combinations:
			current_agent_data = [agent+[[x[0],x[1]] for x in vary_per_agent[agent[0]].items()] for agent in agent_data]
			initial_state("","0",g,current_agent_data)
	elif len(agent_combinations)>0:
		for a in agent_combinations:
			current_agent_data = [agent+[[x[0],x[1]] for x in vary_per_agent[agent[0]].items()] for agent in a]
			initial_state("","0",global_data,current_agent_data)
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
			agent_id = 1
			agent_name = agent_information[i][0]
			for j in range(num_agents):
				initial_state_file.write("<xagent>\n")
				initial_state_file.write("<name>"+str(agent_name)+"</name>\n")
				initial_state_file.write("<id>"+str(agent_id)+"</id>\n")
				for k in agent_information[i]:
					if not (k[0]=="initial_population" or k==agent_name):
						if len(k[1])>1:
							if len(k[1])==4:
								random_method = getattr(random, k[1][2])
								initial_state_file.write("<"+str(k[0])+">"+str(k[1][3](random_method(k[1][0],k[1][1])))+"</"+str(k[0])+">\n")
							elif len(k[1])==3:
								initial_state_file.write("<"+str(k[0])+">"+str(k[1][2](random.uniform(k[1][0],k[1][1])))+"</"+str(k[0])+">\n")
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

############## testing_batch_simulation ############

def run_batch():
	
	##Run for desired number of repeats
	#current_initial_state = PROJECT_DIRECTORY+"iterations"+"/0.xml"
	#for i in range(10):
		#generate_initial_states_predprey()
		#if OS_NAME=='nt':
			#executable = PROJECT_DIRECTORY+"../../bin/x64/Release_Console//PreyPredator_api_test.exe"
			#simulation_command = executable+" "+current_initial_state+" 1000"
		#else:
			#executable = "./"+PROJECT_DIRECTORY+"../../bin/x64/Release_Console//PreyPredator_api_test"
			#simulation_command = executable+" "+current_initial_state+" 1000"
		#print(simulation_command)
		##Run simulation
		#os.system(simulation_command)

		##Parse results
		#results_file = open(PROJECT_DIRECTORY+"/iterations/log.csv","r")
		#results = results_file.readlines()
		#results_file.close()
		#print(results)
	
	return 

def main():
	
	#Initial state creation function
	#initial_state(save_directory, initial_state_file_name, initial_state_global_data_list, initial_state_agent_data_list)

	#Generation functions (will automatically call initial state generation function)
	
	generate_initial_states_predprey()
	
	#Experiment Set user defined functions
	
	#run_batch()
	
	return

if __name__ == "__main__":
	main()
