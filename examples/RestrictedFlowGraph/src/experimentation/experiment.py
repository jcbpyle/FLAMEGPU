
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
import random
import itertools
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

#Generate initial states based on defined ranges/lists/values for all global and agent population variables for experiment rfg_test.
def generate_initial_states_rfg_test():
	global_data = []
	agent_data = []
	
	global_data = {"RAND_SEED":range(0,1499),"INIT_POPULATION":range(1,5)}
	Agent = {"initial_population":[10], "currentEdge":[0],"nextEdge":[0],"nextEdgeRemainingCapacity":[0],"hasIntent":[0],"position":[0],"distanceTravelled":[0],"blockedIterationCount":[0],"speed":[0],"x":[0],"y":[0],"z":[0],"colour":[1.0]}
	Agent_vary_per_agent = {}
	
	agent_data = {"Agent":Agent}
	
	vary_per_agent = {"Agent":Agent_vary_per_agent}

	
	prefix_components = []
	prefix_components += [["RAND_SEED",global_data["RAND_SEED"][0] if len(global_data)>0 else "NA"]]
	prefix_components += [["INIT_POPULATION",global_data["INIT_POPULATION"][0] if len(global_data)>0 else "NA"]]
	
	prefix_strings = [str(y) for x in prefix_components for y in x]
	prefix = "_".join(prefix_strings)
	
	global_combinations = []
	agent_combinations = []
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

generate_initial_states_rfg_test()
#Initial state creation function
#initial_state(save_directory, initial_state_file_name, initial_state_global_data_list, initial_state_agent_data_list)

#ExperimentSet

############## testing_rfg_different_seeds ############
test_var = int(100)

def process_output(test_var):
	some_variable = None

	#Model executable
	executable = ""
	simulation_command = ""
	initial_states = [x[0] for x in os.walk("../../iterations")][1:]
	current_initial_state_loc = initial_states[0]
	current_initial_state = current_initial_state_loc+"/0.xml"
	#print("current",current_initial_state)
	if os.name=='nt':
		executable = PROJECT_DIRECTORY+"../../bin/x64/Release_Console//RestrictedFlowGraph.exe"
		simulation_command = executable+" "+current_initial_state+" 100"
	else:
		executable = "./"+PROJECT_DIRECTORY+"../../bin/x64/Release_Console//RestrictedFlowGraph"
		simulation_command = executable+" "+current_initial_state+" 100"

	
	#print(simulation_command)
	#Run simulation
	os.system(simulation_command)
	#Parse results
	results_file = open(PROJECT_DIRECTORY+"/iterations"+current_initial_state_loc+"/100.xml","r")
	results = results_file.readlines()
	results_file.close()
	#print("results",results)
		
	return some_variable
process_output(test_var)

def save_output(test_var):
	global global_variable
	
	#Model executable
	executable = ""
	simulation_command = ""
	initial_states = [x[0] for x in os.walk("../../iterations")][1:]
	current_initial_state_loc = initial_states[0]
	current_initial_state = current_initial_state_loc+"/0.xml"
	#print("current",current_initial_state)
	if os.name=='nt':
		executable = PROJECT_DIRECTORY+"../../bin/x64/Release_Console//RestrictedFlowGraph.exe"
		simulation_command = executable+" "+current_initial_state+" 100"
	else:
		executable = "./"+PROJECT_DIRECTORY+"../../bin/x64/Release_Console//RestrictedFlowGraph"
		simulation_command = executable+" "+current_initial_state+" 100"

	
	#print(simulation_command)
	#Run simulation
	os.system(simulation_command)
	#Parse results
	results_file = open(PROJECT_DIRECTORY+"/iterations"+current_initial_state_loc+"/100.xml","r")
	results = results_file.readlines()
	results_file.close()
	#print("results",results)
		
	return 
save_output(test_var)
