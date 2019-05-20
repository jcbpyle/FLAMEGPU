# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 14:16:06 2019

@author: James
"""
import os
import random
import pycuda.driver as cuda
import pycuda.autoinit

BASE_DIRECTORY = os.getcwd()
DEVICES = cuda.Device(0).count()


def initial_state_creation(name,agent_information):
    SAVE_DIRECTORY = BASE_DIRECTORY+"/"
    initial_state_file = open(SAVE_DIRECTORY+str(name),"w")
    initial_state_file.write("<states>\n<itno>0</itno>\n<environment>\n")
    #GLOBALS
    initial_state_file.write("<EXAMPLE_GLOBAL>"+str(0.12345)+"</EXAMPLE_GLOBAL>\n")
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
            #AGENT VARIABLES
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
#agents = [["grass",["initial_population",0,5],["x",-1.0,1.0],["y",-1.0,1.0],["available",1,1]],["prey",["initial_population",0,5],["x",-1.0,1.0],["y",-1.0,1.0],["life",5,50]],["predator",["initial_population",0,5],["x",-1.0,1.0],["y",-1.0,1.0],["life",5,50]]]
#initial_state_creation("testing.xml",agents)

def create_experiment_state(x):
    for i in range(x):
        new_agent_info = [["grass",["initial_population",1,int(random.uniform(1,10))],["x",-1.0,1.0],["y",-1.0,1.0],["available",1,1]],
                          ["prey",["initial_population",1,int(random.uniform(1,10))],["x",-1.0,1.0],["y",-1.0,1.0],["life",int(random.uniform(1,10)),int(random.uniform(50,75))]],
                          ["predator",["initial_population",1,int(random.uniform(1,10))],["x",-1.0,1.0],["y",-1.0,1.0],["life",int(random.uniform(1,10)),int(random.uniform(50,75))]],
                          ["test_agent",["initial_population",3,3],["fixed_x",0.75,0.75],["fixed_y",-1.0,-1.0],["available",1,1]]]
        new_file_name = "testing"+str(i)+".xml"
        print(i,"grass",new_agent_info[0][1][2],"prey",new_agent_info[1][1][2],"predator",new_agent_info[2][1][2])
        initial_state_creation(new_file_name,new_agent_info)
        
create_experiment_state(5)