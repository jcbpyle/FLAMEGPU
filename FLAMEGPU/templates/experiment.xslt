<?xml version="1.0" encoding="utf-8"?>
<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
                xmlns:xmml="http://www.dcs.shef.ac.uk/~paul/XMML"
                xmlns:gpu="http://www.dcs.shef.ac.uk/~paul/XMMLGPU"
                xmlns:exp="https://jcbpyle.github.io/XMMLExperiment">
<xsl:output method="text" version="1.0" encoding="UTF-8" indent="yes" />
<!--Main template-->
<xsl:template match="/">
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

<xsl:if test="exp:Experimentation/exp:Imports"><xsl:for-each select="exp:Experimentation/exp:Imports/exp:Import"><xsl:if test="exp:From">from <xsl:value-of select="exp:From" />&#160;</xsl:if>import <xsl:value-of select="exp:Module" /><xsl:text>&#xa;</xsl:text></xsl:for-each></xsl:if>
import os
import random
import itertools
import pycuda.driver as cuda
import pycuda.autoinit

BASE_DIRECTORY = os.getcwd()+"/"
PROJECT_DIRECTORY = BASE_DIRECTORY+"../../"
GPUS_AVAILABLE = cuda.Device(0).count()
<xsl:if test="exp:Experimentation/exp:InitialStates">
#InitialStates

<xsl:if test="exp:Experimentation/exp:InitialStates/exp:InitialStateFile">
initial_state_files = []
<xsl:for-each select="exp:Experimentation/exp:InitialStates/exp:InitialStateFile">
<xsl:if test="exp:FileName">
initial_state_files += ["/<xsl:value-of select="exp:Location"/>/<xsl:value-of select="exp:FileName"/>.xml"]
</xsl:if>
</xsl:for-each>
</xsl:if>

<xsl:if test="exp:Experimentation/exp:InitialStates/exp:InitialStateFunction">
<xsl:for-each select="exp:Experimentation/exp:InitialStates/exp:InitialStateFunction">
#Initial state generator function to be created by the user
def initial_state_generator_function_<xsl:value-of select="exp:FunctionName"/>():

	return<xsl:text>&#xa;</xsl:text>
</xsl:for-each>
</xsl:if>

<xsl:if test="exp:Experimentation/exp:InitialStates/@baseDirectory">
#Initial state file creation.
def initial_state(save_location,file_name,global_information,agent_information):
	SAVE_DIRECTORY = PROJECT_DIRECTORY+"<xsl:value-of select="exp:Experimentation/exp:InitialStates/@baseDirectory"/>"+"/"
	if not os.path.exists(SAVE_DIRECTORY+str(save_location)):
		os.mkdir(SAVE_DIRECTORY+str(save_location))
	print(SAVE_DIRECTORY,save_location,file_name)
	initial_state_file = open(SAVE_DIRECTORY+str(save_location)+"/"+str(file_name)+".xml","w")
	initial_state_file.write("&lt;states&gt;\n&lt;itno&gt;0&lt;/itno&gt;\n&lt;environment&gt;\n")
	if len(global_information)>0:
		for g in range(len(global_information)):
			initial_state_file.write("&lt;"+str(global_information[g][0])+"&gt;"+str(gloabl_information[g][1])+"&lt;/"+str(global_information[g][0])+"&gt;\n")
	initial_state_file.write("&lt;/environment&gt;\n")
	if len(agent_information)>0:
		for i in range(len(agent_information)):
			ind = [x[0] for x in agent_information[i]].index("initial_population")
			num_agents = int(random.uniform(agent_information[i][ind][1],agent_information[i][ind][2]))
			agent_id = 0
			agent_name = agent_information[i][0]
			for j in range(num_agents):
				initial_state_file.write("&lt;xagent&gt;\n")
				initial_state_file.write("&lt;name&gt;"+str(agent_name)+"&lt;/name&gt;\n")
				initial_state_file.write("&lt;id&gt;"+str(agent_id)+"&lt;/id&gt;\n")
				for k in agent_information[i]:
					if not (k[0]=="initial_population" or k==agent_name):
						if type(k[1])==type(int()):
							initial_state_file.write("&lt;"+str(k[0])+"&gt;"+str(int(random.uniform(k[1],k[2])))+"&lt;/"+str(k[0])+"&gt;\n")
						elif type(k[1])==type(float()):
							initial_state_file.write("&lt;"+str(k[0])+"&gt;"+str(random.uniform(k[1],k[2]))+"&lt;/"+str(k[0])+"&gt;\n")
				initial_state_file.write("&lt;/xagent&gt;\n")
				agent_id += 1
	initial_state_file.write("&lt;/states&gt;")
	return
</xsl:if>
<xsl:if test="exp:Experimentation/exp:InitialStates/exp:InitialStateGenerator">
<xsl:for-each select="exp:Experimentation/exp:InitialStates/exp:InitialStateGenerator">
#Generate initial states based on defined ranges/lists/values for all global and agent population variables<xsl:if test="exp:GeneratorName"> for experiment <xsl:value-of select="exp:GeneratorName"/></xsl:if>.
def generate_initial_states<xsl:if test="exp:GeneratorName">_<xsl:value-of select="exp:GeneratorName"/></xsl:if>():
	global_data = []
	agent_data = []
	<xsl:if test="exp:Globals">
	<xsl:for-each select="exp:Globals/exp:Global">
	<xsl:if test="exp:Value">global_data += [["<xsl:value-of select="exp:Name"/>", <xsl:choose><xsl:when test="exp:Value/exp:FixedValue"><xsl:value-of select="exp:Value/exp:FixedValue"/></xsl:when><xsl:when test="exp:Value/exp:List"><xsl:choose><xsl:when test="exp:Value/exp:List/exp:Select">random.choices([<xsl:value-of select="exp:Value/exp:List/exp:Items"/>],k=<xsl:value-of select="exp:Value/exp:List/exp:Select"/>)</xsl:when><xsl:otherwise>[<xsl:value-of select="exp:Value/exp:List/exp:Items"/>]</xsl:otherwise></xsl:choose></xsl:when><xsl:when test="exp:Value/exp:Range"><xsl:choose><xsl:when test="exp:Value/exp:Range/exp:Select">[<xsl:value-of select="exp:Value/exp:Type"/>(random.<xsl:value-of select="exp:Value/exp:Range/exp:Distribution"/>(<xsl:value-of select="exp:Value/exp:Range/exp:Min"/>,<xsl:value-of select="exp:Value/exp:Range/exp:Max"/>)) for i in range(<xsl:value-of select="exp:Value/exp:Range/exp:Select"/>)]</xsl:when><xsl:otherwise>range(<xsl:value-of select="exp:Value/exp:Range/exp:Min"/>,<xsl:value-of select="exp:Value/exp:Range/exp:Max"/><xsl:if test="exp:Value/exp:Range/exp:Step">,<xsl:value-of select="exp:Value/exp:Range/exp:Step"/></xsl:if>)</xsl:otherwise></xsl:choose></xsl:when><xsl:when test="exp:Value/exp:Random">random.<xsl:value-of select="exp:Value/exp:Random/exp:Function"/>(<xsl:value-of select="exp:Value/exp:Random/exp:Arguments"/>)</xsl:when><xsl:otherwise>[]</xsl:otherwise></xsl:choose>]]<xsl:text>&#xa;</xsl:text><xsl:text>&#x9;</xsl:text></xsl:if>
	</xsl:for-each>
	</xsl:if>
	<xsl:if test="exp:Populations">
	<xsl:for-each select="exp:Populations/exp:Population">
	<xsl:if test="exp:Agent">agent_data += [["<xsl:value-of select="exp:Agent"/>",["initial_population",<xsl:if test="exp:InitialPopulationCount/exp:FixedValue"><xsl:value-of select="exp:InitialPopulationCount/exp:FixedValue"/>,<xsl:value-of select="exp:InitialPopulationCount/exp:FixedValue"/></xsl:if><xsl:if test="exp:InitialPopulationCount/exp:Range"><xsl:choose><xsl:when test="exp:InitialPopulationCount/exp:Range/exp:Select">[int(random.<xsl:value-of select="exp:InitialPopulationCount/exp:Range/exp:Distribution"/>(<xsl:value-of select="exp:InitialPopulationCount/exp:Range/exp:Min"/>,<xsl:value-of select="exp:InitialPopulationCount/exp:Range/exp:Max"/>)) for i in range(<xsl:value-of select="exp:InitialPopulationCount/exp:Range/exp:Select"/>)]</xsl:when><xsl:otherwise>range(<xsl:value-of select="exp:InitialPopulationCount/exp:Range/exp:Min"/>,<xsl:value-of select="exp:InitialPopulationCount/exp:Range/exp:Max"/><xsl:if test="exp:InitialPopulationCount/exp:Range/exp:Step">,<xsl:value-of select="exp:InitialPopulationCount/exp:Range/exp:Step"/></xsl:if>)</xsl:otherwise></xsl:choose></xsl:if>],<xsl:for-each select="exp:Variables/exp:Variable"><xsl:text>&#xa;</xsl:text><xsl:text>&#x9;</xsl:text><xsl:text>&#x9;</xsl:text><xsl:text>&#x9;</xsl:text><xsl:text>&#x9;</xsl:text><xsl:text>&#x9;</xsl:text>["<xsl:value-of select="exp:Name"/>",<xsl:if test="exp:Value/exp:FixedValue"><xsl:value-of select="exp:Value/exp:FixedValue"/></xsl:if><xsl:if test="exp:Value/exp:Range"><xsl:choose><xsl:when test="exp:Value/exp:Range/exp:Select">[<xsl:value-of select="exp:Value/exp:Type"/>(random.<xsl:value-of select="exp:Value/exp:Range/exp:Distribution"/>(<xsl:value-of select="exp:Value/exp:Range/exp:Min"/>,<xsl:value-of select="exp:Value/exp:Range/exp:Max"/>)) for i in range(<xsl:value-of select="exp:Value/exp:Range/exp:Select"/>)]</xsl:when><xsl:otherwise>range(<xsl:value-of select="exp:Value/exp:Range/exp:Min"/>,<xsl:value-of select="exp:Value/exp:Range/exp:Max"/><xsl:if test="exp:Value/exp:Range/exp:Step">,<xsl:value-of select="exp:Value/exp:Range/exp:Step"/></xsl:if>)</xsl:otherwise></xsl:choose></xsl:if>]<xsl:if test="not(position()=last())">,</xsl:if></xsl:for-each>]]<xsl:text>&#xa;</xsl:text><xsl:text>&#x9;</xsl:text></xsl:if>
	</xsl:for-each>
	</xsl:if>
	print("global_data",global_data)
	#print()
	global_data.sort(key=len)
	agent_data.sort(key=len)
	<xsl:if test="exp:Files/exp:Prefix">
	prefix_components = []
	<xsl:for-each select="exp:Files/exp:Prefix/exp:AltersWith">prefix_components += [["<xsl:value-of select="text()"/>",[x[1][0] for x in global_data if x[0]=="<xsl:value-of select="text()"/>"][0] if len(global_data)>0 else "N/A"]]<xsl:text>&#xa;</xsl:text><xsl:text>&#x9;</xsl:text></xsl:for-each>
	<xsl:for-each select="exp:Files/exp:Prefix/exp:Alteration">prefix_components += [["<xsl:value-of select="exp:Variable/exp:Name"/>", <xsl:if test="exp:Variable/exp:Type = 'str'">"</xsl:if><xsl:value-of select="exp:Variable/exp:Initial"/><xsl:if test="exp:Variable/exp:Type = 'str'">"</xsl:if>]]<xsl:text>&#xa;</xsl:text><xsl:text>&#x9;</xsl:text></xsl:for-each>
	print(prefix_components)
	prefix_strings = [str(y) for x in prefix_components for y in x]
	print(prefix_strings)
	prefix = <xsl:choose><xsl:when test="exp:Files/exp:Prefix/exp:Delimiter">"<xsl:value-of select="exp:Files/exp:Prefix/exp:Delimiter"/>"</xsl:when><xsl:otherwise>"_"</xsl:otherwise></xsl:choose>.join(prefix_strings)
	print(prefix)
	</xsl:if>
	parameter_count = 0
	if len(global_data)>0:
		constructed_data = [x for y in global_data for x in y[1]]
		for current,others in itertools.combinations(global_data,2):
			print(current,others)
			for i in current[1]:
				for j in others[1]:
					#print("outer loop parameter",i,current)
					#print("inner loop parameter",j,others)
					current_global = []
					current_agent = []
					<xsl:choose>
					<xsl:when test="exp:Files/exp:Prefix">
					#initial_state(str(prefix),"<xsl:value-of select="exp:Files/exp:InitialFileName"/>",current_global,current_agent)
					print("prefix components",prefix_components)
					print("current",current[0],current)
					print("others",others[0],others)
					prefix_components = [x if (not x[0]==current[0] and not x[0]==others[0]) else [x[0],i] if x[0]==current[0]  else [x[0],j] for x in prefix_components]
					<xsl:for-each select="exp:Files/exp:Prefix/exp:Alteration">prefix_components = [x if not x[0]=="<xsl:value-of select="exp:Variable/exp:Name"/>" else [x[0],x[1]+<xsl:choose><xsl:when test="exp:Variable/exp:Type = 'str'">"</xsl:when><xsl:otherwise><xsl:value-of select="exp:Variable/exp:Type"/>(</xsl:otherwise></xsl:choose><xsl:value-of select="exp:Variable/exp:Update"/><xsl:choose><xsl:when test="exp:Variable/exp:Type = 'str'">"</xsl:when><xsl:otherwise>)</xsl:otherwise></xsl:choose>] for x in prefix_components]<xsl:text>&#xa;</xsl:text></xsl:for-each> 
					prefix_strings = [str(y) for x in prefix_components for y in x]
					prefix = <xsl:choose><xsl:when test="exp:Files/exp:Prefix/exp:Delimiter">"<xsl:value-of select="exp:Files/exp:Prefix/exp:Delimiter"/>"</xsl:when><xsl:otherwise>"_"</xsl:otherwise></xsl:choose>.join(prefix_strings)
					print(prefix)
					</xsl:when>
					<xsl:otherwise>#initial_state("","<xsl:value-of select="exp:Files/exp:InitialFileName"/>",current_global,current_agent)</xsl:otherwise>
					</xsl:choose>
	return global_data,agent_data

generate_initial_states<xsl:if test="exp:GeneratorName">_<xsl:value-of select="exp:GeneratorName"/></xsl:if>()
#Agent data stored in list of lists
base_agent_information = [<xsl:if test="exp:Populations"><xsl:for-each select="exp:Populations/exp:Population">
["<xsl:value-of select="exp:Agent"/>",["initial_population",<xsl:if test="exp:InitialPopulationCount/exp:FixedValue"><xsl:value-of select="exp:InitialPopulationCount/exp:FixedValue"/>,<xsl:value-of select="exp:InitialPopulationCount/exp:FixedValue"/></xsl:if><xsl:if test="exp:InitialPopulationCount/exp:Range"><xsl:value-of select="exp:InitialPopulationCount/exp:Range/exp:Min"/>,<xsl:value-of select="exp:InitialPopulationCount/exp:Range/exp:Max"/></xsl:if>],<xsl:for-each select="exp:Variables/exp:Variable">["<xsl:value-of select="exp:Name"/>",<xsl:if test="exp:Value/exp:FixedValue"><xsl:value-of select="exp:Value/exp:FixedValue"/>,<xsl:value-of select="exp:Value/exp:FixedValue"/></xsl:if><xsl:if test="exp:Value/exp:Range"><xsl:value-of select="exp:Value/exp:Range/exp:Min"/>,<xsl:value-of select="exp:Value/exp:Range/exp:Max"/></xsl:if>],</xsl:for-each>],</xsl:for-each></xsl:if>]

#Create initial state
#initial_state_creation<xsl:if test="exp:GeneratorName">_<xsl:value-of select="exp:GeneratorName"/></xsl:if>("<xsl:value-of select="exp:Files/exp:FileName"/>",base_agent_information)
</xsl:for-each>
</xsl:if>
</xsl:if>




<xsl:if test="exp:Experimentation/exp:ExperimentSet">
#ExperimentSet
<xsl:for-each select="exp:Experimentation/exp:ExperimentSet/exp:Experiment">
############## <xsl:if test="exp:ExperimentName"><xsl:value-of select="exp:ExperimentName" /></xsl:if> ############
<xsl:if test="exp:Configuration">
<xsl:if test="exp:Configuration/exp:ExperimentVariables">
<xsl:for-each select="exp:Configuration/exp:ExperimentVariables/exp:Variable">
<xsl:value-of select="exp:Name" /> = <xsl:if test="not(exp:Type='tuple')"><xsl:value-of select="exp:Type" /></xsl:if>(<xsl:value-of select="exp:Value" />)
</xsl:for-each>
</xsl:if>

<xsl:if test="exp:Configuration/exp:ExperimentFunctions">
<xsl:for-each select="exp:Configuration/exp:ExperimentFunctions/exp:Function">
def <xsl:value-of select="exp:Name" />(<xsl:for-each select="exp:Arguments/exp:Argument"><xsl:value-of select="text()"/><xsl:if test="not(position()=last())">,</xsl:if></xsl:for-each>):
	<xsl:if test="exp:GlobalVariables">global <xsl:for-each select="exp:GlobalVariables/exp:Global"><xsl:value-of select="text()"/><xsl:if test="not(position()=last())">,&#160;</xsl:if></xsl:for-each><xsl:text>&#xa;</xsl:text><xsl:text>&#x9;</xsl:text></xsl:if>
	<xsl:if test="exp:Returns"><xsl:for-each select="exp:Returns/exp:Return"><xsl:value-of select="text()"/> = None<xsl:text>&#xa;</xsl:text></xsl:for-each></xsl:if>
	#Model executable
	#executable = ""
	#simulation_command = ""
	#if os.name=='nt':
	#	executable = "../../<xsl:value-of select="../../../exp:Model/exp:ExecutableLocation" />/<xsl:value-of select="../../../exp:Model/exp:ModelName" />.exe"
	#	simulation_command = executable+" ../../<xsl:value-of select="../../../exp:InitialState/exp:Location"/>/<xsl:if test="../../../exp:InitialState/exp:File"><xsl:value-of select="../../../exp:InitialState/exp:File"/></xsl:if><xsl:if test="../../../exp:InitialState/exp:FileName"><xsl:value-of select="../../../exp:InitialState/exp:FileName"/></xsl:if><xsl:if test="../../../exp:InitialState/exp:Generator"><xsl:if test="../../../../../exp:InitialStates/exp:InitialState/exp:GeneratorName = ../../../exp:InitialState/exp:Generator"><xsl:value-of select="../../../../../exp:InitialStates/exp:InitialState/exp:Files/exp:FileName"/></xsl:if></xsl:if>.xml <xsl:value-of select="../../exp:Iterations"/>"
	#else:
	#	executable = "./../../<xsl:value-of select="../../../exp:Model/exp:ExecutableLocation" />/<xsl:value-of select="../../../exp:Model/exp:ModelName" />"
	#	simulation_command = executable+" ../../<xsl:value-of select="../../../exp:InitialState/exp:Location"/>/<xsl:if test="../../../exp:InitialState/exp:File"><xsl:value-of select="../../../exp:InitialState/exp:File"/></xsl:if><xsl:if test="../../../exp:InitialState/exp:FileName"><xsl:value-of select="../../../exp:InitialState/exp:FileName"/></xsl:if><xsl:if test="../../../exp:InitialState/exp:Generator"><xsl:if test="../../../../../exp:InitialStates/exp:InitialState/exp:GeneratorName = ../../../exp:InitialState/exp:Generator"><xsl:value-of select="../../../../../exp:InitialStates/exp:InitialState/exp:Files/exp:FileName"/></xsl:if></xsl:if>.xml <xsl:value-of select="../../exp:Iterations"/>"

	<xsl:if test="not(../../../exp:InitialState/exp:Generator)">
	#Run simulation
	#os.system(simulation_command)
	#Parse results
	#results_file = open("../../<xsl:value-of select="../../../exp:InitialState/exp:SimulationOutputLocation"/>","r")
	#results = results_file.readlines()
	#results_file.close()
	</xsl:if>

	<xsl:if test="../../../exp:InitialState/exp:Generator">
	#Initial state creator
	<xsl:if test="../../exp:Repeats">
	#Run for desired number of repeats
	#for i in range(<xsl:value-of select="../../exp:Repeats"/>):
		</xsl:if>#initial_state_creation_<xsl:value-of select="../../../exp:InitialState/exp:Generator"/>(file_name,base_agent_information)
		#Run simulation
		#os.system(simulation_command)
		#Parse results
		#results_file = open("../../<xsl:value-of select="../../../exp:InitialState/exp:SimulationOutputLocation"/>","r")
		#results = results_file.readlines()
		#results_file.close()
	</xsl:if>
	
	return <xsl:if test="exp:Returns"><xsl:for-each select="exp:Returns/exp:Return"><xsl:value-of select="text()"/><xsl:if test="not(position()=last())">&#160;</xsl:if></xsl:for-each></xsl:if><xsl:text>&#xa;</xsl:text>	
</xsl:for-each>
</xsl:if>
<xsl:if test="not(exp:Configuration/exp:ExperimentFunctions) and exp:Configuration"><xsl:if test="exp:Configuration/exp:Repeats">
#Run for desired number of repeats
#for i in range(<xsl:value-of select="exp:Configuration/exp:Repeats"/>):
	</xsl:if>#initial_state_creation_<xsl:value-of select="exp:InitialState/exp:Generator"/>(file_name,base_agent_information)
	#Run simulation
	#os.system(simulation_command)
	#Parse results
	#results_file = open("../../<xsl:value-of select="exp:InitialState/exp:SimulationOutputLocation"/>","r")
	#results = results_file.readlines()
	#results_file.close()
</xsl:if>
</xsl:if>
</xsl:for-each>
</xsl:if>

</xsl:template>
</xsl:stylesheet>