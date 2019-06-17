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
	if not os.path.exists(PROJECT_DIRECTORY+"<xsl:value-of select="exp:Experimentation/exp:InitialStates/@baseDirectory"/>"):
		os.mkdir(PROJECT_DIRECTORY+"<xsl:value-of select="exp:Experimentation/exp:InitialStates/@baseDirectory"/>")
	SAVE_DIRECTORY = PROJECT_DIRECTORY+"<xsl:value-of select="exp:Experimentation/exp:InitialStates/@baseDirectory"/>"+"/"
	if not os.path.exists(SAVE_DIRECTORY+str(save_location)):
		os.mkdir(SAVE_DIRECTORY+str(save_location))
	print("Creating initial state in",SAVE_DIRECTORY,save_location,"/",file_name,"\n")
	initial_state_file = open(SAVE_DIRECTORY+str(save_location)+"/"+str(file_name)+".xml","w")
	initial_state_file.write("&lt;states&gt;\n&lt;itno&gt;0&lt;/itno&gt;\n&lt;environment&gt;\n")
	if len(global_information)>0:
		for g in range(len(global_information[0])):
			initial_state_file.write("&lt;"+str(global_information[0][g])+"&gt;"+str(global_information[1][g])+"&lt;/"+str(global_information[0][g])+"&gt;\n")
	initial_state_file.write("&lt;/environment&gt;\n")
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
				initial_state_file.write("&lt;xagent&gt;\n")
				initial_state_file.write("&lt;name&gt;"+str(agent_name)+"&lt;/name&gt;\n")
				initial_state_file.write("&lt;id&gt;"+str(agent_id)+"&lt;/id&gt;\n")
				for k in agent_information[i]:
					if not (k[0]=="initial_population" or k==agent_name):
						if len(k[1])>1:
							if len(k[1])==3:
								random_method = getattr(random, k[1][2])
								initial_state_file.write("&lt;"+str(k[0])+"&gt;"+str(random_method(k[1][0],k[1][1]))+"&lt;/"+str(k[0])+"&gt;\n")
							else:
								initial_state_file.write("&lt;"+str(k[0])+"&gt;"+str(random.uniform(k[1][0],k[1][1]))+"&lt;/"+str(k[0])+"&gt;\n")
						elif type(k[1][0])==type(int()):
							initial_state_file.write("&lt;"+str(k[0])+"&gt;"+str(int(k[1][0]))+"&lt;/"+str(k[0])+"&gt;\n")
						elif type(k[1][0])==type(float()):
							initial_state_file.write("&lt;"+str(k[0])+"&gt;"+str(float(k[1][0]))+"&lt;/"+str(k[0])+"&gt;\n")
						
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
	global_data = {<xsl:for-each select="exp:Globals/exp:Global">"<xsl:value-of select="exp:Name"/>":<xsl:choose><xsl:when test="exp:Value/exp:FixedValue"><xsl:value-of select="exp:Value/exp:FixedValue"/></xsl:when><xsl:when test="exp:Value/exp:List"><xsl:choose><xsl:when test="exp:Value/exp:List/exp:Select">random.choices([<xsl:value-of select="exp:Value/exp:List/exp:Items"/>],k=<xsl:value-of select="exp:Value/exp:List/exp:Select"/>)</xsl:when><xsl:otherwise>[<xsl:value-of select="exp:Value/exp:List/exp:Items"/>]</xsl:otherwise></xsl:choose></xsl:when><xsl:when test="exp:Value/exp:Range"><xsl:choose><xsl:when test="exp:Value/exp:Range/exp:Select">[<xsl:value-of select="exp:Value/exp:Type"/>(random.<xsl:value-of select="exp:Value/exp:Range/exp:Distribution"/>(<xsl:value-of select="exp:Value/exp:Range/exp:Min"/>,<xsl:value-of select="exp:Value/exp:Range/exp:Max"/>)) for i in range(<xsl:value-of select="exp:Value/exp:Range/exp:Select"/>)]</xsl:when><xsl:otherwise>range(<xsl:value-of select="exp:Value/exp:Range/exp:Min"/>,<xsl:value-of select="exp:Value/exp:Range/exp:Max"/><xsl:if test="exp:Value/exp:Range/exp:Step">,<xsl:value-of select="exp:Value/exp:Range/exp:Step"/></xsl:if>)</xsl:otherwise></xsl:choose></xsl:when><xsl:when test="exp:Value/exp:Random">random.<xsl:value-of select="exp:Value/exp:Random/exp:Function"/>(<xsl:value-of select="exp:Value/exp:Random/exp:Arguments"/>)</xsl:when><xsl:otherwise>[]</xsl:otherwise></xsl:choose><xsl:if test="not(position()=last())">,</xsl:if></xsl:for-each>}
	<!-- <xsl:for-each select="exp:Globals/exp:Global">
	<xsl:if test="exp:Value">global_data += [["<xsl:value-of select="exp:Name"/>", <xsl:choose><xsl:when test="exp:Value/exp:FixedValue"><xsl:value-of select="exp:Value/exp:FixedValue"/></xsl:when><xsl:when test="exp:Value/exp:List"><xsl:choose><xsl:when test="exp:Value/exp:List/exp:Select">random.choices([<xsl:value-of select="exp:Value/exp:List/exp:Items"/>],k=<xsl:value-of select="exp:Value/exp:List/exp:Select"/>)</xsl:when><xsl:otherwise>[<xsl:value-of select="exp:Value/exp:List/exp:Items"/>]</xsl:otherwise></xsl:choose></xsl:when><xsl:when test="exp:Value/exp:Range"><xsl:choose><xsl:when test="exp:Value/exp:Range/exp:Select">[<xsl:value-of select="exp:Value/exp:Type"/>(random.<xsl:value-of select="exp:Value/exp:Range/exp:Distribution"/>(<xsl:value-of select="exp:Value/exp:Range/exp:Min"/>,<xsl:value-of select="exp:Value/exp:Range/exp:Max"/>)) for i in range(<xsl:value-of select="exp:Value/exp:Range/exp:Select"/>)]</xsl:when><xsl:otherwise>range(<xsl:value-of select="exp:Value/exp:Range/exp:Min"/>,<xsl:value-of select="exp:Value/exp:Range/exp:Max"/><xsl:if test="exp:Value/exp:Range/exp:Step">,<xsl:value-of select="exp:Value/exp:Range/exp:Step"/></xsl:if>)</xsl:otherwise></xsl:choose></xsl:when><xsl:when test="exp:Value/exp:Random">random.<xsl:value-of select="exp:Value/exp:Random/exp:Function"/>(<xsl:value-of select="exp:Value/exp:Random/exp:Arguments"/>)</xsl:when><xsl:otherwise>[]</xsl:otherwise></xsl:choose>]]<xsl:text>&#xa;</xsl:text><xsl:text>&#x9;</xsl:text></xsl:if>
	</xsl:for-each> -->
	</xsl:if>
	<xsl:if test="exp:Populations">
	<xsl:for-each select="exp:Populations/exp:Population">
	<xsl:value-of select="exp:Agent"/> = {<xsl:if test="exp:InitialPopulationCount">"initial_population":<xsl:if test="exp:InitialPopulationCount/exp:FixedValue">[<xsl:value-of select="exp:InitialPopulationCount/exp:FixedValue"/>]</xsl:if><xsl:if test="exp:InitialPopulationCount/exp:Range"><xsl:choose><xsl:when test="exp:InitialPopulationCount/exp:Range/exp:Select">[int(random.<xsl:value-of select="exp:InitialPopulationCount/exp:Range/exp:Distribution"/>(<xsl:value-of select="exp:InitialPopulationCount/exp:Range/exp:Min"/>,<xsl:value-of select="exp:InitialPopulationCount/exp:Range/exp:Max"/>)) for i in range(<xsl:value-of select="exp:InitialPopulationCount/exp:Range/exp:Select"/>)]</xsl:when><xsl:otherwise>range(<xsl:value-of select="exp:InitialPopulationCount/exp:Range/exp:Min"/>,<xsl:value-of select="exp:InitialPopulationCount/exp:Range/exp:Max"/><xsl:if test="exp:InitialPopulationCount/exp:Range/exp:Step">,<xsl:value-of select="exp:InitialPopulationCount/exp:Range/exp:Step"/></xsl:if>)</xsl:otherwise></xsl:choose></xsl:if>, </xsl:if><xsl:for-each select="exp:Variables/exp:Variable"><xsl:if test="not(exp:Value/exp:PerAgentRange)">"<xsl:value-of select="exp:Name"/>":<xsl:choose><xsl:when test="exp:Value/exp:FixedValue">[<xsl:value-of select="exp:Value/exp:FixedValue"/>]</xsl:when><xsl:when test="exp:Value/exp:Range"><xsl:choose><xsl:when test="exp:Value/exp:Range/exp:Select">[<xsl:value-of select="exp:Value/exp:Type"/>(random.<xsl:value-of select="exp:Value/exp:Range/exp:Distribution"/>(<xsl:value-of select="exp:Value/exp:Range/exp:Min"/>,<xsl:value-of select="exp:Value/exp:Range/exp:Max"/>)) for i in range(<xsl:value-of select="exp:Value/exp:Range/exp:Select"/>)]</xsl:when><xsl:otherwise>range(<xsl:value-of select="exp:Value/exp:Range/exp:Min"/>,<xsl:value-of select="exp:Value/exp:Range/exp:Max"/><xsl:if test="exp:Value/exp:Range/exp:Step">,<xsl:value-of select="exp:Value/exp:Range/exp:Step"/></xsl:if>)</xsl:otherwise></xsl:choose></xsl:when><xsl:otherwise>[]</xsl:otherwise></xsl:choose><xsl:if test="not(position()=last())">,</xsl:if></xsl:if></xsl:for-each>}
	<xsl:value-of select="exp:Agent"/>_vary_per_agent = {<xsl:for-each select="exp:Variables/exp:Variable"><xsl:if test="exp:Value/exp:PerAgentRange">"<xsl:value-of select="exp:Name"/>":[<xsl:value-of select="exp:Value/exp:PerAgentRange/exp:Min"/>,<xsl:value-of select="exp:Value/exp:PerAgentRange/exp:Max"/><xsl:if test="exp:Value/exp:PerAgentRange/exp:Distribution">,"<xsl:value-of select="exp:Value/exp:PerAgentRange/exp:Distribution"/>"</xsl:if>],</xsl:if></xsl:for-each>}<xsl:text>&#xa;</xsl:text><xsl:text>&#x9;</xsl:text>
	</xsl:for-each>
	agent_data = {<xsl:for-each select="exp:Populations/exp:Population">"<xsl:value-of select="exp:Agent"/>":<xsl:value-of select="exp:Agent"/><xsl:if test="not(position()=last())">,</xsl:if></xsl:for-each>}
	<!-- <xsl:for-each select="exp:Populations/exp:Population">
	<xsl:if test="exp:Agent">agent_data += [["<xsl:value-of select="exp:Agent"/>",["initial_population",<xsl:if test="exp:InitialPopulationCount/exp:FixedValue">[<xsl:value-of select="exp:InitialPopulationCount/exp:FixedValue"/>]</xsl:if><xsl:if test="exp:InitialPopulationCount/exp:Range"><xsl:choose><xsl:when test="exp:InitialPopulationCount/exp:Range/exp:Select">[int(random.<xsl:value-of select="exp:InitialPopulationCount/exp:Range/exp:Distribution"/>(<xsl:value-of select="exp:InitialPopulationCount/exp:Range/exp:Min"/>,<xsl:value-of select="exp:InitialPopulationCount/exp:Range/exp:Max"/>)) for i in range(<xsl:value-of select="exp:InitialPopulationCount/exp:Range/exp:Select"/>)]</xsl:when><xsl:otherwise>range(<xsl:value-of select="exp:InitialPopulationCount/exp:Range/exp:Min"/>,<xsl:value-of select="exp:InitialPopulationCount/exp:Range/exp:Max"/><xsl:if test="exp:InitialPopulationCount/exp:Range/exp:Step">,<xsl:value-of select="exp:InitialPopulationCount/exp:Range/exp:Step"/></xsl:if>)</xsl:otherwise></xsl:choose></xsl:if>],<xsl:for-each select="exp:Variables/exp:Variable"><xsl:text>&#xa;</xsl:text><xsl:text>&#x9;</xsl:text><xsl:text>&#x9;</xsl:text><xsl:text>&#x9;</xsl:text><xsl:text>&#x9;</xsl:text><xsl:text>&#x9;</xsl:text>["<xsl:value-of select="exp:Name"/>",<xsl:if test="exp:Value/exp:FixedValue">[<xsl:value-of select="exp:Value/exp:FixedValue"/>]</xsl:if><xsl:if test="exp:Value/exp:Range"><xsl:choose><xsl:when test="exp:Value/exp:Range/exp:Select">[<xsl:value-of select="exp:Value/exp:Type"/>(random.<xsl:value-of select="exp:Value/exp:Range/exp:Distribution"/>(<xsl:value-of select="exp:Value/exp:Range/exp:Min"/>,<xsl:value-of select="exp:Value/exp:Range/exp:Max"/>)) for i in range(<xsl:value-of select="exp:Value/exp:Range/exp:Select"/>)]</xsl:when><xsl:otherwise>range(<xsl:value-of select="exp:Value/exp:Range/exp:Min"/>,<xsl:value-of select="exp:Value/exp:Range/exp:Max"/><xsl:if test="exp:Value/exp:Range/exp:Step">,<xsl:value-of select="exp:Value/exp:Range/exp:Step"/></xsl:if>)</xsl:otherwise></xsl:choose></xsl:if>]<xsl:if test="not(position()=last())">,</xsl:if></xsl:for-each>]]<xsl:text>&#xa;</xsl:text><xsl:text>&#x9;</xsl:text></xsl:if>
	</xsl:for-each> -->
	vary_per_agent = {<xsl:for-each select="exp:Populations/exp:Population">"<xsl:value-of select="exp:Agent"/>":<xsl:value-of select="exp:Agent"/>_vary_per_agent<xsl:if test="not(position()=last())">,</xsl:if></xsl:for-each>}

	</xsl:if>
	<xsl:if test="exp:Files/exp:Prefix">
	prefix_components = []
	<xsl:for-each select="exp:Files/exp:Prefix/exp:AltersWith">prefix_components += [["<xsl:value-of select="text()"/>",global_data["<xsl:value-of select="text()"/>"][0] if len(global_data)>0 else "NA"]]<xsl:text>&#xa;</xsl:text><xsl:text>&#x9;</xsl:text></xsl:for-each>
	<xsl:for-each select="exp:Files/exp:Prefix/exp:Alteration">prefix_components += [["<xsl:value-of select="exp:Variable/exp:Name"/>", <xsl:if test="exp:Variable/exp:Type = 'str'">"</xsl:if><xsl:value-of select="exp:Variable/exp:Initial"/><xsl:if test="exp:Variable/exp:Type = 'str'">"</xsl:if>]]<xsl:text>&#xa;</xsl:text><xsl:text>&#x9;</xsl:text></xsl:for-each>
	prefix_strings = [str(y) for x in prefix_components for y in x]
	prefix = <xsl:choose><xsl:when test="exp:Files/exp:Prefix/exp:Delimiter">"<xsl:value-of select="exp:Files/exp:Prefix/exp:Delimiter"/>"</xsl:when><xsl:otherwise>"_"</xsl:otherwise></xsl:choose>.join(prefix_strings)
	</xsl:if>

	parameter_count = 0
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
				<xsl:choose>
				<xsl:when test="exp:Files/exp:Prefix">
				initial_state(str(prefix),"<xsl:value-of select="exp:Files/exp:InitialFileName"/>",g,current_agent_data)
				prefix_components = [x if not x[0] in g[0] else [x[0],g[1][g[0].index(x[0])]] for x in prefix_components]
				<xsl:for-each select="exp:Files/exp:Prefix/exp:Alteration">prefix_components = [x if not x[0]=="<xsl:value-of select="exp:Variable/exp:Name"/>" else [x[0],x[1]+<xsl:choose><xsl:when test="exp:Variable/exp:Type = 'str'">"</xsl:when><xsl:otherwise><xsl:value-of select="exp:Variable/exp:Type"/>(</xsl:otherwise></xsl:choose><xsl:value-of select="exp:Variable/exp:Update"/><xsl:choose><xsl:when test="exp:Variable/exp:Type = 'str'">"</xsl:when><xsl:otherwise>)</xsl:otherwise></xsl:choose>] for x in prefix_components]<xsl:text>&#xa;</xsl:text></xsl:for-each> 
				prefix_strings = [str(y) for x in prefix_components for y in x]
				prefix = <xsl:choose><xsl:when test="exp:Files/exp:Prefix/exp:Delimiter">"<xsl:value-of select="exp:Files/exp:Prefix/exp:Delimiter"/>"</xsl:when><xsl:otherwise>"_"</xsl:otherwise></xsl:choose>.join(prefix_strings)
				</xsl:when>
				<xsl:otherwise>initial_state("","<xsl:value-of select="exp:Files/exp:InitialFileName"/>",g,current_agent_data)</xsl:otherwise>
				</xsl:choose>
	elif len(global_combinations)>0:
		for g in global_combinations:
			current_agent_data = [agent+[[x[0],x[1]] for x in vary_per_agent[agent[0]].items()] for agent in agent_data]
			<xsl:choose>
			<xsl:when test="exp:Files/exp:Prefix">
			initial_state(str(prefix),"<xsl:value-of select="exp:Files/exp:InitialFileName"/>",g,current_agent_data)
			prefix_components = [x if not x[0] in g[0] else [x[0],g[1][g[0].index(x[0])]] for x in prefix_components]
			<xsl:for-each select="exp:Files/exp:Prefix/exp:Alteration">prefix_components = [x if not x[0]=="<xsl:value-of select="exp:Variable/exp:Name"/>" else [x[0],x[1]+<xsl:choose><xsl:when test="exp:Variable/exp:Type = 'str'">"</xsl:when><xsl:otherwise><xsl:value-of select="exp:Variable/exp:Type"/>(</xsl:otherwise></xsl:choose><xsl:value-of select="exp:Variable/exp:Update"/><xsl:choose><xsl:when test="exp:Variable/exp:Type = 'str'">"</xsl:when><xsl:otherwise>)</xsl:otherwise></xsl:choose>] for x in prefix_components]<xsl:text>&#xa;</xsl:text></xsl:for-each> 
			prefix_strings = [str(y) for x in prefix_components for y in x]
			prefix = <xsl:choose><xsl:when test="exp:Files/exp:Prefix/exp:Delimiter">"<xsl:value-of select="exp:Files/exp:Prefix/exp:Delimiter"/>"</xsl:when><xsl:otherwise>"_"</xsl:otherwise></xsl:choose>.join(prefix_strings)
			</xsl:when>
			<xsl:otherwise>initial_state("","<xsl:value-of select="exp:Files/exp:InitialFileName"/>",g,current_agent_data)</xsl:otherwise>
			</xsl:choose>
	elif len(agent_combinations)>0:
		for a in agent_combinations:
			current_agent_data = [agent+[[x[0],x[1]] for x in vary_per_agent[agent[0]].items()] for agent in a]
			<xsl:choose>
			<xsl:when test="exp:Files/exp:Prefix">
			initial_state(str(prefix),"<xsl:value-of select="exp:Files/exp:InitialFileName"/>",global_data,current_agent_data)
			prefix_components = [x if not x[0] in a else [x[0],a.index(x[0])[1]] for x in prefix_components]
			<xsl:for-each select="exp:Files/exp:Prefix/exp:Alteration">prefix_components = [x if not x[0]=="<xsl:value-of select="exp:Variable/exp:Name"/>" else [x[0],x[1]+<xsl:choose><xsl:when test="exp:Variable/exp:Type = 'str'">"</xsl:when><xsl:otherwise><xsl:value-of select="exp:Variable/exp:Type"/>(</xsl:otherwise></xsl:choose><xsl:value-of select="exp:Variable/exp:Update"/><xsl:choose><xsl:when test="exp:Variable/exp:Type = 'str'">"</xsl:when><xsl:otherwise>)</xsl:otherwise></xsl:choose>] for x in prefix_components]<xsl:text>&#xa;</xsl:text></xsl:for-each> 
			prefix_strings = [str(y) for x in prefix_components for y in x]
			prefix = <xsl:choose><xsl:when test="exp:Files/exp:Prefix/exp:Delimiter">"<xsl:value-of select="exp:Files/exp:Prefix/exp:Delimiter"/>"</xsl:when><xsl:otherwise>"_"</xsl:otherwise></xsl:choose>.join(prefix_strings)
			</xsl:when>
			<xsl:otherwise>initial_state("","<xsl:value-of select="exp:Files/exp:InitialFileName"/>",global_data,current_agent_data)</xsl:otherwise>
			</xsl:choose>
	else:
		print("No global or agent variations specified for experimentation\n")
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

<xsl:if test="exp:placholder_text">
	if len(global_data)>0:
		<xsl:if test="exp:Globals/exp:Global">
		for g in global_data:
			altering_parameters = [x[0] for x in global_data if not x==g]
			potential_parameter_values = [y for x in global_data for y in x[1] if not x==g]
			for current_value in g[1]:
				#print("len combinations",len(itertools.combinations(potential_parameter_values,len(global_data)-1)))
				for ap in itertools.combinations(potential_parameter_values,len(global_data)-1):
					print("current stff",g,current_value,ap)
				print("actual combinations",list(itertools.combinations(potential_parameter_values,len(global_data)-1)))
		</xsl:if>
		#for c,o,s in itertools.combinations([y for x in global_data for y in x[1] ],3):
			#print("combinations",c,o,s)
		constructed_data = [x for y in global_data for x in y[1]]
		for current_global,other_global in itertools.combinations(global_data,2):
			print(current_global,other_global)
			<!-- for current_agent,other_agent in itertools.combinations(agent_data,2):
				for i in current_global[1]:
					for j in other_global[1]:
						for a in current_agent[1]:
							for b in other_agent[1] -->
			for i in current_global[1]:
				for j in other_global[1]:
					#print("outer loop parameter",i,current_global)
					#print("inner loop parameter",j,other_global)
					#current_global = [x if not x[0]==current_global[0] for x in global_data]
					current_global_data = []
					current_agent_data = []
					

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