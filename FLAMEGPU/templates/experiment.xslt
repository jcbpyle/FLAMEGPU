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

<xsl:if test="exp:Experimentation/xmml:Imports"><xsl:for-each select="exp:Experimentation/xmml:Imports/xmml:Import"><xsl:if test="xmml:From">from <xsl:value-of select="xmml:From" /><xsl:text>&#x20;</xsl:text></xsl:if>import <xsl:value-of select="xmml:Module" /><xsl:text>&#xa;</xsl:text></xsl:for-each></xsl:if>
import os
import random
import itertools
import pycuda.driver as cuda
import pycuda.autoinit

BASE_DIRECTORY = os.getcwd()+"/"
PROJECT_DIRECTORY = BASE_DIRECTORY+"../../"
GPUS_AVAILABLE = cuda.Device(0).count()
OS_NAME = os.name
<xsl:if test="exp:Experimentation/xmml:InitialStates">
#InitialStates
<xsl:if test="exp:Experimentation/xmml:InitialStates/xmml:InitialStateFile">
initial_state_files = []
<xsl:for-each select="exp:Experimentation/xmml:InitialStates/xmml:InitialStateFile">
<xsl:if test="xmml:FileName">
initial_state_files += ["/<xsl:value-of select="xmml:Location"/>/<xsl:value-of select="xmml:FileName"/>.xml"]
</xsl:if>
</xsl:for-each>
</xsl:if>
<xsl:if test="exp:Experimentation/xmml:InitialStates/xmml:InitialStateFunction">
<xsl:for-each select="exp:Experimentation/xmml:InitialStates/xmml:InitialStateFunction">
#Initial state generator function to be created by the user
def initial_state_generator_function_<xsl:value-of select="xmml:FunctionName"/>():

	return<xsl:text>&#xa;</xsl:text>
</xsl:for-each>
</xsl:if>

<xsl:if test="exp:Experimentation/xmml:InitialStates/xmml:InitialStateGenerator">
<xsl:for-each select="exp:Experimentation/xmml:InitialStates/xmml:InitialStateGenerator">
#Generate initial states based on defined ranges/lists/values for all global and agent population variables<xsl:if test="xmml:GeneratorName"> for experiment <xsl:value-of select="xmml:GeneratorName"/></xsl:if>.
def generate_initial_states<xsl:if test="xmml:GeneratorName">_<xsl:value-of select="xmml:GeneratorName"/></xsl:if>():
	global_data = []
	agent_data = []
	vary_per_agent = []
	<xsl:if test="xmml:Globals">
	global_data = {<xsl:for-each select="xmml:Globals/xmml:Global">"<xsl:value-of select="xmml:Name"/>":<xsl:choose><xsl:when test="xmml:Value/xmml:FixedValue">[<xsl:value-of select="xmml:Value/xmml:FixedValue"/>]</xsl:when><xsl:when test="xmml:Value/xmml:List"><xsl:choose><xsl:when test="xmml:Value/xmml:List/xmml:Select">random.choices([<xsl:value-of select="xmml:Value/xmml:List/xmml:Items"/>],k=<xsl:value-of select="xmml:Value/xmml:List/xmml:Select"/>)</xsl:when><xsl:otherwise>[<xsl:value-of select="xmml:Value/xmml:List/xmml:Items"/>]</xsl:otherwise></xsl:choose></xsl:when><xsl:when test="xmml:Value/xmml:Range"><xsl:choose><xsl:when test="xmml:Value/xmml:Range/xmml:Select">[<xsl:value-of select="xmml:Value/xmml:Type"/>(random.<xsl:value-of select="xmml:Value/xmml:Range/xmml:Distribution"/>(<xsl:value-of select="xmml:Value/xmml:Range/xmml:Min"/>,<xsl:value-of select="xmml:Value/xmml:Range/xmml:Max"/>)) for i in range(<xsl:value-of select="xmml:Value/xmml:Range/xmml:Select"/>)]</xsl:when><xsl:otherwise>range(<xsl:value-of select="xmml:Value/xmml:Range/xmml:Min"/>,<xsl:value-of select="xmml:Value/xmml:Range/xmml:Max"/><xsl:if test="xmml:Value/xmml:Range/xmml:Step">,<xsl:value-of select="xmml:Value/xmml:Range/xmml:Step"/></xsl:if>)</xsl:otherwise></xsl:choose></xsl:when><xsl:when test="xmml:Value/xmml:Random"><xsl:if test="xmml:MultipleValues='False'">[</xsl:if>random.<xsl:value-of select="xmml:Value/xmml:Random/xmml:Function"/>(<xsl:value-of select="xmml:Value/xmml:Random/xmml:Arguments"/>)<xsl:if test="xmml:MultipleValues='False'">]</xsl:if></xsl:when><xsl:otherwise>[]</xsl:otherwise></xsl:choose><xsl:if test="not(position()=last())">,</xsl:if></xsl:for-each>}
	<!-- <xsl:for-each select="xmml:Globals/xmml:Global">
	<xsl:if test="xmml:Value">global_data += [["<xsl:value-of select="xmml:Name"/>", <xsl:choose><xsl:when test="xmml:Value/xmml:FixedValue"><xsl:value-of select="xmml:Value/xmml:FixedValue"/></xsl:when><xsl:when test="xmml:Value/xmml:List"><xsl:choose><xsl:when test="xmml:Value/xmml:List/xmml:Select">random.choices([<xsl:value-of select="xmml:Value/xmml:List/xmml:Items"/>],k=<xsl:value-of select="xmml:Value/xmml:List/xmml:Select"/>)</xsl:when><xsl:otherwise>[<xsl:value-of select="xmml:Value/xmml:List/xmml:Items"/>]</xsl:otherwise></xsl:choose></xsl:when><xsl:when test="xmml:Value/xmml:Range"><xsl:choose><xsl:when test="xmml:Value/xmml:Range/xmml:Select">[<xsl:value-of select="xmml:Value/xmml:Type"/>(random.<xsl:value-of select="xmml:Value/xmml:Range/xmml:Distribution"/>(<xsl:value-of select="xmml:Value/xmml:Range/xmml:Min"/>,<xsl:value-of select="xmml:Value/xmml:Range/xmml:Max"/>)) for i in range(<xsl:value-of select="xmml:Value/xmml:Range/xmml:Select"/>)]</xsl:when><xsl:otherwise>range(<xsl:value-of select="xmml:Value/xmml:Range/xmml:Min"/>,<xsl:value-of select="xmml:Value/xmml:Range/xmml:Max"/><xsl:if test="xmml:Value/xmml:Range/xmml:Step">,<xsl:value-of select="xmml:Value/xmml:Range/xmml:Step"/></xsl:if>)</xsl:otherwise></xsl:choose></xsl:when><xsl:when test="xmml:Value/xmml:Random">random.<xsl:value-of select="xmml:Value/xmml:Random/xmml:Function"/>(<xsl:value-of select="xmml:Value/xmml:Random/xmml:Arguments"/>)</xsl:when><xsl:otherwise>[]</xsl:otherwise></xsl:choose>]]<xsl:text>&#xa;</xsl:text><xsl:text>&#x9;</xsl:text></xsl:if>
	</xsl:for-each> -->
	</xsl:if>
	<xsl:if test="xmml:Populations">
	<xsl:for-each select="xmml:Populations/xmml:Population">
	<xsl:value-of select="xmml:Agent"/> = {<xsl:if test="xmml:InitialPopulationCount">"initial_population":<xsl:if test="xmml:InitialPopulationCount/xmml:FixedValue">[<xsl:value-of select="xmml:InitialPopulationCount/xmml:FixedValue"/>]</xsl:if><xsl:if test="xmml:InitialPopulationCount/xmml:Range"><xsl:choose><xsl:when test="xmml:InitialPopulationCount/xmml:Range/xmml:Select">[int(random.<xsl:value-of select="xmml:InitialPopulationCount/xmml:Range/xmml:Distribution"/>(<xsl:value-of select="xmml:InitialPopulationCount/xmml:Range/xmml:Min"/>,<xsl:value-of select="xmml:InitialPopulationCount/xmml:Range/xmml:Max"/>)) for i in range(<xsl:value-of select="xmml:InitialPopulationCount/xmml:Range/xmml:Select"/>)]</xsl:when><xsl:otherwise>range(<xsl:value-of select="xmml:InitialPopulationCount/xmml:Range/xmml:Min"/>,<xsl:value-of select="xmml:InitialPopulationCount/xmml:Range/xmml:Max"/><xsl:if test="xmml:InitialPopulationCount/xmml:Range/xmml:Step">,<xsl:value-of select="xmml:InitialPopulationCount/xmml:Range/xmml:Step"/></xsl:if>)</xsl:otherwise></xsl:choose></xsl:if>, </xsl:if><xsl:for-each select="xmml:Variables/xmml:Variable"><xsl:if test="not(xmml:Value/xmml:PerAgentRange)">"<xsl:value-of select="xmml:Name"/>":<xsl:choose><xsl:when test="xmml:Value/xmml:FixedValue">[<xsl:value-of select="xmml:Value/xmml:FixedValue"/>]</xsl:when><xsl:when test="xmml:Value/xmml:Range"><xsl:choose><xsl:when test="xmml:Value/xmml:Range/xmml:Select">[<xsl:value-of select="xmml:Value/xmml:Type"/>(random.<xsl:value-of select="xmml:Value/xmml:Range/xmml:Distribution"/>(<xsl:value-of select="xmml:Value/xmml:Range/xmml:Min"/>,<xsl:value-of select="xmml:Value/xmml:Range/xmml:Max"/>)) for i in range(<xsl:value-of select="xmml:Value/xmml:Range/xmml:Select"/>)]</xsl:when><xsl:otherwise>range(<xsl:value-of select="xmml:Value/xmml:Range/xmml:Min"/>,<xsl:value-of select="xmml:Value/xmml:Range/xmml:Max"/><xsl:if test="xmml:Value/xmml:Range/xmml:Step">,<xsl:value-of select="xmml:Value/xmml:Range/xmml:Step"/></xsl:if>)</xsl:otherwise></xsl:choose></xsl:when><xsl:otherwise>[]</xsl:otherwise></xsl:choose><xsl:if test="not(position()=last())">,</xsl:if></xsl:if></xsl:for-each>}
	<xsl:value-of select="xmml:Agent"/>_vary_per_agent = {<xsl:for-each select="xmml:Variables/xmml:Variable"><xsl:if test="xmml:Value/xmml:PerAgentRange">"<xsl:value-of select="xmml:Name"/>":[<xsl:value-of select="xmml:Value/xmml:PerAgentRange/xmml:Min"/>,<xsl:value-of select="xmml:Value/xmml:PerAgentRange/xmml:Max"/><xsl:if test="xmml:Value/xmml:PerAgentRange/xmml:Distribution">,"<xsl:value-of select="xmml:Value/xmml:PerAgentRange/xmml:Distribution"/>"</xsl:if><xsl:if test="xmml:Value/xmml:Type">,<xsl:value-of select="xmml:Value/xmml:Type"/></xsl:if>],</xsl:if></xsl:for-each>}<xsl:text>&#xa;</xsl:text><xsl:text>&#x9;</xsl:text>
	</xsl:for-each>
	agent_data = {<xsl:for-each select="xmml:Populations/xmml:Population">"<xsl:value-of select="xmml:Agent"/>":<xsl:value-of select="xmml:Agent"/><xsl:if test="not(position()=last())">,</xsl:if></xsl:for-each>}
	<!-- <xsl:for-each select="xmml:Populations/xmml:Population">
	<xsl:if test="xmml:Agent">agent_data += [["<xsl:value-of select="xmml:Agent"/>",["initial_population",<xsl:if test="xmml:InitialPopulationCount/xmml:FixedValue">[<xsl:value-of select="xmml:InitialPopulationCount/xmml:FixedValue"/>]</xsl:if><xsl:if test="xmml:InitialPopulationCount/xmml:Range"><xsl:choose><xsl:when test="xmml:InitialPopulationCount/xmml:Range/xmml:Select">[int(random.<xsl:value-of select="xmml:InitialPopulationCount/xmml:Range/xmml:Distribution"/>(<xsl:value-of select="xmml:InitialPopulationCount/xmml:Range/xmml:Min"/>,<xsl:value-of select="xmml:InitialPopulationCount/xmml:Range/xmml:Max"/>)) for i in range(<xsl:value-of select="xmml:InitialPopulationCount/xmml:Range/xmml:Select"/>)]</xsl:when><xsl:otherwise>range(<xsl:value-of select="xmml:InitialPopulationCount/xmml:Range/xmml:Min"/>,<xsl:value-of select="xmml:InitialPopulationCount/xmml:Range/xmml:Max"/><xsl:if test="xmml:InitialPopulationCount/xmml:Range/xmml:Step">,<xsl:value-of select="xmml:InitialPopulationCount/xmml:Range/xmml:Step"/></xsl:if>)</xsl:otherwise></xsl:choose></xsl:if>],<xsl:for-each select="xmml:Variables/xmml:Variable"><xsl:text>&#xa;</xsl:text><xsl:text>&#x9;</xsl:text><xsl:text>&#x9;</xsl:text><xsl:text>&#x9;</xsl:text><xsl:text>&#x9;</xsl:text><xsl:text>&#x9;</xsl:text>["<xsl:value-of select="xmml:Name"/>",<xsl:if test="xmml:Value/xmml:FixedValue">[<xsl:value-of select="xmml:Value/xmml:FixedValue"/>]</xsl:if><xsl:if test="xmml:Value/xmml:Range"><xsl:choose><xsl:when test="xmml:Value/xmml:Range/xmml:Select">[<xsl:value-of select="xmml:Value/xmml:Type"/>(random.<xsl:value-of select="xmml:Value/xmml:Range/xmml:Distribution"/>(<xsl:value-of select="xmml:Value/xmml:Range/xmml:Min"/>,<xsl:value-of select="xmml:Value/xmml:Range/xmml:Max"/>)) for i in range(<xsl:value-of select="xmml:Value/xmml:Range/xmml:Select"/>)]</xsl:when><xsl:otherwise>range(<xsl:value-of select="xmml:Value/xmml:Range/xmml:Min"/>,<xsl:value-of select="xmml:Value/xmml:Range/xmml:Max"/><xsl:if test="xmml:Value/xmml:Range/xmml:Step">,<xsl:value-of select="xmml:Value/xmml:Range/xmml:Step"/></xsl:if>)</xsl:otherwise></xsl:choose></xsl:if>]<xsl:if test="not(position()=last())">,</xsl:if></xsl:for-each>]]<xsl:text>&#xa;</xsl:text><xsl:text>&#x9;</xsl:text></xsl:if>
	</xsl:for-each> -->
	vary_per_agent = {<xsl:for-each select="xmml:Populations/xmml:Population">"<xsl:value-of select="xmml:Agent"/>":<xsl:value-of select="xmml:Agent"/>_vary_per_agent<xsl:if test="not(position()=last())">,</xsl:if></xsl:for-each>}

	</xsl:if>
	prefix_components = []
	<xsl:if test="xmml:Files/xmml:Prefix">
	<xsl:for-each select="xmml:Files/xmml:Prefix/xmml:AltersWith">prefix_components += [["<xsl:value-of select="text()"/>",global_data["<xsl:value-of select="text()"/>"][0] if len(global_data)>0 else "NA"]]<xsl:text>&#xa;</xsl:text><xsl:text>&#x9;</xsl:text></xsl:for-each>
	<xsl:for-each select="xmml:Files/xmml:Prefix/xmml:Alteration">prefix_components += [["<xsl:value-of select="xmml:Variable/xmml:Name"/>", <xsl:if test="xmml:Variable/xmml:Type = 'str'">"</xsl:if><xsl:value-of select="xmml:Variable/xmml:Initial"/><xsl:if test="xmml:Variable/xmml:Type = 'str'">"</xsl:if>]]<xsl:text>&#xa;</xsl:text><xsl:text>&#x9;</xsl:text></xsl:for-each>
	prefix_strings = [str(y) for x in prefix_components for y in x]
	prefix = <xsl:choose><xsl:when test="xmml:Files/xmml:Prefix/xmml:Delimiter">"<xsl:value-of select="xmml:Files/xmml:Prefix/xmml:Delimiter"/>"</xsl:when><xsl:otherwise>"_"</xsl:otherwise></xsl:choose>.join(prefix_strings)
	</xsl:if>
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
				<xsl:when test="xmml:Files/xmml:Prefix">
				initial_state(str(prefix),"<xsl:value-of select="xmml:Files/xmml:InitialFileName"/>",g,current_agent_data)
				prefix_components = [x if not x[0] in g[0] else [x[0],g[1][g[0].index(x[0])]] for x in prefix_components]
				<xsl:for-each select="xmml:Files/xmml:Prefix/xmml:Alteration">prefix_components = [x if not x[0]=="<xsl:value-of select="xmml:Variable/xmml:Name"/>" else [x[0],x[1]+<xsl:choose><xsl:when test="xmml:Variable/xmml:Type = 'str'">"</xsl:when><xsl:otherwise><xsl:value-of select="xmml:Variable/xmml:Type"/>(</xsl:otherwise></xsl:choose><xsl:value-of select="xmml:Variable/xmml:Update"/><xsl:choose><xsl:when test="xmml:Variable/xmml:Type = 'str'">"</xsl:when><xsl:otherwise>)</xsl:otherwise></xsl:choose>] for x in prefix_components]<xsl:text>&#xa;</xsl:text></xsl:for-each> 
				prefix_strings = [str(y) for x in prefix_components for y in x]
				prefix = <xsl:choose><xsl:when test="xmml:Files/xmml:Prefix/xmml:Delimiter">"<xsl:value-of select="xmml:Files/xmml:Prefix/xmml:Delimiter"/>"</xsl:when><xsl:otherwise>"_"</xsl:otherwise></xsl:choose>.join(prefix_strings)
				</xsl:when>
				<xsl:otherwise>initial_state("","<xsl:value-of select="xmml:Files/xmml:InitialFileName"/>",g,current_agent_data)</xsl:otherwise>
				</xsl:choose>
	elif len(global_combinations)>0:
		for g in global_combinations:
			current_agent_data = [agent+[[x[0],x[1]] for x in vary_per_agent[agent[0]].items()] for agent in agent_data]
			<xsl:choose>
			<xsl:when test="xmml:Files/xmml:Prefix">
			initial_state(str(prefix),"<xsl:value-of select="xmml:Files/xmml:InitialFileName"/>",g,current_agent_data)
			prefix_components = [x if not x[0] in g[0] else [x[0],g[1][g[0].index(x[0])]] for x in prefix_components]
			<xsl:for-each select="xmml:Files/xmml:Prefix/xmml:Alteration">prefix_components = [x if not x[0]=="<xsl:value-of select="xmml:Variable/xmml:Name"/>" else [x[0],x[1]+<xsl:choose><xsl:when test="xmml:Variable/xmml:Type = 'str'">"</xsl:when><xsl:otherwise><xsl:value-of select="xmml:Variable/xmml:Type"/>(</xsl:otherwise></xsl:choose><xsl:value-of select="xmml:Variable/xmml:Update"/><xsl:choose><xsl:when test="xmml:Variable/xmml:Type = 'str'">"</xsl:when><xsl:otherwise>)</xsl:otherwise></xsl:choose>] for x in prefix_components]<xsl:text>&#xa;</xsl:text></xsl:for-each> 
			prefix_strings = [str(y) for x in prefix_components for y in x]
			prefix = <xsl:choose><xsl:when test="xmml:Files/xmml:Prefix/xmml:Delimiter">"<xsl:value-of select="xmml:Files/xmml:Prefix/xmml:Delimiter"/>"</xsl:when><xsl:otherwise>"_"</xsl:otherwise></xsl:choose>.join(prefix_strings)
			</xsl:when>
			<xsl:otherwise>initial_state("","<xsl:value-of select="xmml:Files/xmml:InitialFileName"/>",g,current_agent_data)</xsl:otherwise>
			</xsl:choose>
	elif len(agent_combinations)>0:
		for a in agent_combinations:
			current_agent_data = [agent+[[x[0],x[1]] for x in vary_per_agent[agent[0]].items()] for agent in a]
			<xsl:choose>
			<xsl:when test="xmml:Files/xmml:Prefix">
			initial_state(str(prefix),"<xsl:value-of select="xmml:Files/xmml:InitialFileName"/>",global_data,current_agent_data)
			prefix_components = [x if not x[0] in a else [x[0],a.index(x[0])[1]] for x in prefix_components]
			<xsl:for-each select="xmml:Files/xmml:Prefix/xmml:Alteration">prefix_components = [x if not x[0]=="<xsl:value-of select="xmml:Variable/xmml:Name"/>" else [x[0],x[1]+<xsl:choose><xsl:when test="xmml:Variable/xmml:Type = 'str'">"</xsl:when><xsl:otherwise><xsl:value-of select="xmml:Variable/xmml:Type"/>(</xsl:otherwise></xsl:choose><xsl:value-of select="xmml:Variable/xmml:Update"/><xsl:choose><xsl:when test="xmml:Variable/xmml:Type = 'str'">"</xsl:when><xsl:otherwise>)</xsl:otherwise></xsl:choose>] for x in prefix_components]<xsl:text>&#xa;</xsl:text></xsl:for-each> 
			prefix_strings = [str(y) for x in prefix_components for y in x]
			prefix = <xsl:choose><xsl:when test="xmml:Files/xmml:Prefix/xmml:Delimiter">"<xsl:value-of select="xmml:Files/xmml:Prefix/xmml:Delimiter"/>"</xsl:when><xsl:otherwise>"_"</xsl:otherwise></xsl:choose>.join(prefix_strings)
			</xsl:when>
			<xsl:otherwise>initial_state("","<xsl:value-of select="xmml:Files/xmml:InitialFileName"/>",global_data,current_agent_data)</xsl:otherwise>
			</xsl:choose>
	else:
		print("No global or agent variations specified for experimentation\n")
	return global_data,agent_data
</xsl:for-each>
</xsl:if>
<xsl:if test="exp:Experimentation/xmml:InitialStates/@baseDirectory">
#Initial state file creation.
def initial_state(save_location,file_name,global_information,agent_information):
	if not os.path.exists(PROJECT_DIRECTORY+"<xsl:value-of select="exp:Experimentation/xmml:InitialStates/@baseDirectory"/>"):
		os.mkdir(PROJECT_DIRECTORY+"<xsl:value-of select="exp:Experimentation/xmml:InitialStates/@baseDirectory"/>")
	SAVE_DIRECTORY = PROJECT_DIRECTORY+"<xsl:value-of select="exp:Experimentation/xmml:InitialStates/@baseDirectory"/>"+"/"
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
			agent_id = 1
			agent_name = agent_information[i][0]
			for j in range(num_agents):
				initial_state_file.write("&lt;xagent&gt;\n")
				initial_state_file.write("&lt;name&gt;"+str(agent_name)+"&lt;/name&gt;\n")
				initial_state_file.write("&lt;id&gt;"+str(agent_id)+"&lt;/id&gt;\n")
				for k in agent_information[i]:
					if not (k[0]=="initial_population" or k==agent_name):
						if len(k[1])>1:
							if len(k[1])==4:
								random_method = getattr(random, k[1][2])
								initial_state_file.write("&lt;"+str(k[0])+"&gt;"+str(k[1][3](random_method(k[1][0],k[1][1])))+"&lt;/"+str(k[0])+"&gt;\n")
							elif len(k[1])==3:
								initial_state_file.write("&lt;"+str(k[0])+"&gt;"+str(k[1][2](random.uniform(k[1][0],k[1][1])))+"&lt;/"+str(k[0])+"&gt;\n")
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
</xsl:if>
<xsl:if test="exp:Experimentation/xmml:ExperimentSet">
#ExperimentSet
<xsl:for-each select="exp:Experimentation/xmml:ExperimentSet/xmml:Experiment">
############## <xsl:if test="xmml:ExperimentName"><xsl:value-of select="xmml:ExperimentName" /></xsl:if> ############
<xsl:if test="xmml:Configuration">
<xsl:if test="xmml:Configuration/xmml:ExperimentFunctions">
<xsl:for-each select="xmml:Configuration/xmml:ExperimentFunctions/xmml:Function">
def <xsl:value-of select="xmml:Name" />(<xsl:for-each select="xmml:Arguments/xmml:Argument"><xsl:value-of select="text()"/><xsl:if test="not(position()=last())">,</xsl:if></xsl:for-each>):
	<xsl:if test="xmml:GlobalVariables">global <xsl:for-each select="xmml:GlobalVariables/xmml:Global"><xsl:value-of select="text()"/><xsl:if test="not(position()=last())">,<xsl:text>&#x20;</xsl:text></xsl:if></xsl:for-each><xsl:text>&#xa;</xsl:text><xsl:text>&#x9;</xsl:text></xsl:if>
	<xsl:if test="xmml:Returns"><xsl:for-each select="xmml:Returns/xmml:Return"><xsl:value-of select="text()"/> = None<xsl:text>&#xa;</xsl:text><xsl:text>&#x9;</xsl:text></xsl:for-each></xsl:if>
	<xsl:if test="../../../xmml:InitialState/xmml:Generator">
	<xsl:if test="../../xmml:Repeats">
	##Run for desired number of repeats
	#current_initial_state = PROJECT_DIRECTORY+"<xsl:value-of select="../../../xmml:InitialState/xmml:Location"/>"+"/0.xml"
	#for i in range(<xsl:value-of select="../../xmml:Repeats"/>):
		#generate_initial_states_<xsl:value-of select="../../../xmml:InitialState/xmml:Generator"/>()
		#if OS_NAME=='nt':
			#executable = PROJECT_DIRECTORY+"<xsl:value-of select="../../../xmml:Model/xmml:ExecutableLocation" />/<xsl:value-of select="../../../xmml:Model/xmml:ModelName" />.exe"
			#simulation_command = executable+" "+current_initial_state+" <xsl:value-of select="../../xmml:Iterations"/>"
		#else:
			#executable = "./"+PROJECT_DIRECTORY+"<xsl:value-of select="../../../xmml:Model/xmml:ExecutableLocation" />/<xsl:value-of select="../../../xmml:Model/xmml:ModelName" />"
			#simulation_command = executable+" "+current_initial_state+" <xsl:value-of select="../../xmml:Iterations"/>"
		#print(simulation_command)
		##Run simulation
		#os.system(simulation_command)

		##Parse results
		#results_file = open(PROJECT_DIRECTORY+"/<xsl:value-of select="../../../xmml:SimulationOutput/xmml:Location"/>/<xsl:value-of select="../../../xmml:SimulationOutput/xmml:FileName"/>","r")
		#results = results_file.readlines()
		#results_file.close()
		#print(results)
	</xsl:if>
	</xsl:if>
	<xsl:if test="not(../../../xmml:InitialState/xmml:Generator)">
	##Model executable
	#executable = ""
	#simulation_command = ""
	#os_walk = os.walk("../../<xsl:value-of select="../../../../../xmml:InitialStates/xmml:InitialStateFile/xmml:Location"/>")
	#if len(os_walk)>1:
		#initial_states = [x[0] for x in os_walk][1:]
	#else:
		#initial_states = [x[0] for x in os_walk]
	#for i in initial_states:
		#current_initial_state = i+"/0.xml"
		#if OS_NAME=='nt':
			#executable = PROJECT_DIRECTORY+"<xsl:value-of select="../../../xmml:Model/xmml:ExecutableLocation" />/<xsl:value-of select="../../../xmml:Model/xmml:ModelName" />.exe"
			#simulation_command = executable+" "+current_initial_state+" <xsl:value-of select="../../xmml:Iterations"/>"
		#else:
			#executable = "./"+PROJECT_DIRECTORY+"<xsl:value-of select="../../../xmml:Model/xmml:ExecutableLocation" />/<xsl:value-of select="../../../xmml:Model/xmml:ModelName" />"
			#simulation_command = executable+" "+current_initial_state+" <xsl:value-of select="../../xmml:Iterations"/>"
		#print(simulation_command)
		
		
		##Run simulation
		#os.system(simulation_command)

		##Parse results
		#results_file = open(PROJECT_DIRECTORY+"/<xsl:value-of select="../../../xmml:SimulationOutput/xmml:Location"/>"+i+"/<xsl:value-of select="../../../xmml:SimulationOutput/xmml:FileName"/>","r")
		#results = results_file.readlines()
		#results_file.close()
		#print(results)
		</xsl:if>
	return <xsl:if test="xmml:Returns"><xsl:for-each select="xmml:Returns/xmml:Return"><xsl:value-of select="text()"/><xsl:if test="not(position()=last())">,<xsl:text>&#x20;</xsl:text></xsl:if></xsl:for-each></xsl:if><xsl:text>&#xa;</xsl:text>
</xsl:for-each>
</xsl:if>
<xsl:if test="not(xmml:Configuration/xmml:ExperimentFunctions) and xmml:Configuration"><xsl:if test="xmml:Configuration/xmml:Repeats">
#Run for desired number of repeats
#for i in range(<xsl:value-of select="xmml:Configuration/xmml:Repeats"/>):
	</xsl:if>#initial_state_creation_<xsl:value-of select="xmml:InitialState/xmml:Generator"/>(file_name,base_agent_information)
	#Run simulation
	#os.system(simulation_command)
	#Parse results
	#results_file = open("../../<xsl:value-of select="xmml:SimulationOutput/xmml:Location"/>"+INSERT_FILE_DIRECTORY_AND_NAME_HERE,"r")
	#results = results_file.readlines()
	#results_file.close()
</xsl:if>
</xsl:if>
</xsl:for-each>
</xsl:if>
def main():
	<xsl:if test="exp:Experimentation/xmml:InitialStates/xmml:InitialStateFunction">
	<xsl:for-each select="exp:Experimentation/xmml:InitialStates/xmml:InitialStateFunction">
	#Initial state generator function to be created by the user
	initial_state_generator_function_<xsl:value-of select="xmml:FunctionName"/>()
	</xsl:for-each>
	</xsl:if>
	<xsl:if test="exp:Experimentation/xmml:InitialStates/xmml:InitialStateGenerator">
	#Initial state creation function
	#initial_state(save_directory, initial_state_file_name, initial_state_global_data_list, initial_state_agent_data_list)

	#Generation functions (will automatically call initial state generation function)
	<xsl:for-each select="exp:Experimentation/xmml:InitialStates/xmml:InitialStateGenerator">
	generate_initial_states<xsl:if test="xmml:GeneratorName">_<xsl:value-of select="xmml:GeneratorName"/></xsl:if>()
	</xsl:for-each>
	</xsl:if>
	<xsl:if test="exp:Experimentation/xmml:ExperimentSet">
	#Experiment Set user defined functions
	<xsl:for-each select="exp:Experimentation/xmml:ExperimentSet/xmml:Experiment">
	<xsl:if test="xmml:Configuration/xmml:ExperimentVariables">
	<xsl:for-each select="xmml:Configuration/xmml:ExperimentVariables/xmml:Variable">
	<xsl:value-of select="xmml:Name" /> = <xsl:if test="not(xmml:Type='tuple')"><xsl:value-of select="xmml:Type" /></xsl:if>(<xsl:value-of select="xmml:Value" />)
	</xsl:for-each>
	</xsl:if>
	<xsl:if test="xmml:Configuration/xmml:ExperimentFunctions">
	<xsl:for-each select="xmml:Configuration/xmml:ExperimentFunctions/xmml:Function">
	#<xsl:if test="xmml:Returns"><xsl:for-each select="xmml:Returns/xmml:Return"><xsl:value-of select="text()"/><xsl:if test="not(position()=last())">,</xsl:if><xsl:text>&#x20;</xsl:text></xsl:for-each>= </xsl:if><xsl:value-of select="xmml:Name" />(<xsl:for-each select="xmml:Arguments/xmml:Argument"><xsl:value-of select="text()"/><xsl:if test="not(position()=last())">,</xsl:if></xsl:for-each>)
	</xsl:for-each>
	</xsl:if>
	</xsl:for-each>
	</xsl:if>
	return

if __name__ == "__main__":
	main()
</xsl:template>
</xsl:stylesheet>