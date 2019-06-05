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
import pycuda.driver as cuda
import pycuda.autoinit

BASE_DIRECTORY = os.getcwd()
GPUS_AVAILABLE = cuda.Device(0).count()
<xsl:if test="exp:Experimentation/exp:InitialStates">
#InitialStates
<xsl:for-each select="exp:Experimentation/exp:InitialStates/exp:InitialState">
#Initial state file creation<xsl:if test="exp:ConfigurationName"> for experiment <xsl:value-of select="exp:ConfigurationName"/></xsl:if>.
def initial_state_creation<xsl:if test="exp:ConfigurationName">_<xsl:value-of select="exp:ConfigurationName"/></xsl:if>(file_name,agent_information):
	SAVE_DIRECTORY = BASE_DIRECTORY+"../../<xsl:value-of select="exp:Location"/>"+"/"
	SAVE_DIRECTORY = BASE_DIRECTORY+"/"
	initial_state_file = open(SAVE_DIRECTORY+str(file_name)+".xml","w")
	initial_state_file.write("&lt;states&gt;\n&lt;itno&gt;0&lt;/itno&gt;\n&lt;environment&gt;\n")
	<xsl:if test="exp:Globals">
	<xsl:for-each select="exp:Globals/exp:global">
	<xsl:if test="exp:value/exp:fixed_value">initial_state_file.write("&lt;<xsl:value-of select="exp:name"/>&gt;"+str(<xsl:value-of select="exp:value"/>)+"&lt;/<xsl:value-of select="exp:name"/>&gt;\n")
	</xsl:if>
	<xsl:if test="exp:value/exp:range">initial_state_file.write("&lt;<xsl:value-of select="exp:name"/>&gt;"+str(<xsl:if test="exp:value/exp:type='int'">int(</xsl:if>random.uniform(<xsl:value-of select="exp:value/exp:range/exp:min"/>,<xsl:value-of select="exp:value/exp:range/exp:max"/><xsl:if test="exp:value/exp:type='int'">)</xsl:if>))+"&lt;/<xsl:value-of select="exp:name"/>&gt;\n")
	</xsl:if>
	</xsl:for-each>
	</xsl:if>
	initial_state_file.write("&lt;/environment&gt;\n")
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

#Agent data stored in list of lists
base_agent_information = [<xsl:if test="exp:Populations"><xsl:for-each select="exp:Populations/exp:population">
["<xsl:value-of select="exp:agent"/>",["initial_population",<xsl:if test="exp:InitialPopulationCount/exp:fixed_value"><xsl:value-of select="exp:InitialPopulationCount/exp:fixed_value"/>,<xsl:value-of select="exp:InitialPopulationCount/exp:fixed_value"/></xsl:if><xsl:if test="exp:InitialPopulationCount/exp:range"><xsl:value-of select="exp:InitialPopulationCount/exp:range/exp:min"/>,<xsl:value-of select="exp:InitialPopulationCount/exp:range/exp:max"/></xsl:if>],<xsl:for-each select="exp:Variables/exp:variable">["<xsl:value-of select="exp:name"/>",<xsl:if test="exp:value/exp:fixed_value"><xsl:value-of select="exp:value/exp:fixed_value"/>,<xsl:value-of select="exp:value/exp:fixed_value"/></xsl:if><xsl:if test="exp:value/exp:range"><xsl:value-of select="exp:value/exp:range/exp:min"/>,<xsl:value-of select="exp:value/exp:range/exp:max"/></xsl:if>],</xsl:for-each>],</xsl:for-each></xsl:if>]

#Create initial state
#initial_state_creation<xsl:if test="exp:ConfigurationName">_<xsl:value-of select="exp:ConfigurationName"/></xsl:if>("<xsl:value-of select="exp:FileName"/>",base_agent_information)
</xsl:for-each>
</xsl:if>


<xsl:if test="exp:Experimentation/exp:ExperimentSet">
#ExperimentSet
<xsl:for-each select="exp:Experimentation/exp:ExperimentSet/exp:Experiment">
############## <xsl:if test="exp:ExperimentName"><xsl:value-of select="exp:ExperimentName" /></xsl:if> ############
#Model executable
#winexe = ""+str(<xsl:value-of select="exp:Model/exp:ExecutableLocation" />)+str(<xsl:value-of select="exp:Model/exp:ModelName" />)+".exe"
#unexe = "./"+str(<xsl:value-of select="exp:Model/exp:ExecutableLocation" />)+str(<xsl:value-of select="exp:Model/exp:ModelName" />)
#Initial state creator
#initial_state_creation_<xsl:value-of select="exp:InitialStateGenerator"/>(file_name,base_agent_information)
<xsl:if test="exp:Configuration">
<xsl:if test="exp:Configuration/exp:ExperimentVariables">
<xsl:for-each select="exp:Configuration/exp:ExperimentVariables/exp:Variable">
<xsl:value-of select="exp:Name" /> = <xsl:value-of select="exp:Type" />(<xsl:value-of select="exp:Value" />)
</xsl:for-each>
</xsl:if>

<xsl:if test="exp:Configuration/exp:ExperimentFunctions">
<xsl:for-each select="exp:Configuration/exp:ExperimentFunctions/exp:Function">
def <xsl:value-of select="exp:Name" />(<xsl:for-each select="exp:Arguments/exp:Argument"><xsl:value-of select="text()"/>,</xsl:for-each>placeholder=None):
	<xsl:if test="exp:GlobalVariables">global <xsl:for-each select="exp:GlobalVariables/exp:Global"><xsl:value-of select="text()"/>&#160;</xsl:for-each>&#xa;</xsl:if>
	<xsl:if test="exp:Returns"><xsl:for-each select="exp:Returns/exp:Return"><xsl:value-of select="text()"/> = None&#xa;</xsl:for-each></xsl:if>
	return <xsl:if test="exp:Returns"><xsl:for-each select="exp:Returns/exp:Return"><xsl:value-of select="text()"/>&#160;</xsl:for-each></xsl:if>
&#xa;	
</xsl:for-each>
</xsl:if>
</xsl:if>
</xsl:for-each>
</xsl:if>

</xsl:template>
</xsl:stylesheet>