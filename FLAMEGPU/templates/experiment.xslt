<?xml version="1.0" encoding="utf-8"?>
<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
                xmlns:xmml="http://www.dcs.shef.ac.uk/~paul/XMML"
                xmlns:gpu="http://www.dcs.shef.ac.uk/~paul/XMMLGPU"
                xmlns:experimentation="https://jcbpyle.github.io/website">
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

import os
import sys
import threading
import queue
import random
import datetime
import pycuda.driver as cuda
import pycuda.autoinit

BASE_DIRECTORY = os.getcwd()
<xsl:if test="experimentation:Experimentation/experimentation:Experiment/experimentation:InitialState/experimentation:Location">
#Initial state file creation.
def initial_state_creation(file_name,agent_information):
	SAVE_DIRECTORY = BASE_DIRECTORY+"../../<xsl:value-of select="experimentation:Experimentation/experimentation:Experiment/experimentation:InitialState/experimentation:Location"/>"+"/"
	SAVE_DIRECTORY = BASE_DIRECTORY+"/"
	initial_state_file = open(SAVE_DIRECTORY+str(file_name)+".xml","w")
	initial_state_file.write("&lt;states&gt;\n&lt;itno&gt;0&lt;/itno&gt;\n&lt;environment&gt;\n")
	<xsl:if test="experimentation:Experimentation/experimentation:Experiment/experimentation:Globals">
	<xsl:for-each select="experimentation:Experimentation/experimentation:Experiment/experimentation:Globals/experimentation:global">
	<xsl:if test="experimentation:value/experimentation:fixed_value">initial_state_file.write("&lt;<xsl:value-of select="experimentation:name"/>&gt;"+str(<xsl:value-of select="experimentation:value"/>)+"&lt;/<xsl:value-of select="experimentation:name"/>&gt;\n")
	</xsl:if>
	<xsl:if test="experimentation:value/experimentation:range">initial_state_file.write("&lt;<xsl:value-of select="experimentation:name"/>&gt;"+str(<xsl:if test="experimentation:value/experimentation:type='int'">int(</xsl:if>random.uniform(<xsl:value-of select="experimentation:value/experimentation:range/experimentation:min"/>,<xsl:value-of select="experimentation:value/experimentation:range/experimentation:max"/><xsl:if test="experimentation:value/experimentation:type='int'">)</xsl:if>))+"&lt;/<xsl:value-of select="experimentation:name"/>&gt;\n")
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

base_agent_information = [<xsl:for-each select="experimentation:Experimentation/experimentation:Experiment/experimentation:Populations/experimentation:population">
["<xsl:value-of select="experimentation:agent"/>",["initial_population",<xsl:if test="experimentation:InitialPopulationCount/experimentation:fixed_value"><xsl:value-of select="experimentation:InitialPopulationCount/experimentation:fixed_value"/>,<xsl:value-of select="experimentation:InitialPopulationCount/experimentation:fixed_value"/></xsl:if><xsl:if test="experimentation:InitialPopulationCount/experimentation:range"><xsl:value-of select="experimentation:InitialPopulationCount/experimentation:range/experimentation:min"/>,<xsl:value-of select="experimentation:InitialPopulationCount/experimentation:range/experimentation:max"/></xsl:if>],<xsl:for-each select="experimentation:Variables/experimentation:variable">["<xsl:value-of select="experimentation:name"/>",<xsl:if test="experimentation:value/experimentation:fixed_value"><xsl:value-of select="experimentation:value/experimentation:fixed_value"/>,<xsl:value-of select="experimentation:value/experimentation:fixed_value"/></xsl:if><xsl:if test="experimentation:value/experimentation:range"><xsl:value-of select="experimentation:value/experimentation:range/experimentation:min"/>,<xsl:value-of select="experimentation:value/experimentation:range/experimentation:max"/></xsl:if>],</xsl:for-each>],</xsl:for-each>]

initial_state_creation("<xsl:value-of select="experimentation:Experimentation/experimentation:Experiment/experimentation:InitialState/experimentation:DefaultName"/>",base_agent_information)
<!-- def create_x_initial_states(x):
	for i in range(x):
		new_agent_information = []
		new_file_name = ""
		initial_state_creation()
	return
initial_state_creation();

<xsl:if test="gpu:experimentation/xmml:InitialState/xmml:Populations">
	<xsl:for-each select="gpu:experimentation/xmml:InitialState/xmml:Populations/xmml:population">
	num_agents_<xsl:value-of select="xmml:agent"/> = <xsl:if test="xmml:InitialPopulationCount/xmml:fixed_value"><xsl:value-of select="xmml:InitialPopulationCount/xmml:fixed_value"/></xsl:if><xsl:if test="xmml:InitialPopulationCount/xmml:range">int(random.uniform(<xsl:value-of select="xmml:InitialPopulationCount/xmml:range/xmml:min"/>,<xsl:value-of select="xmml:InitialPopulationCount/xmml:range/xmml:max"/>))
	</xsl:if>
	<xsl:value-of select="xmml:agent"/>_id_count = 0
	for i in range(num_agents_<xsl:value-of select="xmml:agent"/>):
		initial_state_file.write("&lt;xagent&gt;\n")
		initial_state_file.write("&lt;name&gt;"+"<xsl:value-of select="xmml:agent"/>"+"&lt;/name&gt;\n")
		initial_state_file.write("&lt;id&gt;"+str(<xsl:value-of select="xmml:agent"/>_id_count)+"&lt;/id&gt;\n")
		<xsl:for-each select="xmml:Variables/xmml:variable">
		<xsl:if test="xmml:value/xmml:fixed_value">initial_state_file.write("&lt;<xsl:value-of select="xmml:name"/>&gt;"+str(<xsl:value-of select="xmml:value/xmml:fixed_value"/>)+"&lt;/<xsl:value-of select="xmml:name"/>&gt;\n")
		</xsl:if>
		<xsl:if test="xmml:value/xmml:range">initial_state_file.write("&lt;<xsl:value-of select="xmml:name"/>&gt;"+str(<xsl:if test="xmml:value/xmml:type='int'">int(</xsl:if>random.uniform(<xsl:value-of select="xmml:value/xmml:range/xmml:min"/>,<xsl:value-of select="xmml:value/xmml:range/xmml:max"/><xsl:if test="xmml:value/xmml:type='int'">)</xsl:if>))+"&lt;/<xsl:value-of select="xmml:name"/>&gt;\n")
		</xsl:if>
		</xsl:for-each>
		initial_state_file.write("&lt;/xagent&gt;\n")
		<xsl:value-of select="xmml:agent"/>_id_count += 1
	</xsl:for-each>
	</xsl:if>
	initial_state_file.write("&lt;/states&gt;\n")
	initial_state_file.close()
	return -->
</xsl:if> <!--Initial state creation-->


</xsl:template>
</xsl:stylesheet>