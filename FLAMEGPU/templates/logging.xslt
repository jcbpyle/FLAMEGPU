<?xml version="1.0" encoding="utf-8"?>
<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
                xmlns:xmml="http://www.dcs.shef.ac.uk/~paul/XMML"
                xmlns:gpu="http://www.dcs.shef.ac.uk/~paul/XMMLGPU"
                xmlns:experimentation="https://jcbpyle.github.io/website">
<xsl:output method="text" version="1.0" encoding="UTF-8" indent="yes" />
<xsl:include href = "./_common_templates.xslt" />
<!--Main template-->
<xsl:template match="/">
<xsl:call-template name="copyrightNotice"></xsl:call-template>
/*
 * Copyright 2019 University of Sheffield.
 * Author: James Pyle
 * Contact: jcbpyle1@sheffield.ac.uk (https://jcbpyle.github.io/)
 *
 * University of Sheffield retain all intellectual property and 
 * proprietary rights in and to this software and related documentation. 
 * Any use, reproduction, disclosure, or distribution of this software 
 * and related documentation without an express license agreement from
 * University of Sheffield is strictly prohibited.
 *
 * For terms of licence agreement please attached licence or view licence 
 * on www.flamegpu.com website.
 * 
 */

#ifndef _FLAMEGPU_LOGGING_FUNCTIONS
#define _FLAMEGPU_LOGGING_FUNCTIONS

<xsl:if test="gpu:xmodel/xmml:logging">
FILE *results_file;
<xsl:for-each select="gpu:xmodel/xmml:logging/xmml:log">

<xsl:if test="xmml:InitLog">
 /**
 * Automatically generated using logging.xslt
 <xsl:if test="xmml:InitLog/xmml:LogAgentTypeCount">*</xsl:if>
 */
 __FLAME_GPU_INIT_FUNC__ void  <xsl:value-of select="xmml:InitLog/xmml:name"/>() {
    char results_file[1024];
    sprintf(results_file, "%s%s", getOutputDir(), "<xsl:value-of select="xmml:InitLog/xmml:file"/>");
    simulation_results = fopen(results_file, "a");
    <xsl:if test="xmml:InitLog/xmml:LogGlobal"><xsl:value-of select="xmml:InitLog/xmml:LogGlobal/xmml:variable_type"/> log_variable = *get_<xsl:value-of select="xmml:InitLog/xmml:LogGlobal/xmml:global"/>();
    fprintf(simulation_results, "<xsl:if test="xmml:InitLog/xmml:identifier">%s,</xsl:if><xsl:if test="xmml:InitLog/xmml:LogGlobal/xmml:variable_type='float'">%f,</xsl:if><xsl:if test="xmml:InitLog/xmml:LogGlobal/xmml:variable_type='int'">%d,</xsl:if><xsl:if test="xmml:InitLog/xmml:LogGlobal/xmml:variable_type='char'">%s,</xsl:if>", <xsl:if test="xmml:InitLog/xmml:identifier">"<xsl:value-of select="xmml:InitLog/xmml:identifier"/>", </xsl:if>log_variable);
    fclose(simulation_results);</xsl:if>
    <xsl:if test="xmml:InitLog/xmml:LogAgentTypeCount">int log_variable = get_agent_<xsl:value-of select="xmml:InitLog/xmml:LogAgentTypeCount/xmml:agent"/>_<xsl:value-of select="xmml:InitLog/xmml:LogAgentTypeCount/xmml:state"/>_count();
    fprintf(simulation_results, "<xsl:if test="xmml:InitLog/xmml:identifier">%s,</xsl:if>%d,", <xsl:if test="xmml:InitLog/xmml:identifier">"<xsl:value-of select="xmml:InitLog/xmml:identifier"/>", </xsl:if>log_variable);
    fclose(simulation_results);</xsl:if>
    <xsl:if test="xmml:InitLog/xmml:LogFunction"><xsl:value-of select="xmml:InitLog/xmml:LogFunction/xmml:return_type"/> log_variable = <xsl:value-of select="xmml:InitLog/xmml:LogFunction/xmml:function"/>();
    fprintf(simulation_results, "<xsl:if test="xmml:InitLog/xmml:identifier">%s,</xsl:if><xsl:if test="xmml:InitLog/xmml:LogFunction/xmml:return_type='float'">%f,</xsl:if><xsl:if test="xmml:InitLog/xmml:LogFunction/xmml:return_type='int'">%d,</xsl:if><xsl:if test="xmml:InitLog/xmml:LogFunction/xmml:return_type='char'">%s,</xsl:if>", <xsl:if test="xmml:InitLog/xmml:identifier">"<xsl:value-of select="xmml:InitLog/xmml:identifier"/>", </xsl:if>log_variable);
    fclose(simulation_results);</xsl:if>
 }
</xsl:if>
<xsl:if test="xmml:ExitLog">
 __FLAME_GPU_EXIT_FUNC__ void  <xsl:value-of select="xmml:ExitLog/xmml:name"/>() { 
    char results_file[1024];
    sprintf(results_file, "%s%s", getOutputDir(), "<xsl:value-of select="xmml:ExitLog/xmml:file"/>");
    simulation_results = fopen(results_file, "a");
    <xsl:if test="xmml:ExitLog/xmml:LogGlobal"><xsl:value-of select="xmml:ExitLog/xmml:LogGlobal/xmml:variable_type"/> log_variable = *get_<xsl:value-of select="xmml:ExitLog/xmml:LogGlobal/xmml:global"/>();
    fprintf(simulation_results, "<xsl:if test="xmml:ExitLog/xmml:identifier">%s,</xsl:if><xsl:if test="xmml:ExitLog/xmml:LogGlobal/xmml:variable_type='float'">%f,</xsl:if><xsl:if test="xmml:ExitLog/xmml:LogGlobal/xmml:variable_type='int'">%d,</xsl:if><xsl:if test="xmml:ExitLog/xmml:LogGlobal/xmml:variable_type='char'">%s,</xsl:if>", <xsl:if test="xmml:ExitLog/xmml:identifier">"<xsl:value-of select="xmml:ExitLog/xmml:identifier"/>", </xsl:if>log_variable);
    fclose(simulation_results);</xsl:if>
    <xsl:if test="xmml:ExitLog/xmml:LogAgentTypeCount">int log_variable = get_agent_<xsl:value-of select="xmml:ExitLog/xmml:LogAgentTypeCount/xmml:agent"/>_<xsl:value-of select="xmml:ExitLog/xmml:LogAgentTypeCount/xmml:state"/>_count();
    fprintf(simulation_results, "<xsl:if test="xmml:ExitLog/xmml:identifier">%s,</xsl:if>%d,", <xsl:if test="xmml:ExitLog/xmml:identifier">"<xsl:value-of select="xmml:ExitLog/xmml:identifier"/>", </xsl:if>log_variable);
    fclose(simulation_results);</xsl:if>
    <xsl:if test="xmml:ExitLog/xmml:LogFunction"><xsl:value-of select="xmml:ExitLog/xmml:LogFunction/xmml:return_type"/> log_variable = <xsl:value-of select="xmml:ExitLog/xmml:LogFunction/xmml:function"/>();
    fprintf(simulation_results, "<xsl:if test="xmml:ExitLog/xmml:identifier">%s,</xsl:if><xsl:if test="xmml:ExitLog/xmml:LogFunction/xmml:return_type='float'">%f,</xsl:if><xsl:if test="xmml:ExitLog/xmml:LogFunction/xmml:return_type='int'">%d,</xsl:if><xsl:if test="xmml:ExitLog/xmml:LogFunction/xmml:return_type='char'">%s,</xsl:if>", <xsl:if test="xmml:ExitLog/xmml:identifier">"<xsl:value-of select="xmml:ExitLog/xmml:identifier"/>", </xsl:if>log_variable);
    fclose(simulation_results);</xsl:if>
 }
</xsl:if>
<xsl:if test="xmml:StepLog">
 __FLAME_GPU_STEP_FUNC__ void  <xsl:value-of select="xmml:StepLog/xmml:name"/>() { 
    char results_file[1024];
    sprintf(results_file, "%s%s", getOutputDir(), "<xsl:value-of select="xmml:StepLog/xmml:file"/>");
    simulation_results = fopen(results_file, "a");
    <xsl:if test="xmml:StepLog/xmml:LogGlobal"><xsl:value-of select="xmml:StepLog/xmml:LogGlobal/xmml:variable_type"/> log_variable = *get_<xsl:value-of select="xmml:StepLog/xmml:LogGlobal/xmml:global"/>();
    fprintf(simulation_results, "<xsl:if test="xmml:StepLog/xmml:identifier">%s,</xsl:if><xsl:if test="xmml:StepLog/xmml:LogGlobal/xmml:variable_type='float'">%f,</xsl:if><xsl:if test="xmml:StepLog/xmml:LogGlobal/xmml:variable_type='int'">%d,</xsl:if><xsl:if test="xmml:StepLog/xmml:LogGlobal/xmml:variable_type='char'">%s,</xsl:if>", <xsl:if test="xmml:StepLog/xmml:identifier">"<xsl:value-of select="xmml:StepLog/xmml:identifier"/>", </xsl:if>log_variable);
    fclose(simulation_results);</xsl:if>
    <xsl:if test="xmml:StepLog/xmml:LogAgentTypeCount">int log_variable = get_agent_<xsl:value-of select="xmml:StepLog/xmml:LogAgentTypeCount/xmml:agent"/>_<xsl:value-of select="xmml:StepLog/xmml:LogAgentTypeCount/xmml:state"/>_count();
    fprintf(simulation_results, "<xsl:if test="xmml:StepLog/xmml:identifier">%s,</xsl:if>%d,", <xsl:if test="xmml:StepLog/xmml:identifier">"<xsl:value-of select="xmml:StepLog/xmml:identifier"/>", </xsl:if>log_variable);
    fclose(simulation_results);</xsl:if>
    <xsl:if test="xmml:StepLog/xmml:LogFunction"><xsl:value-of select="xmml:StepLog/xmml:LogFunction/xmml:return_type"/> log_variable = <xsl:value-of select="xmml:StepLog/xmml:LogFunction/xmml:function"/>();
    fprintf(simulation_results, "<xsl:if test="xmml:StepLog/xmml:identifier">%s,</xsl:if><xsl:if test="xmml:StepLog/xmml:LogFunction/xmml:return_type='float'">%f,</xsl:if><xsl:if test="xmml:StepLog/xmml:LogFunction/xmml:return_type='int'">%d,</xsl:if><xsl:if test="xmml:StepLog/xmml:LogFunction/xmml:return_type='char'">%s,</xsl:if>", <xsl:if test="xmml:StepLog/xmml:identifier">"<xsl:value-of select="xmml:StepLog/xmml:identifier"/>", </xsl:if>log_variable);
    fclose(simulation_results);</xsl:if>
 }
</xsl:if>
</xsl:for-each>
</xsl:if>
#endif //_FLAMEGPU_LOGGING_FUNCTIONS
</xsl:template>
</xsl:stylesheet>