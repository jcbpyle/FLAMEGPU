
/*
 * Copyright 2011 University of Sheffield.
 * Author: Dr Paul Richmond 
 * Contact: p.richmond@sheffield.ac.uk (http://www.paulrichmond.staff.shef.ac.uk)
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

#ifndef _FLAMEGPU_FUNCTIONS
#define _FLAMEGPU_FUNCTIONS

#include <header.h>

FILE *results_file;

 /**
 * Automatically generated using logging_and_functions.xslt
 
 */
 __FLAME_GPU_INIT_FUNC__ void  log_all_pred_life() {
    char results_file[1024];
    sprintf(results_file, "%s%s", getOutputDir(), "log.csv");
    simulation_results = fopen(results_file, "a");
    
    xmachine_memory_pred_list* host_agent_list = get_host_pred_default2_agents();
    int agent_count = get_agent_pred_default2_count();
    fprintf(simulation_results, "%s,[", "pls");
    for (int i=0; i<agent_count; i++){
        fprintf(simulation_results, "(%d, %f),", host_agent_list->id[i], host_agent_list->life[i]);
    }
    fprintf(simulation_results, "]");
    fclose(simulation_results);
 }

 __FLAME_GPU_EXIT_FUNC__ void  log_all_final_pred_life() { 
    char results_file[1024];
    sprintf(results_file, "%s%s", getOutputDir(), "log.csv");
    simulation_results = fopen(results_file, "a");
    
    cudaError_t cudaStatus;
    xmachine_memory_pred_list* host_agent_list = get_host_pred_default2_agents();
    xmachine_memory_pred_list* device_agent_list = get_device_pred_default2_agents();
    int agent_count = get_agent_pred_default2_count();
    cudaStatus = cudaMemcpy(host_agent_list, device_agent_list, sizeof(xmachine_memory_pred_list), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Error Copying pred Agent default2 State Memory from GPU: %s\n", cudaGetErrorString(cudaStatus));
        exit(cudaStatus);
    }
    fprintf(simulation_results, "%s,[", "ple");
    for (int i=0; i<agent_count; i++){
        fprintf(simulation_results, "(%d, %f),", host_agent_list->id[i], host_agent_list->life[i]);
    }
    fprintf(simulation_results, "]");
    fclose(simulation_results);
 }

 /**
 * Automatically generated using logging_and_functions.xslt
 *
 */
 __FLAME_GPU_INIT_FUNC__ void  global_log_prey() {
    char results_file[1024];
    sprintf(results_file, "%s%s", getOutputDir(), "log.csv");
    simulation_results = fopen(results_file, "a");
    int log_variable = get_agent_prey_default1_count();
    fprintf(simulation_results, "%s,%d,", "num_prey", log_variable);
    fclose(simulation_results);
 }

 /**
 * Automatically generated using logging_and_functions.xslt
 *
 */
 __FLAME_GPU_INIT_FUNC__ void  global_log_predator() {
    char results_file[1024];
    sprintf(results_file, "%s%s", getOutputDir(), "log.csv");
    simulation_results = fopen(results_file, "a");
    int log_variable = get_agent_predator_default2_count();
    fprintf(simulation_results, "%s,%d,", "num_pred", log_variable);
    fclose(simulation_results);
 }

 /**
 * Automatically generated using logging_and_functions.xslt
 *
 */
 __FLAME_GPU_INIT_FUNC__ void  global_log_grass() {
    char results_file[1024];
    sprintf(results_file, "%s%s", getOutputDir(), "log.csv");
    simulation_results = fopen(results_file, "a");
    int log_variable = get_agent_grass_default3_count();
    fprintf(simulation_results, "%s,%d,", "num_grass", log_variable);
    fclose(simulation_results);
 }

 /**
 * Automatically generated using logging_and_functions.xslt
 
 */
 __FLAME_GPU_INIT_FUNC__ void  global_log_prey_reproduction() {
    char results_file[1024];
    sprintf(results_file, "%s%s", getOutputDir(), "log.csv");
    simulation_results = fopen(results_file, "a");
    float log_variable = *get_REPRODUCE_PREY_PROB();
    fprintf(simulation_results, "%s,%f,", "prey_reproduction", log_variable);
    fclose(simulation_results);
 }

 /**
 * Automatically generated using logging_and_functions.xslt
 
 */
 __FLAME_GPU_INIT_FUNC__ void  global_log_pred_reproduction() {
    char results_file[1024];
    sprintf(results_file, "%s%s", getOutputDir(), "log.csv");
    simulation_results = fopen(results_file, "a");
    float log_variable = *get_REPRODUCE_PREDATOR_PROB();
    fprintf(simulation_results, "%s,%f,", "pred_reproduction", log_variable);
    fclose(simulation_results);
 }

 /**
 * Automatically generated using logging_and_functions.xslt
 
 */
 __FLAME_GPU_INIT_FUNC__ void  global_log_prey_energy_gain() {
    char results_file[1024];
    sprintf(results_file, "%s%s", getOutputDir(), "log.csv");
    simulation_results = fopen(results_file, "a");
    int log_variable = *get_GAIN_FROM_FOOD_PREY();
    fprintf(simulation_results, "%s,%d,", "prey_energy", log_variable);
    fclose(simulation_results);
 }

 /**
 * Automatically generated using logging_and_functions.xslt
 
 */
 __FLAME_GPU_INIT_FUNC__ void  global_log_pred_energy_gain() {
    char results_file[1024];
    sprintf(results_file, "%s%s", getOutputDir(), "log.csv");
    simulation_results = fopen(results_file, "a");
    int log_variable = *get_GAIN_FROM_FOOD_PREDATOR();
    fprintf(simulation_results, "%s,%d,", "pred_energy", log_variable);
    fclose(simulation_results);
 }

 /**
 * Automatically generated using logging_and_functions.xslt
 
 */
 __FLAME_GPU_INIT_FUNC__ void  global_log_grass_regrowth() {
    char results_file[1024];
    sprintf(results_file, "%s%s", getOutputDir(), "log.csv");
    simulation_results = fopen(results_file, "a");
    int log_variable = *get_GRASS_REGROW_CYCLES();
    fprintf(simulation_results, "%s,%d,", "grass_regrowth", log_variable);
    fclose(simulation_results);
 }

 __FLAME_GPU_EXIT_FUNC__ void  primary_fitness_output() { 
    char results_file[1024];
    sprintf(results_file, "%s%s", getOutputDir(), "log.csv");
    simulation_results = fopen(results_file, "a");
    
 }

 __FLAME_GPU_EXIT_FUNC__ void  secondary_fitness_output() { 
    char results_file[1024];
    sprintf(results_file, "%s%s", getOutputDir(), "log.csv");
    simulation_results = fopen(results_file, "a");
    
 }

 __FLAME_GPU_EXIT_FUNC__ void  tertiary_fitness_output() { 
    char results_file[1024];
    sprintf(results_file, "%s%s", getOutputDir(), "log.csv");
    simulation_results = fopen(results_file, "a");
    
 }

  


#endif //_FLAMEGPU_FUNCTIONS
