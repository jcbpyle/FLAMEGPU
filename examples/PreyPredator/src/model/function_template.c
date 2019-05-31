
/*
 * FLAME GPU v 1.5.X for CUDA 9
 * Copyright University of Sheffield.
 * Original Author: Dr Paul Richmond (user contributions tracked on https://github.com/FLAMEGPU/FLAMEGPU)
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


/**
 * generate_files FLAMEGPU Init function
 * Automatically generated using functions.xslt
 */
__FLAME_GPU_INIT_FUNC__ void generate_files(){

}

/**
 * log_all_pred_life FLAMEGPU Init function
 * Automatically generated using functions.xslt
 */
__FLAME_GPU_INIT_FUNC__ void log_all_pred_life(){

}

/**
 * global_log_prey FLAMEGPU Init function
 * Automatically generated using functions.xslt
 */
__FLAME_GPU_INIT_FUNC__ void global_log_prey(){

}

/**
 * global_log_predator FLAMEGPU Init function
 * Automatically generated using functions.xslt
 */
__FLAME_GPU_INIT_FUNC__ void global_log_predator(){

}

/**
 * global_log_grass FLAMEGPU Init function
 * Automatically generated using functions.xslt
 */
__FLAME_GPU_INIT_FUNC__ void global_log_grass(){

}

/**
 * global_log_prey_reproduction FLAMEGPU Init function
 * Automatically generated using functions.xslt
 */
__FLAME_GPU_INIT_FUNC__ void global_log_prey_reproduction(){

}

/**
 * global_log_pred_reproduction FLAMEGPU Init function
 * Automatically generated using functions.xslt
 */
__FLAME_GPU_INIT_FUNC__ void global_log_pred_reproduction(){

}

/**
 * global_log_prey_energy_gain FLAMEGPU Init function
 * Automatically generated using functions.xslt
 */
__FLAME_GPU_INIT_FUNC__ void global_log_prey_energy_gain(){

}

/**
 * global_log_pred_energy_gain FLAMEGPU Init function
 * Automatically generated using functions.xslt
 */
__FLAME_GPU_INIT_FUNC__ void global_log_pred_energy_gain(){

}

/**
 * global_log_grass_regrowth FLAMEGPU Init function
 * Automatically generated using functions.xslt
 */
__FLAME_GPU_INIT_FUNC__ void global_log_grass_regrowth(){

}

/**
 * outputToLogFile FLAMEGPU Step function
 * Automatically generated using functions.xslt
 */
__FLAME_GPU_STEP_FUNC__ void outputToLogFile(){

}

/**
 * primary_fitness_output FLAMEGPU Exit function
 * Automatically generated using functions.xslt
 */
__FLAME_GPU_EXIT_FUNC__ void primary_fitness_output(){

}

/**
 * secondary_fitness_output FLAMEGPU Exit function
 * Automatically generated using functions.xslt
 */
__FLAME_GPU_EXIT_FUNC__ void secondary_fitness_output(){

}

/**
 * tertiary_fitness_output FLAMEGPU Exit function
 * Automatically generated using functions.xslt
 */
__FLAME_GPU_EXIT_FUNC__ void tertiary_fitness_output(){

}

/**
 * log_pred_final_life FLAMEGPU Exit function
 * Automatically generated using functions.xslt
 */
__FLAME_GPU_EXIT_FUNC__ void log_pred_final_life(){

}

/**
 * prey_output_location FLAMEGPU Agent Function
 * Automatically generated using functions.xslt
 * @param agent Pointer to an agent structure of type xmachine_memory_prey. This represents a single agent instance and can be modified directly.
 * @param prey_location_messages Pointer to output message list of type xmachine_message_prey_location_list. Must be passed as an argument to the add_prey_location_message function.
 */
__FLAME_GPU_FUNC__ int prey_output_location(xmachine_memory_prey* agent, xmachine_message_prey_location_list* prey_location_messages){
    
    /* 
    //Template for message output function
    int id = 0;
    float x = 0;
    float y = 0;
    
    add_prey_location_message(prey_location_messages, id, x, y);
    */     
    
    return 0;
}

/**
 * prey_move FLAMEGPU Agent Function
 * Automatically generated using functions.xslt
 * @param agent Pointer to an agent structure of type xmachine_memory_prey. This represents a single agent instance and can be modified directly.
 
 */
__FLAME_GPU_FUNC__ int prey_move(xmachine_memory_prey* agent){
    
    return 0;
}

/**
 * prey_eaten FLAMEGPU Agent Function
 * Automatically generated using functions.xslt
 * @param agent Pointer to an agent structure of type xmachine_memory_prey. This represents a single agent instance and can be modified directly.
 * @param pred_location_messages  pred_location_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_pred_location_message and get_next_pred_location_message functions.* @param prey_eaten_messages Pointer to output message list of type xmachine_message_prey_eaten_list. Must be passed as an argument to the add_prey_eaten_message function.
 */
__FLAME_GPU_FUNC__ int prey_eaten(xmachine_memory_prey* agent, xmachine_message_pred_location_list* pred_location_messages, xmachine_message_prey_eaten_list* prey_eaten_messages){
    
    /*
    //Template for input message iteration
    xmachine_message_pred_location* current_message = get_first_pred_location_message(pred_location_messages);
    while (current_message)
    {
        //INSERT MESSAGE PROCESSING CODE HERE
        
        current_message = get_next_pred_location_message(current_message, pred_location_messages);
    }
    */
    
    /* 
    //Template for message output function
    int pred_id = 0;
    
    add_prey_eaten_message(prey_eaten_messages, pred_id);
    */     
    
    return 0;
}

/**
 * prey_eat_or_starve FLAMEGPU Agent Function
 * Automatically generated using functions.xslt
 * @param agent Pointer to an agent structure of type xmachine_memory_prey. This represents a single agent instance and can be modified directly.
 * @param grass_eaten_messages  grass_eaten_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_grass_eaten_message and get_next_grass_eaten_message functions.
 */
__FLAME_GPU_FUNC__ int prey_eat_or_starve(xmachine_memory_prey* agent, xmachine_message_grass_eaten_list* grass_eaten_messages){
    
    /*
    //Template for input message iteration
    xmachine_message_grass_eaten* current_message = get_first_grass_eaten_message(grass_eaten_messages);
    while (current_message)
    {
        //INSERT MESSAGE PROCESSING CODE HERE
        
        current_message = get_next_grass_eaten_message(current_message, grass_eaten_messages);
    }
    */
    
    return 0;
}

/**
 * prey_reproduction FLAMEGPU Agent Function
 * Automatically generated using functions.xslt
 * @param agent Pointer to an agent structure of type xmachine_memory_prey. This represents a single agent instance and can be modified directly.
 * @param prey_agents Pointer to agent list of type xmachine_memory_prey_list. This must be passed as an argument to the add_prey_agent function to add a new agent.* @param rand48 Pointer to the seed list of type RNG_rand48. Must be passed as an argument to the rand48 function for generating random numbers on the GPU.
 */
__FLAME_GPU_FUNC__ int prey_reproduction(xmachine_memory_prey* agent, xmachine_memory_prey_list* prey_agents, RNG_rand48* rand48){
    
    /* 
    //Template for agent output functions 
    int new_id = 0;
    int new_x = 0;
    int new_y = 0;
    float new_type = 0;
    float new_fx = 0;
    float new_fy = 0;
    float new_steer_x = 0;
    float new_steer_y = 0;
    int new_life = 0;
    
    add_prey_agent(prey_agents, new_id, new_x, new_y, new_type, new_fx, new_fy, new_steer_x, new_steer_y, new_life);
    */
    
    return 0;
}

/**
 * pred_output_location FLAMEGPU Agent Function
 * Automatically generated using functions.xslt
 * @param agent Pointer to an agent structure of type xmachine_memory_predator. This represents a single agent instance and can be modified directly.
 * @param pred_location_messages Pointer to output message list of type xmachine_message_pred_location_list. Must be passed as an argument to the add_pred_location_message function.
 */
__FLAME_GPU_FUNC__ int pred_output_location(xmachine_memory_predator* agent, xmachine_message_pred_location_list* pred_location_messages){
    
    /* 
    //Template for message output function
    int id = 0;
    float x = 0;
    float y = 0;
    
    add_pred_location_message(pred_location_messages, id, x, y);
    */     
    
    return 0;
}

/**
 * pred_move FLAMEGPU Agent Function
 * Automatically generated using functions.xslt
 * @param agent Pointer to an agent structure of type xmachine_memory_predator. This represents a single agent instance and can be modified directly.
 
 */
__FLAME_GPU_FUNC__ int pred_move(xmachine_memory_predator* agent){
    
    return 0;
}

/**
 * pred_eat_or_starve FLAMEGPU Agent Function
 * Automatically generated using functions.xslt
 * @param agent Pointer to an agent structure of type xmachine_memory_predator. This represents a single agent instance and can be modified directly.
 * @param prey_eaten_messages  prey_eaten_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_prey_eaten_message and get_next_prey_eaten_message functions.
 */
__FLAME_GPU_FUNC__ int pred_eat_or_starve(xmachine_memory_predator* agent, xmachine_message_prey_eaten_list* prey_eaten_messages){
    
    /*
    //Template for input message iteration
    xmachine_message_prey_eaten* current_message = get_first_prey_eaten_message(prey_eaten_messages);
    while (current_message)
    {
        //INSERT MESSAGE PROCESSING CODE HERE
        
        current_message = get_next_prey_eaten_message(current_message, prey_eaten_messages);
    }
    */
    
    return 0;
}

/**
 * pred_reproduction FLAMEGPU Agent Function
 * Automatically generated using functions.xslt
 * @param agent Pointer to an agent structure of type xmachine_memory_predator. This represents a single agent instance and can be modified directly.
 * @param predator_agents Pointer to agent list of type xmachine_memory_predator_list. This must be passed as an argument to the add_predator_agent function to add a new agent.* @param rand48 Pointer to the seed list of type RNG_rand48. Must be passed as an argument to the rand48 function for generating random numbers on the GPU.
 */
__FLAME_GPU_FUNC__ int pred_reproduction(xmachine_memory_predator* agent, xmachine_memory_predator_list* predator_agents, RNG_rand48* rand48){
    
    /* 
    //Template for agent output functions 
    int new_id = 0;
    int new_x = 0;
    int new_y = 0;
    float new_type = 0;
    float new_fx = 0;
    float new_fy = 0;
    float new_steer_x = 0;
    float new_steer_y = 0;
    int new_life = 0;
    
    add_predator_agent(predator_agents, new_id, new_x, new_y, new_type, new_fx, new_fy, new_steer_x, new_steer_y, new_life);
    */
    
    return 0;
}

/**
 * grass_output_location FLAMEGPU Agent Function
 * Automatically generated using functions.xslt
 * @param agent Pointer to an agent structure of type xmachine_memory_grass. This represents a single agent instance and can be modified directly.
 * @param grass_location_messages Pointer to output message list of type xmachine_message_grass_location_list. Must be passed as an argument to the add_grass_location_message function.
 */
__FLAME_GPU_FUNC__ int grass_output_location(xmachine_memory_grass* agent, xmachine_message_grass_location_list* grass_location_messages){
    
    /* 
    //Template for message output function
    int id = 0;
    float x = 0;
    float y = 0;
    
    add_grass_location_message(grass_location_messages, id, x, y);
    */     
    
    return 0;
}

/**
 * grass_eaten FLAMEGPU Agent Function
 * Automatically generated using functions.xslt
 * @param agent Pointer to an agent structure of type xmachine_memory_grass. This represents a single agent instance and can be modified directly.
 * @param prey_location_messages  prey_location_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_prey_location_message and get_next_prey_location_message functions.* @param grass_eaten_messages Pointer to output message list of type xmachine_message_grass_eaten_list. Must be passed as an argument to the add_grass_eaten_message function.
 */
__FLAME_GPU_FUNC__ int grass_eaten(xmachine_memory_grass* agent, xmachine_message_prey_location_list* prey_location_messages, xmachine_message_grass_eaten_list* grass_eaten_messages){
    
    /*
    //Template for input message iteration
    xmachine_message_prey_location* current_message = get_first_prey_location_message(prey_location_messages);
    while (current_message)
    {
        //INSERT MESSAGE PROCESSING CODE HERE
        
        current_message = get_next_prey_location_message(current_message, prey_location_messages);
    }
    */
    
    /* 
    //Template for message output function
    int prey_id = 0;
    
    add_grass_eaten_message(grass_eaten_messages, prey_id);
    */     
    
    return 0;
}

/**
 * grass_growth FLAMEGPU Agent Function
 * Automatically generated using functions.xslt
 * @param agent Pointer to an agent structure of type xmachine_memory_grass. This represents a single agent instance and can be modified directly.
 * @param rand48 Pointer to the seed list of type RNG_rand48. Must be passed as an argument to the rand48 function for generating random numbers on the GPU.
 */
__FLAME_GPU_FUNC__ int grass_growth(xmachine_memory_grass* agent, RNG_rand48* rand48){
    
    return 0;
}

  


#endif //_FLAMEGPU_FUNCTIONS
