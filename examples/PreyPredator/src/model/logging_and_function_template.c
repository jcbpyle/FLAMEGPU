
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

/**
 * prey_output_location FLAMEGPU Agent Function
 * Automatically generated using functions.xslt
 * @param agent Pointer to an agent structure of type xmachine_memory_prey. This represents a single agent instance and can be modified directly.
 * @param prey_location_messages Pointer to output message list of type xmachine_message_prey_location_list. Must be passed as an argument to the add_prey_location_message function ??.
 */
__FLAME_GPU_FUNC__ int prey_output_location(xmachine_memory_prey* agent, xmachine_message_prey_location_list* prey_location_messages){

    
    /* 
    //Template for message output function use int id = 0;
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
 * @param pred_location_messages  pred_location_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_pred_location_message and get_next_pred_location_message functions.* @param prey_eaten_messages Pointer to output message list of type xmachine_message_prey_eaten_list. Must be passed as an argument to the add_prey_eaten_message function ??.
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
    //Template for message output function use int pred_id = 0;
    
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
    //Template for agent output functions int id = 0;
    int x = 0;
    int y = 0;
    float type = 0;
    float fx = 0;
    float fy = 0;
    float steer_x = 0;
    float steer_y = 0;
    int life = 0;
    
    add_prey_agent(prey_agents, int id, int x, int y, float type, float fx, float fy, float steer_x, float steer_y, int life);
    */
    
    return 0;
}

/**
 * pred_output_location FLAMEGPU Agent Function
 * Automatically generated using functions.xslt
 * @param agent Pointer to an agent structure of type xmachine_memory_predator. This represents a single agent instance and can be modified directly.
 * @param pred_location_messages Pointer to output message list of type xmachine_message_pred_location_list. Must be passed as an argument to the add_pred_location_message function ??.
 */
__FLAME_GPU_FUNC__ int pred_output_location(xmachine_memory_predator* agent, xmachine_message_pred_location_list* pred_location_messages){

    
    /* 
    //Template for message output function use int id = 0;
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
    //Template for agent output functions int id = 0;
    int x = 0;
    int y = 0;
    float type = 0;
    float fx = 0;
    float fy = 0;
    float steer_x = 0;
    float steer_y = 0;
    int life = 0;
    
    add_predator_agent(predator_agents, int id, int x, int y, float type, float fx, float fy, float steer_x, float steer_y, int life);
    */
    
    return 0;
}

/**
 * grass_output_location FLAMEGPU Agent Function
 * Automatically generated using functions.xslt
 * @param agent Pointer to an agent structure of type xmachine_memory_grass. This represents a single agent instance and can be modified directly.
 * @param grass_location_messages Pointer to output message list of type xmachine_message_grass_location_list. Must be passed as an argument to the add_grass_location_message function ??.
 */
__FLAME_GPU_FUNC__ int grass_output_location(xmachine_memory_grass* agent, xmachine_message_grass_location_list* grass_location_messages){

    
    /* 
    //Template for message output function use int id = 0;
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
 * @param prey_location_messages  prey_location_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_prey_location_message and get_next_prey_location_message functions.* @param grass_eaten_messages Pointer to output message list of type xmachine_message_grass_eaten_list. Must be passed as an argument to the add_grass_eaten_message function ??.
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
    //Template for message output function use int prey_id = 0;
    
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
