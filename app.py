from flask import Flask, render_template, jsonify, request, send_file
import pygame
import sys
import random
import math
import numpy as np
import io
import base64
import threading
import time
from agent import Agent
from food import Food
from environment import Environment
from population import Population
from neural_network_visualizer import NeuralNetworkVisualizer

# Initialize Flask
app = Flask(__name__)

# Initialize Pygame
pygame.init()

# Constants
SCREEN_WIDTH = 900
SCREEN_HEIGHT = 600
SIMULATION_WIDTH = 900
SIMULATION_HEIGHT = 600
BACKGROUND_COLOR = (30, 30, 30)
TEXT_COLOR = (200, 200, 200)

# Global simulation state
simulation_state = {
    'environment': None,
    'population': None,
    'foods': [],
    'generation': 1,
    'best_fitness': 0,
    'generation_time': 0,
    'paused': False,
    'running': False,
    'screen': None,
    'clock': None,
    'nn_visualizer': None,
    'last_frame': None,
    'stats': {
        'alive_count': 0,
        'stuck_count': 0,
        'fps': 60
    }
}

def initialize_simulation():
    """Initialize the simulation"""
    global simulation_state
    
    print("Initializing simulation...")
    
    # Create environment
    simulation_state['environment'] = Environment(SIMULATION_WIDTH, SIMULATION_HEIGHT)
    print(f"Environment created: {SIMULATION_WIDTH}x{SIMULATION_HEIGHT}")
    
    # Create population
    population_size = 50
    simulation_state['population'] = Population(population_size, simulation_state['environment'])
    print(f"Population created with {len(simulation_state['population'].agents)} agents")
    
    # Create food
    food_count = 20
    simulation_state['foods'] = []
    for _ in range(food_count):
        x = random.randint(30, SIMULATION_WIDTH-30)
        y = random.randint(30, SIMULATION_HEIGHT-30)
        simulation_state['foods'].append(Food(x, y))
    print(f"Created {len(simulation_state['foods'])} food items")
    
    # Create neural network visualizer
    simulation_state['nn_visualizer'] = NeuralNetworkVisualizer(
        SIMULATION_WIDTH + 20, 20, 260, 300
    )
    
    # Reset stats
    simulation_state['generation'] = 1
    simulation_state['best_fitness'] = 0
    simulation_state['generation_time'] = 0
    simulation_state['paused'] = False
    simulation_state['running'] = True
    
    print("Simulation initialized successfully!")

def reset_food():
    """Reset all food to new random positions"""
    for food in simulation_state['foods']:
        food.position_x = random.randint(30, SIMULATION_WIDTH-30)
        food.position_y = random.randint(30, SIMULATION_HEIGHT-30)

def next_generation():
    """Move to next generation"""
    simulation_state['generation'] += 1
    simulation_state['generation_time'] = 0
    
    # Evolve population
    simulation_state['population'].evolve()
    reset_food()

def run_simulation_step():
    """Run one step of the simulation"""
    if simulation_state['paused'] or not simulation_state['population']:
        return
    
    dt = 1.0 / 60.0  # 60 FPS
    simulation_state['generation_time'] += dt
    
    # Update agents
    simulation_state['population'].update(simulation_state['foods'], dt)
    
    # Check food collisions
    for agent in simulation_state['population'].agents:
        if agent.alive:
            for food in simulation_state['foods']:
                if agent.check_food_collision(food):
                    agent.energy += 50
                    agent.food_eaten += 1
                    food.position_x = random.randint(30, SIMULATION_WIDTH-30)
                    food.position_y = random.randint(30, SIMULATION_HEIGHT-30)
    
    # Update best fitness
    if simulation_state['population'].agents:
        current_best = max([agent.get_fitness() for agent in simulation_state['population'].agents])
        simulation_state['best_fitness'] = max(simulation_state['best_fitness'], current_best)
    
    # Check if generation should end
    all_dead = all(not agent.alive for agent in simulation_state['population'].agents)
    timeout = simulation_state['generation_time'] > 45  # 45 seconds timeout
    
    if all_dead or timeout:
        print(f"Generation {simulation_state['generation']} ended - All dead: {all_dead}, Timeout: {timeout}")
        next_generation()
    
    # Update stats
    simulation_state['stats']['alive_count'] = sum(1 for agent in simulation_state['population'].agents if agent.alive)
    simulation_state['stats']['stuck_count'] = sum(1 for agent in simulation_state['population'].agents if agent.alive and agent.is_stuck)

def render_simulation():
    """Render the simulation to an image"""
    if not simulation_state['population']:
        return None
    
    # Create a surface for rendering
    surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
    surface.fill(BACKGROUND_COLOR)
    
    # Draw environment
    simulation_state['environment'].draw(surface)
    
    # Draw food
    for food in simulation_state['foods']:
        food.draw(surface)
    
    # Draw agents
    for agent in simulation_state['population'].agents:
        if agent.alive:
            agent.draw(surface)
    
    # Get best agent for neural network visualization
    best_agent = simulation_state['population'].get_best_agent()
    if best_agent:
        simulation_state['nn_visualizer'].update(best_agent.brain, best_agent.last_inputs, best_agent.last_outputs)
        simulation_state['nn_visualizer'].draw(surface)
    
    # Convert to bytes
    pygame.image.save(surface, "temp_simulation.png")
    
    # Convert to base64
    with open("temp_simulation.png", "rb") as f:
        img_data = f.read()
    
    return base64.b64encode(img_data).decode()

def simulation_thread():
    """Background thread for running the simulation"""
    while simulation_state['running']:
        try:
            if not simulation_state['paused']:
                run_simulation_step()
                # Render and store the frame
                simulation_state['last_frame'] = render_simulation()
            time.sleep(1.0 / 60.0)  # 60 FPS
        except Exception as e:
            print(f"Simulation error: {e}")
            time.sleep(0.1)  # Brief pause on error

# Initialize simulation first
initialize_simulation()

# Start simulation thread
simulation_thread = threading.Thread(target=simulation_thread, daemon=True)
simulation_thread.start()

# Give simulation time to start
time.sleep(0.5)

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/api/status')
def get_status():
    """Get current simulation status"""
    return jsonify({
        'generation': simulation_state['generation'],
        'best_fitness': simulation_state['best_fitness'],
        'alive_count': simulation_state['stats']['alive_count'],
        'stuck_count': simulation_state['stats']['stuck_count'],
        'time': simulation_state['generation_time'],
        'paused': simulation_state['paused']
    })

@app.route('/api/frame')
def get_frame():
    """Get current simulation frame"""
    if simulation_state['last_frame']:
        return jsonify({'frame': simulation_state['last_frame']})
    return jsonify({'frame': None})

@app.route('/api/control', methods=['POST'])
def control_simulation():
    """Control the simulation"""
    action = request.json.get('action')
    
    if action == 'pause':
        simulation_state['paused'] = True
    elif action == 'resume':
        simulation_state['paused'] = False
    elif action == 'reset':
        initialize_simulation()
    elif action == 'next_generation':
        next_generation()
    
    return jsonify({'status': 'success'})

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 8000))
    app.run(debug=False, host='0.0.0.0', port=port)
