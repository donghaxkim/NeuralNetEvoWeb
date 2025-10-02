import streamlit as st
import pygame
import sys
import random
import math
import numpy as np
import io
import base64
from agent import Agent
from food import Food
from environment import Environment
from population import Population
from neural_network_visualizer import NeuralNetworkVisualizer

# Initialize Pygame
pygame.init()

# Constants
SCREEN_WIDTH = 900
SCREEN_HEIGHT = 600
SIMULATION_WIDTH = 900
SIMULATION_HEIGHT = 600
BACKGROUND_COLOR = (30, 30, 30)
TEXT_COLOR = (200, 200, 200)

# Initialize Streamlit
st.set_page_config(
    page_title="Neural Network Evolution Simulation",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Neural Network Evolution Simulation")
st.markdown("Watch AI agents evolve to find food using neural networks and genetic algorithms!")

# Sidebar controls
st.sidebar.header("Controls")

# Initialize session state
if 'simulation_data' not in st.session_state:
    st.session_state.simulation_data = {
        'environment': Environment(SIMULATION_WIDTH, SIMULATION_HEIGHT),
        'population': None,
        'foods': [],
        'generation': 1,
        'best_fitness': 0,
        'generation_time': 0,
        'paused': False,
        'screen': None,
        'clock': None,
        'nn_visualizer': None
    }

def initialize_simulation():
    """Initialize or reset the simulation"""
    data = st.session_state.simulation_data
    
    # Create population
    population_size = st.sidebar.slider("Population Size", 10, 100, 50)
    data['population'] = Population(population_size, data['environment'])
    
    # Create food
    food_count = st.sidebar.slider("Food Count", 5, 50, 20)
    data['foods'] = []
    for _ in range(food_count):
        x = random.randint(30, SIMULATION_WIDTH-30)
        y = random.randint(30, SIMULATION_HEIGHT-30)
        data['foods'].append(Food(x, y))
    
    # Create neural network visualizer
    data['nn_visualizer'] = NeuralNetworkVisualizer(
        SIMULATION_WIDTH + 20, 20, 260, 300
    )
    
    # Reset stats
    data['generation'] = 1
    data['best_fitness'] = 0
    data['generation_time'] = 0
    data['paused'] = False

def reset_food():
    """Reset all food to new random positions"""
    data = st.session_state.simulation_data
    for food in data['foods']:
        food.position_x = random.randint(30, SIMULATION_WIDTH-30)
        food.position_y = random.randint(30, SIMULATION_HEIGHT-30)

def next_generation():
    """Move to next generation"""
    data = st.session_state.simulation_data
    data['generation'] += 1
    data['generation_time'] = 0
    
    # Evolve population
    data['population'].evolve()
    reset_food()

def run_simulation_step():
    """Run one step of the simulation"""
    data = st.session_state.simulation_data
    
    if data['paused'] or not data['population']:
        return
    
    dt = 1.0 / 60.0  # 60 FPS
    data['generation_time'] += dt
    
    # Update agents
    data['population'].update(data['foods'], dt)
    
    # Check food collisions
    for agent in data['population'].agents:
        if agent.alive:
            for food in data['foods']:
                if agent.check_food_collision(food):
                    agent.energy += 50
                    agent.food_eaten += 1
                    food.position_x = random.randint(30, SIMULATION_WIDTH-30)
                    food.position_y = random.randint(30, SIMULATION_HEIGHT-30)
    
    # Update best fitness
    if data['population'].agents:
        current_best = max([agent.get_fitness() for agent in data['population'].agents])
        data['best_fitness'] = max(data['best_fitness'], current_best)
    
    # Check if generation should end
    all_dead = all(not agent.alive for agent in data['population'].agents)
    timeout = data['generation_time'] > 45  # 45 seconds timeout
    
    if all_dead or timeout:
        next_generation()

def render_simulation():
    """Render the simulation to an image"""
    data = st.session_state.simulation_data
    
    if not data['population']:
        return None
    
    # Create a surface for rendering
    surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
    surface.fill(BACKGROUND_COLOR)
    
    # Draw environment
    data['environment'].draw(surface)
    
    # Draw food
    for food in data['foods']:
        food.draw(surface)
    
    # Draw agents
    for agent in data['population'].agents:
        if agent.alive:
            agent.draw(surface)
    
    # Get best agent for neural network visualization
    best_agent = data['population'].get_best_agent()
    if best_agent:
        data['nn_visualizer'].update(best_agent.brain, best_agent.last_inputs, best_agent.last_outputs)
        data['nn_visualizer'].draw(surface)
    
    # Convert to image
    pygame.image.save(surface, "temp_simulation.png")
    
    # Convert to base64 for display
    with open("temp_simulation.png", "rb") as f:
        img_data = f.read()
    
    return base64.b64encode(img_data).decode()

# Initialize simulation if needed
if st.session_state.simulation_data['population'] is None:
    initialize_simulation()

# Control buttons
col1, col2, col3, col4 = st.sidebar.columns(4)

with col1:
    if st.button("▶️ Start"):
        st.session_state.simulation_data['paused'] = False

with col2:
    if st.button("⏸️ Pause"):
        st.session_state.simulation_data['paused'] = True

with col3:
    if st.button("🔄 Reset"):
        initialize_simulation()
        st.rerun()

with col4:
    if st.button("⏭️ Next Gen"):
        next_generation()

# Auto-play toggle
auto_play = st.sidebar.checkbox("Auto-play", value=True)

# Main simulation area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Simulation")
    
    # Run simulation step if auto-play is on
    if auto_play and not st.session_state.simulation_data['paused']:
        run_simulation_step()
    
    # Render and display simulation
    img_data = render_simulation()
    if img_data:
        st.image(f"data:image/png;base64,{img_data}", width=SCREEN_WIDTH)

with col2:
    st.subheader("Statistics")
    
    data = st.session_state.simulation_data
    if data['population']:
        alive_count = sum(1 for agent in data['population'].agents if agent.alive)
        stuck_count = sum(1 for agent in data['population'].agents if agent.alive and agent.is_stuck)
        
        st.metric("Generation", data['generation'])
        st.metric("Best Fitness", f"{data['best_fitness']:.1f}")
        st.metric("Agents Alive", f"{alive_count}/{len(data['population'].agents)}")
        st.metric("Stuck Agents", stuck_count)
        st.metric("Time", f"{data['generation_time']:.1f}s")
        
        # Neural network visualization
        st.subheader("Neural Network (Best Agent)")
        if data['nn_visualizer'] and data['population'].get_best_agent():
            # Create a separate surface for NN visualization
            nn_surface = pygame.Surface((260, 300))
            nn_surface.fill((40, 40, 40))
            data['nn_visualizer'].draw(nn_surface)
            pygame.image.save(nn_surface, "temp_nn.png")
            st.image("temp_nn.png", width=260)

# Instructions
st.sidebar.markdown("""
### Instructions
- **Start/Pause**: Control simulation
- **Reset**: Start over with new population
- **Next Gen**: Force evolution to next generation
- **Auto-play**: Automatically run simulation

### How it works:
1. Agents use neural networks to make decisions
2. They try to find and eat food (red dots)
3. Better performing agents survive and reproduce
4. Neural networks evolve over generations
""")

# Clean up temporary files
import os
for temp_file in ["temp_simulation.png", "temp_nn.png"]:
    if os.path.exists(temp_file):
        os.remove(temp_file)
