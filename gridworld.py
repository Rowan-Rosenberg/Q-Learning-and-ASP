""" 
Simple gridworld environment with matplotlib visualization for machine learning.
The main function allows for testing the environment.
Written by: Rowan Rosenberg March 2025
"""

import numpy as np
import matplotlib.pyplot as plt

class GridWorld:
    def __init__(self, grid_size=(10, 10), walls=None, rewards=None, start=(0, 0)):
        """
        grid_size: tuple (rows, cols)
        walls: set of (row, col) positions that are walls
        rewards: ordered list mapping (row, col) positions to reward values
        terminals: set of (row, col) positions where the episode terminates
        start: starting (row, col) position of the agent
        """
        self.rows, self.cols = grid_size
        self.walls = walls if walls is not None else set()
        self.rewards = rewards if rewards is not None else []
        self.start = start
        self.agent_pos = start
        self.current_reward = 0

    def reset(self):
        """Resets the agent to the starting position."""
        self.agent_pos = self.start
        self.current_reward = 0
        return (self.agent_pos, self.current_reward)

    def step(self, action):
        """
        Takes an action and updates the agent's position.
        action: one of "up", "down", "left", "right".
        Returns: (new_state, reward, done)
        """
        r, c = self.agent_pos
        new_r, new_c = r, c

        if action == "up":
            new_r -= 1
        elif action == "down":
            new_r += 1
        elif action == "left":
            new_c -= 1
        elif action == "right":
            new_c += 1
        else:
            raise ValueError("Invalid action. Choose from 'up', 'down', 'left', or 'right'.")

        # Check grid boundaries; if out of bounds, remain in the same position.
        if new_r < 0 or new_r >= self.rows or new_c < 0 or new_c >= self.cols:
            new_r, new_c = r, c

        # Check if the new position is a wall; if so, don't move.
        if (new_r, new_c) in self.walls:
            new_r, new_c = r, c

        self.agent_pos = (new_r, new_c)

        # Get reward if the agent is on the current reward square.
        if self.rewards[self.current_reward] == self.agent_pos:
            reward = 1
            self.current_reward += 1
        else:
            reward = 0

        # Check if all the rewards are collected.
        done = self.current_reward >= len(self.rewards)

        return (self.agent_pos, self.current_reward), reward, done

    def render_plot(self, ax):
        """Updates the matplotlib axis with the current gridworld state."""
        ax.clear()  # Clear previous drawings

        # Create a color grid: white for empty cells
        grid_colors = np.ones((self.rows, self.cols, 3))
        
        # Color walls as black
        for (r, c) in self.walls:
            grid_colors[r, c] = [0.0, 0.0, 0.0]
            
        # Color rewards based on collection order:
        for idx, pos in enumerate(self.rewards):
            r, c = pos
            if idx < self.current_reward:
                grid_colors[r, c] = [0.8, 0.8, 0.8]  # Already collected (gray)
            elif idx == self.current_reward:
                grid_colors[r, c] = [0.6, 1.0, 0.6]  # Current target (light green)
            else:
                grid_colors[r, c] = [0.6, 0.6, 1.0]  # Future rewards (light blue)

        # Display the grid
        ax.imshow(grid_colors, interpolation='none')

        # Set grid lines
        ax.set_xticks(np.arange(-0.5, self.cols, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, self.rows, 1), minor=True)
        ax.grid(which='minor', color='gray', linewidth=1)
        ax.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

        # Annotate each reward cell with its order number (if not yet collected)
        for idx, pos in enumerate(self.rewards):
            r, c = pos
            if idx >= self.current_reward:
                ax.text(c, r, f"{idx+1}", ha='center', va='center', color='black', fontsize=12)
        
        # Annotate the agent's current position with "A"
        r, c = self.agent_pos
        ax.text(c, r, "A", ha='center', va='center', color='blue', fontsize=14, fontweight='bold')
        
        plt.draw()  # Update the figure


def main():
    # Walls in the grid
    walls = {(1, 2), (2, 1), (2, 2), (5, 0), (5, 2), (5, 3), (5, 4), (0, 6), (1, 6), (2, 6), (3, 6), (7,2), (7, 3), (5, 7), (6, 7)}
    # Reward squares in order of collection
    rewards = [(0,8), (6,3), (3,3)]

    # Initialize the environment
    env = GridWorld(grid_size=(10, 10), walls=walls, rewards=rewards, start=(0, 0))
    
    # Enable interactive mode and create a figure and axis.
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 10))
    env.render_plot(ax)
    
    done = False
    while not done:
        action = input("Enter action (up, down, left, right): ").strip().lower()
        try:
            state, reward, done = env.step(action)
            print(f"New state: {state} | Reward: {reward} | Done: {done}")
            env.render_plot(ax)
            plt.pause(0.01)
        except ValueError as e:
            print(e)
        
        if done:
            print("Episode finished")
            break

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()