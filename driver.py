import matplotlib.pyplot as plt
from agent import QLearningAgent
from gridworld import GridWorld

def train_agent(num_episodes=500, max_steps=100):
    actions = ['up', 'down', 'left', 'right']
    agent = QLearningAgent(actions)
    
    # Define gridworld parameters.
    # Walls in the grid
    walls = {(1, 2), (2, 1), (2, 2), (5, 0), (5, 2), (5, 3), (5, 4), (0, 6), (1, 6), (2, 6), (3, 6), (7,2), (7, 3), (5, 7), (6, 7)}
    # Reward squares with corresponding reward values
    rewards = [(0,8), (6,3), (3,3)]

    env = GridWorld(grid_size=(10, 10), walls=walls, rewards=rewards, start=(0, 0))
    
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        for step in range(max_steps):
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.learn(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            if done:
                break
        agent.update_epsilon()
        print(f"Episode {episode + 1}/{num_episodes} - Total Reward: {total_reward} - Epsilon: {agent.epsilon:.3f}")
    return agent

def render_trained_model(model_filename="q_learning_model.pkl", max_steps=100, delay=0.5):
    actions = ['up', 'down', 'left', 'right']
    # Load the trained agent.
    agent = QLearningAgent(actions)
    agent.load(model_filename)
    # Set epsilon to 0 for purely greedy behavior.
    agent.epsilon = 0.0

    # Define gridworld parameters (these must match your training settings).
    walls = {(1, 2), (2, 1), (2, 2), (5, 0), (5, 2), (5, 3), (5, 4),
             (0, 6), (1, 6), (2, 6), (3, 6), (7, 2), (7, 3), (5, 7), (6, 7)}
    rewards = [(0, 8), (6, 3), (3, 3)]  # Rewards must be collected in order.
    env = GridWorld(grid_size=(10, 10), walls=walls, rewards=rewards, start=(0, 0))
    
    # Reset the environment; state is augmented (position, current_reward_index)
    state = env.reset()
    
    # Set up interactive mode and create a non-blocking plot window.
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 10))
    env.render_plot(ax)
    plt.show(block=False)
    
    step = 0
    done = False
    print("Running the trained model:")
    while not done and step < max_steps:
        action = agent.choose_action(state)
        state, reward, done = env.step(action)
        print(f"Step {step+1}: Action = {action}, State = {state}, Reward = {reward}, Done = {done}")
        env.render_plot(ax)
        plt.pause(delay)  # Pause to allow the plot to update and be observed.
        step += 1

    # Turn off interactive mode and keep the final plot open.
    plt.ioff()
    plt.show()

def print_q_table(agent):
    print("Q-table contents:")
    for state, action_values in agent.Q.items():
        print(f"State {state}:")
        for action, q_value in action_values.items():
            print(f"  {action}: {q_value:.2f}")
    print()

def main():

    model_filename = "q_learning_model.pkl"
    exit = False
    
    while not exit:

        action = input("Choose an option: \n 1. Train the agent and save \n 2. Load the trained model and render it \n 3. Print the model Q-table \n 4. Exit \n")

        if action == "1":
            # Train the agent.
            episodes = int(input("Enter the number of episodes to train (1000 is appropriate): "))
            trained_agent = train_agent(num_episodes=episodes, max_steps=100)
            # Save the Q-learning model to a file.
            trained_agent.save(model_filename)
            print(f"Trained model saved to {model_filename}")
        elif action == "2":
            # Load the trained model from the file.
            loaded_agent = QLearningAgent(actions=['up', 'down', 'left', 'right'])
            loaded_agent.load(model_filename)
            render_trained_model(model_filename)
        elif action == "3":
            # Print the Q-table of the trained agent.
            print_q_table(trained_agent)
        elif action == "4":
            exit = True
        else:
            print("Invalid option. Please try again.")
        

if __name__ == "__main__":
    main()
