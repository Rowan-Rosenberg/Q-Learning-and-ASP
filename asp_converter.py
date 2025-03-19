import pickle

def load_data(filename):
   
    # Loads the exported model and environment data from a pickle file.
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data

def generate_asp(data, output_filename="asp_representation.lp", grid_size=(10, 10)):
    
    # Writes an ASP (Clingo) representation of the gridworld and Q-table to a file.
    with open(output_filename, "w") as f:
        # Write grid size
        f.write(f"grid_size({grid_size[0]}, {grid_size[1]}).\n\n")
        
        # Write walls as facts.
        f.write("% Walls\n")
        for wall in data["walls"]:
            # Each wall is a tuple (row, col)
            f.write(f"wall({wall[0]}, {wall[1]}).\n")
        f.write("\n")
        
        # Write rewards as facts.
        # Each reward is a tuple (row, col) and we assign an index (0-indexed)
        f.write("% Rewards\n")
        for idx, reward in enumerate(data["rewards"]):
            f.write(f"reward({idx}, {reward[0]}, {reward[1]}).\n")
        f.write("\n")
        
        # Write Q-table as facts.
        # Each key in the Q-table is a state, represented as ((row, col), reward_count)
        f.write("% Q-table\n")
        for state, action_dict in data["q_table"].items():
            # Unpack state: expected to be ((row, col), reward_count)
            pos, reward_count = state
            r, c = pos
            for action, q_value in action_dict.items():
                # In ASP we represent the action as an atom; note that actions are strings.
                # Q-values are output as a float formatted to four decimals.
                f.write(f"q_value({r}, {c}, {reward_count}, {action}, {q_value:.4f}).\n")
        f.write("\n")
        
    print(f"ASP representation exported to '{output_filename}'")

def main():

    # Load the exported model and environment data from a file.
    input_filename = "env_and_model.pkl"
    data = load_data(input_filename)
    # Generate the ASP representation of the gridworld and Q-table.
    generate_asp(data)

if __name__ == "__main__":
    main()
