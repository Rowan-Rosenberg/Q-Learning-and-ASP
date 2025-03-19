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

        # Write movement rules
        f.write("\n% Define row and column domains based on grid_size.\n")
        f.write("row(0..Rmax-1) :- grid_size(Rmax, _).\n")
        f.write("col(0..Cmax-1) :- grid_size(_, Cmax).\n")

        f.write("\n% Define possible actions.\n")
        f.write("action(up).\n")
        f.write("action(down).\n")
        f.write("action(left).\n")
        f.write("action(right).\n")

        f.write("\n% Define when a move is possible.\n")
        f.write("can_move(R, C, up)    :- row(R), col(C), R > 0.\n")
        f.write("can_move(R, C, down)  :- row(R), col(C), grid_size(Rmax, _), R < Rmax - 1.\n")
        f.write("can_move(R, C, left)  :- row(R), col(C), C > 0.\n")
        f.write("can_move(R, C, right) :- row(R), col(C), grid_size(_, Cmax), C < Cmax - 1.\n")

        f.write("\n% Define next_position for valid moves.\n")
        f.write("next_position(R, C, up, Rnew, C) :-\n")
        f.write("can_move(R, C, up),\n")
        f.write("Rnew = R - 1.\n")

        f.write("next_position(R, C, down, Rnew, C) :-\n")
        f.write("can_move(R, C, down),\n")
        f.write("Rnew = R + 1.\n")

        f.write("next_position(R, C, left, R, Cnew) :-\n")
        f.write("can_move(R, C, left),\n")
        f.write("Cnew = C - 1.\n")

        f.write("next_position(R, C, right, R, Cnew) :-\n")
        f.write("can_move(R, C, right),\n")
        f.write("Cnew = C + 1.\n")

        f.write("\n% If a move is not possible, the agent stays in place.\n")
        f.write("next_position(R, C, A, R, C) :-\n")
        f.write("row(R), col(C), action(A),\n")
        f.write("not can_move(R, C, A).\n")
                
        # Write Q-table as facts.
        # Each key in the Q-table is a state, represented as ((row, col), reward_count)
        f.write("% Q-table\n")
        for state, action_dict in data["q_table"].items():
            # Unpack state: expected to be ((row, col), reward_count)
            pos, reward_count = state
            r, c = pos
            for action, q_value in action_dict.items():
                # Scale the floating point value and convert it to an integer.
                scaled_value = int(round(q_value * 10000))
                f.write(f"q_value({r}, {c}, {reward_count}, {action}, {scaled_value}).\n")
        f.write("\n")
        
         # Compute the maximum Q-value for each state using an aggregate.
        f.write("% For each state, compute the maximum Q-value\n")
        f.write("max_q_value(R, C, RC, Max) :- state(R, C, RC), Max = #max { Q, A : q_value(R, C, RC, A, Q) }.\n\n")
        
        # Define best actions: those whose Q-value equals the maximum for their state.
        f.write("% Define best actions as those achieving the maximum Q-value\n")
        f.write("best_action(R, C, RC, A) :- q_value(R, C, RC, A, Q), max_q_value(R, C, RC, Q).\n\n")
        
        # Use a choice rule to select exactly one best action per state.
        # When multiple best actions exist, one is selected arbitrarily.
        f.write("% For each state, choose exactly one best action (if tied, one is chosen arbitrarily)\n")
        f.write("{ chosen_action(R, C, RC, A) : best_action(R, C, RC, A) } = 1 :- state(R, C, RC).\n")

        # Define space for user entered ASP
        f.write("\n\n% ----------- Enter your own ASP rules here ------------- \n\n")

    print(f"ASP representation exported to '{output_filename}'")

def main():

    # Load the exported model and environment data from a file.
    input_filename = "env_and_model.pkl"
    data = load_data(input_filename)
    # Generate the ASP representation of the gridworld and Q-table.
    generate_asp(data)

if __name__ == "__main__":
    main()
