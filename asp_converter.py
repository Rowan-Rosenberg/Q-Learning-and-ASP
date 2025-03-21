"""
This script converts the Q-learning model and environment data into an ASP representation.
The ASP representation can then be used to simulate the agent's behavior
Written by: Rowan Rosenberg March 2025
"""

import pickle
from textwrap import dedent

def load_data(filename):
    #Load model and environment data from pickle file.
    with open(filename, "rb") as f:
        return pickle.load(f)

def generate_asp(data, output_filename="asp_representation.lp", grid_size=(10, 10)):
    #Generate ASP representation of gridworld and Q-table.
    with open(output_filename, "w") as f:
        Writer(f, grid_size).write_all(data)

class Writer:
    #Helper class for structured ASP generation
    
    def __init__(self, file, grid_size):
        self.f = file
        self.grid_size = grid_size
        self.templates = {
            'grid_size': "grid_size({0}, {1}).\n",
            'wall': "wall({0}, {1}).\n",
            'reward': "reward({0}, {1}, {2}).\n",
            'q_value': "q_value({0}, {1}, {2}, {3}, {4}).\n"
        }
        
    def write_all(self, data):
        self._write_header()
        self._write_walls(data["walls"])
        self._write_rewards(data["rewards"])
        self._write_movement_rules()
        self._write_q_table(data["q_table"])
        self._write_logic_rules()
        self._write_user_section()
        print(f"ASP representation exported to '{self.f.name}'")

    def _write_header(self):
        self.f.write("% Gridworld ASP Representation\n\n")
        self.f.write(self.templates['grid_size'].format(*self.grid_size))

    def _write_walls(self, walls):
        self._write_section("Walls", (self.templates['wall'].format(*w) for w in walls))

    def _write_rewards(self, rewards):
        entries = (self.templates['reward'].format(i, *r) 
                 for i, r in enumerate(rewards))
        self._write_section("Rewards", entries)

    def _write_movement_rules(self):
        self.f.write(dedent('''
            % Grid structure and movement rules
            row(0..Rmax-1) :- grid_size(Rmax, _).
            col(0..Cmax-1) :- grid_size(_, Cmax).
            
            action(up; down; left; right).
            
            % Movement possibilities
            can_move(R, C, up)    :- row(R), col(C), R > 0.
            can_move(R, C, down)  :- row(R), col(C), grid_size(Rmax, _), R < Rmax-1.
            can_move(R, C, left)  :- row(R), col(C), C > 0.
            can_move(R, C, right) :- row(R), col(C), grid_size(_, Cmax), C < Cmax-1.
            
            % Position transitions
            next_position(R, C, up,    R-1, C  ) :- can_move(R, C, up).
            next_position(R, C, down,  R+1, C  ) :- can_move(R, C, down).
            next_position(R, C, left,  R,   C-1) :- can_move(R, C, left).
            next_position(R, C, right, R,   C+1) :- can_move(R, C, right).
            next_position(R, C, A,     R,   C  ) :- row(R), col(C), action(A), not can_move(R, C, A).
        ''').replace('            ', '') + '\n')

    def _write_q_table(self, q_table):
        #Write Q-table entries with correct position indexing
        entries = (
            self.templates['q_value'].format(
                pos[0], pos[1], rc, act, int(round(val * 10000))  # scaled Q-value
            )
            for (pos, rc), actions in q_table.items()
            for act, val in actions.items()
        )
        self._write_section("Q-table", entries)

    def _write_logic_rules(self):
        self.f.write(dedent('''
            % State logic
            state(R, C, RC) :- q_value(R, C, RC, _, _).
            
            % Best action selection
            max_q_value(R, C, RC, Max) :- 
                state(R, C, RC), 
                Max = #max { Q, A : q_value(R, C, RC, A, Q) }.
            
            best_action(R, C, RC, A) :- 
                q_value(R, C, RC, A, Q), 
                max_q_value(R, C, RC, Q).
            
            { chosen_action(R, C, RC, A) : best_action(R, C, RC, A) } = 1 :- 
                state(R, C, RC).
            
            % State transitions
            next_state(Rnew, Cnew, RC + 1) :-
                current_state(R, C, RC),
                chosen_action(R, C, RC, A),
                next_position(R, C, A, Rnew, Cnew),
                not wall(Rnew, Cnew),
                reward(RC, Rnew, Cnew).
            
            next_state(Rnew, Cnew, RC) :-
                current_state(R, C, RC),
                chosen_action(R, C, RC, A),
                next_position(R, C, A, Rnew, Cnew),
                not wall(Rnew, Cnew),
                not reward(RC, Rnew, Cnew).
            
            next_state(R, C, RC) :-
                current_state(R, C, RC),
                chosen_action(R, C, RC, A),
                next_position(R, C, A, Rtemp, Ctemp),
                wall(Rtemp, Ctemp).
            
            #show next_state/3.
        ''').replace('            ', '') + '\n')

    def _write_user_section(self):
        self.f.write("\n% ----------- User Configuration -----------\n")
        self.f.write("% Update the initial state as needed:\n")
        self.f.write("current_state(0, 0, 0).\n")

    def _write_section(self, title, entries):
        self.f.write(f"\n% {title}\n")
        self.f.writelines(entries)
        self.f.write("\n")

def main():
    # Load the model and environment data from the pickle file.
    data = load_data("env_and_model.pkl")
    # Generate the ASP representation of the data. Add gridsize parameter if not 10x10
    generate_asp(data)

if __name__ == "__main__":
    main()