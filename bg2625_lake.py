import random
import math
import sys
import numpy

class FrozenLake(object):

    value_policy_map = {}

    def __init__(self, width, height, start, targets, blocked, holes):
        self.initial_state = start 
        self.width = width
        self.height = height
        self.targets = targets
        self.holes = holes
        self.blocked = blocked

        self.actions = ('n', 's', 'e', 'w')
        self.states = set()
        for x in range(width):
            for y in range(height):
                if (x,y) not in self.targets and (x,y) not in self.holes and (x,y) not in self.blocked:
                    self.states.add((x,y))

        # Parameters for the simulation
        self.gamma = 0.9
        self.success_prob = 0.8
        self.hole_reward = -5.0
        self.target_reward = 1.0
        self.living_reward = -0.1

    #### Internal functions for running policies ###

    def get_transitions(self, state, action):
        """
        Return a list of (successor, probability) pairs that
        can result from taking action from state
        """
        result = []
        x, y = state
        remain_p = 0.0

        if action == "n":
            success = (x, y - 1)
            fail = [(x + 1, y), (x - 1, y)]
        elif action == "s":
            success = (x, y + 1)
            fail = [(x + 1, y), (x - 1, y)]
        elif action == "e":
            success = (x + 1, y)
            fail = [(x, y - 1), (x, y + 1)]
        elif action == "w":
            success = (x - 1, y)
            fail = [(x, y - 1), (x, y + 1)]

        if success[0] < 0 or success[0] > self.width - 1 or \
                success[1] < 0 or success[1] > self.height - 1 or \
                success in self.blocked:
            remain_p += self.success_prob
        else:
            result.append((success, self.success_prob))

        for i, j in fail:
            if i < 0 or i > self.width - 1 or \
                    j < 0 or j > self.height - 1 or \
                    (i, j) in self.blocked:
                remain_p += (1 - self.success_prob) / 2
            else:
                result.append(((i, j), (1 - self.success_prob) / 2))

        if remain_p > 0.0:
            result.append(((x, y), remain_p))
        return result

    def move(self, state, action):
        """
        Return the state that results from taking this action
        """
        transitions = self.get_transitions(state, action)
        new_state = random.choices([i[0] for i in transitions], weights=[i[1] for i in transitions])
        return new_state[0]

    def simple_policy_rollout(self, policy):
        """
        Return (Boolean indicating success of trial, total rewards) pair
        """
        state = self.initial_state
        rewards = 0
        while True:
            if state in self.targets:
                return (True, rewards+self.target_reward)
            if state in self.holes:
                return (False, rewards+self.hole_reward)
            state = self.move(state, policy[state])
            rewards += self.living_reward

    def QValue_to_value(self, Qvalues):
        """
        Given a dictionary of q-values corresponding to (state, action) pairs,
        return a dictionary of optimal values for each state
        """
        values = {}
        for state in self.states:
            values[state] = -float("inf")
            for action in self.actions:
                values[state] = max(values[state], Qvalues[(state, action)])
        return values


    #### Some useful functions for you to visualize and test your MDP algorithms ###

    def test_policy(self, policy, t=100):
        """
        Following the policy t times, return (Rate of success, average total rewards)
        """
        numSuccess = 0.0
        totalRewards = 0.0
        for i in range(t):
            result = self.simple_policy_rollout(policy)
            if result[0]:
                numSuccess += 1
            totalRewards += result[1]
        return (numSuccess/t, totalRewards/t)

    def get_random_policy(self):
        """
        Generate a random policy.
        """
        policy = {}
        for i in range(self.width):
            for j in range(self.height):
                policy[(i,j)] = random.choice(self.actions)
        return policy

    def gen_rand_set(width, height, size):
        """
        Generate a random set of grid spaces.
        Useful for creating randomized maps.
        """
        mySet = set([])
        while len(mySet) < size:
            mySet.add((random.randint(0, width), random.randint(0, height)))
        return mySet


    def print_map(self, policy=None):
        """
        Print out a map of the frozen pond, where * indicates start state,
        T indicates target states, # indicates blocked states, and O indicates holes.
        A policy may optimally be provided, which will be printed out on the map as well.
        """
        sys.stdout.write(" ")
        for i in range(2*self.width):
            sys.stdout.write("--")
        sys.stdout.write("\n")
        for j in range(self.height):
            sys.stdout.write("|")
            for i in range(self.width):
                if (i, j) in self.targets:
                    sys.stdout.write("T\t")
                elif (i, j) in self.holes:
                    sys.stdout.write("O\t")
                elif (i, j) in self.blocked:
                    sys.stdout.write("#\t")
                else:
                    if policy and (i, j) in policy:
                        a = policy[(i, j)]
                        if a == "n":
                            sys.stdout.write("^")
                        elif a == "s":
                            sys.stdout.write("v")
                        elif a == "e":
                            sys.stdout.write(">")
                        elif a == "w":
                            sys.stdout.write("<")
                        sys.stdout.write("\t")
                    elif (i, j) == self.initial_state:
                        sys.stdout.write("*\t")
                    else:
                        sys.stdout.write(".\t")
            sys.stdout.write("|")
            sys.stdout.write("\n")
        sys.stdout.write(" ")
        for i in range(2*self.width):
            sys.stdout.write("--")
        sys.stdout.write("\n")

    def print_values(self, values):
        """
        Given a dictionary {state: value}, print out the values on a grid
        """
        for j in range(self.height):
            for i in range(self.width):
                if (i, j) in self.holes:
                    value = self.hole_reward
                elif (i, j) in self.targets:
                    value = self.target_reward
                elif (i, j) in self.blocked:
                    value = 0.0
                else:
                    value = values[(i, j)]
                print("%10.2f" % value, end="")
            print()


    #### Your code starts here ###

    def value_iteration(self, threshold=0.001):
        """
        The value iteration algorithm to iteratively compute an optimal
        value function for all states.
        """
        values = dict((state, 0.0) for state in self.states)
        while True:
            delta = 0
            values1 = values.copy()
            for state in self.states:
                action_values = []
                for action in self.actions:
                    transition_states = lake.get_transitions(state, action)
                    trans_sum = 0

                    for (s1, p) in transition_states:
                        s1_reward = 0
                        if s1 in self.targets:
                            s1_reward = self.target_reward
                        elif s1 in self.holes:
                            s1_reward = self.hole_reward
                        elif s1 in self.blocked:
                            s1_reward = 0
                        else:
                            s1_reward = values1[s1]

                        trans_sum += p * (self.living_reward + self.gamma * s1_reward)
                    action_values.append(trans_sum)

                max_action_value = max(action_values)
                values[state] = max_action_value

                delta = max(delta, abs(values[state] - values1[state]))
                if delta < threshold:
                    return values1

        return values

    def extract_policy(self, values):
        """
        Given state values, return the best policy.
        """
        policy = {}
        total_value = 0
        for state in self.states:
            action_sum = []
            for action in self.actions:
                transition_states = lake.get_transitions(state, action)
                trans_sum = 0
                for (next_state, probability) in transition_states:
                    state_reward = 0
                    if next_state in self.targets:
                        state_reward = self.target_reward
                    elif next_state in self.holes:
                        state_reward = self.hole_reward
                    else:
                        state_reward = values[next_state]

                    trans_sum += probability * (self.living_reward + self.gamma * state_reward)
                action_sum.append(trans_sum)
            max_action_index = numpy.argmax(action_sum)
            policy[state] = self.actions[max_action_index]

        return policy



    def Qlearner(self, alpha, epsilon, num_robots):
        """
        Implement Q-learning with the alpha and epsilon parameters provided.
        Runs number of episodes equal to num_robots.
        """
        Qvalues = {}

        robot_count = num_robots
        final_alpha = 0.1 * alpha
        final_epsilon = 0.1 * epsilon
        current_alpha = alpha
        current_epsilon = epsilon

        for state in self.states:
            for action in self.actions:
                Qvalues[(state, action)] = 0

        state = self.initial_state
        while robot_count > 0:
            chosen_action = 'n'
            best_action = lake.get_best_action_value(Qvalues, state)[0]

            if random.uniform(0, 1) < current_epsilon:
                chosen_action = random.choice(self.actions)
            else:
                chosen_action = best_action

            """
            random_action = random.choice(self.actions)
            chosen_action = random.choices([random_action, best_action], [epsilon, 1 - epsilon])[0]
            """
            next_state = lake.move(state, chosen_action)
            if next_state in self.targets:
                robot_count -= 1
                Qvalues[(state, chosen_action)] = (1 - current_alpha) * Qvalues[(state, chosen_action)] + \
                                           current_alpha * (self.living_reward + self.gamma * self.target_reward)
                state = self.initial_state


            elif next_state in self.holes:
                robot_count -= 1
                Qvalues[(state, chosen_action)] = (1 - current_alpha) * Qvalues[(state, chosen_action)] + \
                                                  current_alpha * (self.living_reward + self.gamma * self.hole_reward)
                state = self.initial_state

            else:
                Qvalues[(state, chosen_action)] = (1 - current_alpha) * Qvalues[(state, chosen_action)] + \
                                           current_alpha * (self.living_reward + self.gamma * lake.get_best_action_value(Qvalues, next_state)[1])
                state = next_state

            if robot_count < 0.2 * num_robots:
                if current_alpha > final_alpha:
                    current_alpha -= 1 / num_robots
                if current_epsilon > final_epsilon:
                    current_epsilon -= 1 / num_robots


        return Qvalues

    def get_best_action_value(self, Qvalues, state):

        action_value_map = {}
        for action in self.actions:
            action_value_map[action] = Qvalues[(state, action)]

        best_action = max(action_value_map.keys(), key = (lambda key:action_value_map[key]))
        best_Qvalue = action_value_map[best_action]
        return (best_action, best_Qvalue)



if __name__ == "__main__":
   
    # Create a lake simulation
    width = 8
    height = 8
    start = (0,0)
    targets = set([(3,4)])
    blocked = set([(3,3), (2,3), (2,4)])
    holes = set([(4, 0), (4, 1), (3, 0), (3, 1), (6, 4), (6, 5), (0, 7), (0, 6), (1, 7)])
    lake = FrozenLake(width, height, start, targets, blocked, holes)

    #lake.gen_rand_set(height, size=64)

    """
    rand_policy = lake.get_random_policy()
    lake.print_map()
    lake.print_map(rand_policy)
    print(lake.test_policy(rand_policy))

    """
    lake.print_map()
    opt_values = lake.value_iteration()
    lake.print_values(opt_values)
    opt_policy = lake.extract_policy(opt_values)
    lake.print_map(opt_policy)
    print(lake.test_policy(opt_policy))
    #
    Qvalues = lake.Qlearner(alpha=0.6, epsilon=0.6, num_robots=60)
    learned_values = lake.QValue_to_value(Qvalues)
    learned_policy = lake.extract_policy(learned_values)
    lake.print_map(learned_policy)
    print(lake.test_policy(learned_policy))