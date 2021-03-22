import random
import pickle

class QLearn:
    def __init__(self, actions, epsilon, alpha, gamma):
        self.q = {}
        self.epsilon = epsilon  # exploration constant
        self.alpha = alpha      # discount constant
        self.gamma = gamma      # discount factor
        self.actions = actions

    def loadQ(self, filename):
        '''
        Load the Q state-action values from a pickle file.
        '''
        
        with open('{}.pickle'.format(filename), 'rb') as f:
            temp = pickle.load(f)

        f.close()

        self.q = temp

        print("Loaded file: {}".format(filename+".pickle"))

    def saveQ(self, filename):
        '''
        Save the Q state-action values in a pickle file.
        '''
        # just using a text file cause its easier lol

        with open("{}.txt".format(filename), "w") as f:
            for state_action in self.q:
                state = state_action[0]
                action = state_action[1]
                reward = self.q[state_action]
                f.write("State: {} Action: {} Reward: {}".format(state, action, reward))

        f.close()
        
        with open('{}.pickle'.format(filename), 'wb') as f:
            pickle.dump(self.q, f, protocol=pickle.HIGHEST_PROTOCOL)

        print("Wrote to file: {}".format(filename+".pickle"))

    def getQ(self, state, action):
        #returns 0 if state,action not found
        return self.q.get((state, action), 0.0)

    def learnQ(self, state, action, reward, value):
        '''
        Q-learning:
            Q(s, a) += alpha * (reward(s,a) + max(Q(s') - Q(s,a))
        '''
        oldv = self.q.get((state, action), None)
        if oldv is None:
            self.q[(state, action)] = reward
        else:
            self.q[(state, action)] = oldv + self.alpha * (value - oldv)

    def chooseAction(self, state, return_q=False):
        q = [self.getQ(state, a) for a in self.actions]
        maxQ = max(q)

        if random.random() < self.epsilon:
            # rand_val = random.random()
            # num_actions = len(self.actions)
            # action = sum([1 if rand_val > (float(i) / num_actions) else 0
            #             for i in range(num_actions)])-1
            # #print("Num actions: {} | Random value: {} | Random action:{}".
            #      #format(num_actions, rand_val, action))
            # return action
            # ORIGINAL exploration code below
            minQ = min(q)
            mag = max(abs(minQ), abs(maxQ))
            # add random values to all the actions, recalculate maxQ
            q = [q[i] + random.random() * mag - .5 * mag
                 for i in range(len(self.actions))]
            maxQ = max(q)

        count = q.count(maxQ)
        # In case there are several state-action max values
        # we select a random one among them
        if count > 1:
            best = [i for i in range(len(self.actions)) if q[i] == maxQ]
            i = random.choice(best)
        else:
            i = q.index(maxQ)

        action = self.actions[i]
        if return_q:  # if they want it, give it!
            return action, q
        return action

    def learn(self, state1, action1, reward, state2):
        maxqnew = max([self.getQ(state2, a) for a in self.actions])
        self.learnQ(state1, action1, reward, reward + self.gamma*maxqnew)
