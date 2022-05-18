import numpy as np
from collections import Counter


############ SIMULATOR ############

class Simulator:
    '''
    Simulates a recommender system setup where we have say A arms corresponding to items and U users initially.
    The number of users U cannot change but the number of arms A can increase over time
    '''

    def __init__(self, num_users=10, num_arms=5, num_features=10, update_freq=20, update_arms_strategy='none'):
        '''
        Initializes the attributes of the simulation

        Args:
            num_users: The number of users in the simulation
            num_arms: The number of arms/items in the simulation initially
            num_features: number of features for arms and users
            update_freq: number of steps after which we update the number of arms
            update_arms_strategy: strategy to update the arms. One of 'popular', 'corrective', 'counterfactual'
        '''
        self.num_users = num_users
        self.num_arms = num_arms
        self.num_features = num_features
        self.update_freq = update_freq
        self.update_arms_strategy = update_arms_strategy
        self.arms = {}  ## arm_id: np.array
        self.users = {}  ## user_id: np.array
        self._init(means=np.arange(-5, 6), scale=1.0)
        self.steps = 0  ## number of steps since last arm update
        self.logs = []  ## each element is of the form [user_id, arm_id, best_arm_id]

    def _init(self, means, scale):
        '''
        Initializes the arms and users from a normal distribution where
        each mean is randomly sampled from [-5,5] and variance is always 1.0
        '''
        for i in range(self.num_users):
            v = []
            for _ in range(self.num_features):
                v.append(np.random.normal(loc=np.random.choice(means), scale=scale))
            self.users[i] = np.array(v).reshape(-1)
        for i in range(self.num_arms):
            v = []
            for _ in range(self.num_features):
                v.append(np.random.normal(loc=np.random.choice(means), scale=scale))
            self.arms[i] = np.array(v).reshape(-1)

    def reset(self):
        '''
        Returns a random user context to begin the simulation
        '''
        user_ids = list(self.users.keys())
        user = np.random.choice(user_ids)
        return user, self.users[user]

    def get_reward(self, user_id, arm_id):
        '''
        Returns a reward of 0 if the arm chosen is the best arm else -1
        '''
        user_context = self.users[user_id]
        best_arm_id, best_score = None, None
        for a_id, arm in self.arms.items():
            score = arm.dot(user_context)
            if not best_arm_id:
                best_arm_id = a_id
                best_score = score
                continue
            if best_score < score:
                best_arm_id = a_id
                best_score = score
        ## Update the logs
        self.logs.append([user_id, arm_id, best_arm_id])
        if arm_id == best_arm_id:
            return 0
        else:
            return -1

    def update_arms(self):
        '''
        Three strategies to add a new arm. We will base all these decisions only on the past self.update_frequency (call this K) users' decisions
        1. Counterfactual: Assume there exists some arm that is better than all existing arms for the past K users.
                        We are optimizing to find this arm using SGD on this dataset of K users and their true best arms.
                        The loss is the difference in scores between our current arm and the true best arms of the K users
        2. Corrective: For all the users in the past K users where we got the arm wrong, create a new arm which is the average of their true best arms
        3. Popular: Simply create a new arm which is the mean of the two most popular arms in the last K steps
        4. None: Don't update arms

        Returns True if we added a new arm else False
        '''
        if self.update_arms_strategy == 'none':
            return False
        if self.update_arms_strategy == 'popular':
            '''
            Hints:
            We want to create a new arm with features as the mean of the two most popular arms
            Iterate through the logs and find the two most popular arms.
            Note that each element in self.logs is of the form [user_id, chosen arm_id, best arm_id]
            If there is only one arm in the logs, we don't add a new arm, simply return False
            Find the mean of the true theta of the two most popular arms and add a new arm to the Simulator with a new ID
            Note that self.arms is a dictionary of the form {arm_id: theta} where theta is np.array
            The new arm ID should be the next integer in the sequence of arm IDs
            Don't forget to update self.num_arms
            '''

            if len(self.logs) == 1:
                return False
            else:
                counts = {k: 0 for k in self.arms.keys()}
                for i in range(len(self.logs)):
                    counts[self.logs[i][1]] += 1
                counts = {k: v for k, v in sorted(counts.items(), key=lambda item: item[1])}

                best_arms = [list(counts.keys())[0], list(counts.keys())[1]]
                new_arm = (self.arms[best_arms[0]] + self.arms[best_arms[1]]) / 2

                self.arms[max(list(self.arms.keys())) + 1] = new_arm
                self.num_arms += 1

        if self.update_arms_strategy == 'corrective':
            '''
            Hints:
            We want to create a new arm with features as the weighted mean of the true best arms for users with incorrect predictions
            Iterate through the logs and find the users with incorrect predictions and get their true best arms.
            Note that each element in self.logs is of the form [user_id, chosen arm_id, best arm_id]
            Find the weighted mean of these best_arms and add a new arm to the simulator
            Note that self.arms is a dictionary of the form {arm_id: theta} where theta is np.array
            The new arm ID should be the next integer in the sequence of arm IDs
            Don't forget to update self.num_arms
            '''

            counts = {k: 0 for k in self.arms.keys()}
            for i in range(len(self.logs)):
                if self.logs[i][1] != self.logs[i][2]:
                    counts[self.logs[i][2]] += 1
            new_arm = np.zeros(list(self.arms.values())[0].shape)

            for k, v in counts.items():
                new_arm += v * self.arms[k]
            new_arm /= sum(list(counts.values()))

            self.arms[max(list(self.arms.keys())) + 1] = new_arm
            self.num_arms += 1

        if self.update_arms_strategy == 'counterfactual':
            '''
            Hints:
            We want to create a new arm that optimizes the objective given in the HW PDF
            Initialize the new theta to an array of zeros and the learning rate eta to 0.1
            Perform one update of batch gradient ascent over the logs
            Use the update equation in the HW PDF to update the new theta
            Note that each element in self.logs is of the form [user_id, chosen arm_id, best arm_id]
            Note that self.arms is a dictionary of the form {arm_id: theta} where theta is np.array
            The new ID should be the next integer in the sequence of arm IDs
            Don't forget to update self.num_arms
            '''

            new_arm = np.zeros(list(self.arms.values())[0].shape)
            eta = 0.1

            delta = 0
            for i in range(len(self.logs)):
                x = self.users[self.logs[i][0]]
                temp = np.matmul(new_arm.transpose(), x) - np.matmul(self.arms[self.logs[i][1]].transpose(), x)
                delta += temp*x
            new_arm += eta * delta

            self.arms[max(list(self.arms.keys())) + 1] = new_arm
            self.num_arms += 1

        return True

    def step(self, user_id, arm_id):
        '''
        Takes a step in the simulation, calculates rewards, updates logs, increases arms, and returns the new user context
        Args:
            user_id: The id of the user for which the arm was chosen
            arm_id: The id of the arm chosen
        Returns:
            new user_id for the next step of the simulation
            the user context corresponding to this new user_id
            the reward for the current user-arm interaction (0 if the best arm was chosen, -1 otherwise)
            arm_added: boolean which is True if a new arm was added in this step and False otherwise
        '''
        ## Update number of steps
        self.steps += 1
        ## Get the reward for the arm played. This also updates the logs
        reward = self.get_reward(user_id, arm_id)
        ## Update the arms (add a new arm)
        arm_added = False
        if self.steps % self.update_freq == 0:
            arm_added = self.update_arms()
            self.logs = []
            self.steps = 0
        # if arm_added:
        #	print(f'Added a new arm via {self.update_arms_strategy} strategy! New number of arms is {self.num_arms}')
        ## Get the next user context
        user_ids = list(self.users.keys())
        user = np.random.choice(user_ids)
        return user, self.users[user], reward, arm_added
