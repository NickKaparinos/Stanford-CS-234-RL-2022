from abc import ABC, abstractmethod

import numpy as np
import csv
import os

import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from data import load_data, LABEL_KEY

import pdb


def dose_class(weekly_dose):
    if weekly_dose < 21:
        return 'low'
    elif 21 <= weekly_dose and weekly_dose <= 49:
        return 'medium'
    else:
        return 'high'


# Base classes
class BanditPolicy(ABC):
    @abstractmethod
    def choose(self, x): pass

    @abstractmethod
    def update(self, x, a, r): pass


class StaticPolicy(BanditPolicy):
    def update(self, x, a, r): pass


class RandomPolicy(StaticPolicy):
    def __init__(self, probs=None):
        self.probs = probs if probs is not None else [1. / 3., 1. / 3., 1. / 3.]

    def choose(self, x):
        return np.random.choice(('low', 'medium', 'high'), p=self.probs)


# Baselines
class FixedDosePolicy(StaticPolicy):
    def choose(self, x):
        """
        Args:
            x: Dictionary containing the possible patient features.
        Returns:
            output: string containing one of ('low', 'medium', 'high')

        TODO:
        Please implement the fixed dose algorithm.
        """

        return 'medium'


class ClinicalDosingPolicy(StaticPolicy):
    def choose(self, x):
        """
        Args:
            x: Dictionary containing the possible patient features.
        Returns:
            output: string containing one of ('low', 'medium', 'high')

        TODO:
        Please implement the Clinical Dosing algorithm.

        Hint:
            - You may need to do a little data processing here.
            - Look at the "main" function to see the key values of the features you can use. The
                age in decades is implemented for you as an example.
            - You can treat Unknown race as missing or mixed race.
            - Use dose_class() implemented for you.
        """
        age_in_decades = x['Age in decades']

        if isinstance(x['Age'], float):
            if np.isnan(x['Age']):
                age = 4
            else:
                age = x['Age'] // 10
        else:
            age = int(x['Age'][0])
        dose = 4.0376 - 0.2546 * age + 0.0118 * x['Height (cm)'] + 0.0134 * x['Weight (kg)']
        dose += - 0.6752 * (x['Race'] == 'Asian') + 0.4060 * (x['Race'] in ['Black', 'African American'])
        dose += 0.0443 * (x['Race'] in ['Mixed', 'Unknown']) + 1.2799 * 0 - 0.5695 * x['Amiodarone (Cordarone)']
        return dose_class(dose ** 2)


# Upper Confidence Bound Linear Bandit
class LinUCB(BanditPolicy):
    def __init__(self, n_arms, features, alpha=1.):
        """
        See Algorithm 1 from paper:
            "A Contextual-Bandit Approach to Personalized News Article Recommendation"

        Args:
            n_arms: int, the number of different arms/ actions the algorithm can take
            features: list of strings, contains the patient features to use
            alpha: float, hyperparameter for step size.

        TODO:
        Please initialize the following internal variables for the Disjoint Linear Upper Confidence Bound Bandit algorithm.
        Please refer to the paper to understadard what they are.
        Please feel free to add additional internal variables if you need them, but they are not necessary.

        Hints:
        Keep track of a seperate A, b for each action (this is what the Disjoint in the algorithm name means)
        """
        #######################################################
        #########   YOUR CODE HERE - ~5 lines.   #############
        self.n_arms = n_arms
        self.features = features
        self.alpha = alpha
        self.d = len(features)
        self.A = [np.eye(self.d) for _ in range(n_arms)]
        self.b = [np.zeros((self.d, 1)) for _ in range(n_arms)]
        #######################################################
        #########          END YOUR CODE.          ############

    def choose(self, x):
        """
        See Algorithm 1 from paper:
            "A Contextual-Bandit Approach to Personalized News Article Recommendation"

        Args:
            x: Dictionary containing the possible patient features.
        Returns:
            output: string containing one of ('low', 'medium', 'high')

        TODO:
        Please implement the "forward pass" for Disjoint Linear Upper Confidence Bound Bandit algorithm.
        """
        #######################################################
        #########   YOUR CODE HERE - ~7 lines.   #############
        x_temp = []
        for feature in self.features:
            x_temp.append(x[feature])
        x = np.array(x_temp).reshape(-1, 1)

        p = []
        for a in range(self.n_arms):
            theta = np.matmul(np.linalg.inv(self.A[a]), self.b[a])
            p_temp = np.matmul(theta.transpose(), x) + self.alpha * np.square(
                np.matmul(np.matmul(x.transpose(), np.linalg.inv(self.A[a])), x))
            p.append(p_temp[0, 0])
        a_t = np.argmax(p)
        return ('low', 'medium', 'high')[a_t]

        #######################################################
        #########

    def update(self, x, a, r):
        """
        See Algorithm 1 from paper:
            "A Contextual-Bandit Approach to Personalized News Article Recommendation"

        Args:
            x: Dictionary containing the possible patient features.
            a: string, indicating the action your algorithem chose ('low', 'medium', 'high')
            r: the reward you recieved for that action
        Returns:
            Nothing

        TODO:
        Please implement the update step for Disjoint Linear Upper Confidence Bound Bandit algorithm.

        Hint: Which parameters should you update?
        """
        #######################################################
        #########   YOUR CODE HERE - ~4 lines.   #############
        x_temp = []
        for feature in self.features:
            x_temp.append(x[feature])
        x = np.array(x_temp).reshape(-1, 1)

        a = ('low', 'medium', 'high').index(a)
        self.A[a] += np.matmul(x, x.transpose())
        self.b[a] += r * x
        #######################################################
        #########          END YOUR CODE.          ############


# eGreedy Linear bandit
class eGreedyLinB(LinUCB):
    def __init__(self, n_arms, features, alpha=1.):
        super(eGreedyLinB, self).__init__(n_arms, features, alpha=1.)
        self.time = 0

    def choose(self, x):
        """
        Args:
            x: Dictionary containing the possible patient features.
        Returns:
            output: string containing one of ('low', 'medium', 'high')

        TODO:
        Instead of using the Upper Confidence Bound to find which action to take,
        compute the probability of each action using a simple dot product between Theta & the input features.
        Then use an epsilion greedy algorithm to choose the action.
        Use the value of epsilon provided
        """

        self.time += 1
        epsilon = float(1. / self.time) * self.alpha

        x_temp = []
        for feature in self.features:
            x_temp.append(x[feature])
        x = np.array(x_temp).reshape(-1, 1)

        p = []
        for a in range(self.n_arms):
            theta = np.matmul(self.A[a], x)
            p.append(theta[0, 0])
        a_best = np.argmax(p)

        probabilities = [epsilon / self.n_arms for _ in range(self.n_arms)]
        probabilities[a_best] += 1 - epsilon
        assert (sum(probabilities) - 1) < 1e-5
        action = np.random.choice(('low', 'medium', 'high'), p=probabilities)

        return action


def run(data, learner, large_error_penalty=False):
    # Shuffle
    data = data.sample(frac=1)
    T = len(data)
    n_egregious = 0
    correct = np.zeros(T, dtype=bool)
    for t in range(T):
        x = dict(data.iloc[t])
        label = x.pop(LABEL_KEY)
        action = learner.choose(x)
        correct[t] = (action == dose_class(label))
        reward = int(correct[t]) - 1
        if (action == 'low' and dose_class(label) == 'high') or (action == 'high' and dose_class(label) == 'low'):
            n_egregious += 1
            reward = large_error_penalty
        learner.update(x, action, reward)

    return {
        'total_fraction_correct': np.mean(correct),
        'average_fraction_incorrect': np.mean([
            np.mean(~correct[:t]) for t in range(1, T)]),
        'fraction_incorrect_per_time': [
            np.mean(~correct[:t]) for t in range(1, T)],
        'fraction_egregious': float(n_egregious) / T
    }


def main(args):
    data = load_data()

    frac_incorrect = []
    features = [
        'Age in decades',
        'Height (cm)', 'Weight (kg)',
        'Male', 'Female',
        'Asian', 'Black', 'White', 'Unknown race',
        'Carbamazepine (Tegretol)',
        'Phenytoin (Dilantin)',
        'Rifampin or Rifampicin',
        'Amiodarone (Cordarone)'
    ]

    extra_features = [
        'VKORC1AG', 'VKORC1AA', 'VKORC1UN',
        'CYP2C912', 'CYP2C913', 'CYP2C922',
        'CYP2C923', 'CYP2C933', 'CYP2C9UN'
    ]

    features = features + extra_features

    if args.run_fixed:
        avg = []
        for i in range(args.runs):
            print('Running fixed')
            results = run(data, FixedDosePolicy())
            avg.append(results["fraction_incorrect_per_time"])
            print([(x, results[x]) for x in results if x != "fraction_incorrect_per_time"])
        frac_incorrect.append(("Fixed", np.mean(np.asarray(avg), 0)))

    if args.run_clinical:
        avg = []
        for i in range(args.runs):
            print('Runnining clinical')
            results = run(data, ClinicalDosingPolicy())
            avg.append(results["fraction_incorrect_per_time"])
            print([(x, results[x]) for x in results if x != "fraction_incorrect_per_time"])
        frac_incorrect.append(("Clinical", np.mean(np.asarray(avg), 0)))

    if args.run_linucb:
        avg = []
        for i in range(args.runs):
            print('Running LinUCB bandit')
            results = run(data, LinUCB(3, features, alpha=args.alpha), large_error_penalty=args.large_error_penalty)
            avg.append(results["fraction_incorrect_per_time"])
            print([(x, results[x]) for x in results if x != "fraction_incorrect_per_time"])
        frac_incorrect.append(("LinUCB", np.mean(np.asarray(avg), 0)))

    if args.run_egreedy:
        avg = []
        for i in range(args.runs):
            print('Running eGreedy bandit')
            results = run(data, eGreedyLinB(3, features, alpha=args.ep), large_error_penalty=args.large_error_penalty)
            avg.append(results["fraction_incorrect_per_time"])
            print([(x, results[x]) for x in results if x != "fraction_incorrect_per_time"])
        frac_incorrect.append(("eGreedy", np.mean(np.asarray(avg), 0)))

    if args.run_thompson:
        avg = []
        for i in range(args.runs):
            print('Running Thompson Sampling bandit')
            results = run(data, ThomSampB(3, features, alpha=args.v2), large_error_penalty=args.large_error_penalty)
            avg.append(results["fraction_incorrect_per_time"])
            print([(x, results[x]) for x in results if x != "fraction_incorrect_per_time"])
        frac_incorrect.append(("Thompson", np.mean(np.asarray(avg), 0)))

    os.makedirs('results', exist_ok=True)
    if frac_incorrect != []:
        for algorithm, results in frac_incorrect:
            with open(f'results/{algorithm}.csv', 'w') as f:
                csv.writer(f).writerows(results.reshape(-1, 1).tolist())
    frac_incorrect = []
    for filename in os.listdir('results'):
        if filename.endswith('.csv'):
            algorithm = filename.split('.')[0]
            with open(os.path.join('results', filename), 'r') as f:
                frac_incorrect.append((algorithm, np.array(list(csv.reader(f))).astype('float64').squeeze()))
    plt.xlabel("examples seen")
    plt.ylabel("fraction_incorrect")
    legend = []
    for name, values in frac_incorrect:
        legend.append(name)
        plt.plot(values[10:])
    plt.ylim(0.0, 1.0)
    plt.legend(legend)
    plt.savefig(os.path.join('results', 'fraction_incorrect.png'))


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--run-fixed', action='store_true')
    parser.add_argument('--run-clinical', action='store_true')
    parser.add_argument('--run-linucb', action='store_true')
    parser.add_argument('--run-egreedy', action='store_true')
    parser.add_argument('--run-thompson', action='store_true')
    parser.add_argument('--alpha', type=float, default=1.)
    parser.add_argument('--ep', type=float, default=1)
    parser.add_argument('--v2', type=float, default=0.001)
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--large-error-penalty', type=float, default=-1)
    args = parser.parse_args()
    main(args)
