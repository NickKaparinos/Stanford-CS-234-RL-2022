from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from itertools import product
from sim import Simulator


############ BANDIT ALGORITHM ############

# Upper Confidence Bound Linear Bandit
class LinUCB:
    def __init__(self, num_arms, num_features, alpha=1.):
        """
        See Algorithm 1 from paper:
            "A Contextual-Bandit Approach to Personalized News Article Recommendation"

        Args:
            num_arms: Initial number of arms
            num_features: Number of features of each arm
            alpha: float, hyperparameter for step size.

        Hints:
        Keep track of a seperate A, b for each arm (this is what the Disjoint in the algorithm name means)
        You may also want to keep track of the number of features to add new parameters for new arms
        """
        self.n_arms = num_arms
        self.alpha = alpha
        self.d = num_features
        self.A = [np.eye(self.d) for _ in range(num_arms)]
        self.b = [np.zeros((self.d, 1)) for _ in range(num_arms)]

    def choose(self, x):
        """
        See Algorithm 1 from paper:
            "A Contextual-Bandit Approach to Personalized News Article Recommendation"

        Args:
            x: numpy array of user features
        Returns:
            output: index of the chosen action

        Please implement the "forward pass" for Disjoint Linear Upper Confidence Bound Bandit algorithm.
        You can return the index of the chosen action directly. No need to return a string name for the action as you did in A4
        """
        p = []
        for a in range(self.n_arms):
            theta = np.matmul(np.linalg.inv(self.A[a]), self.b[a])
            p_temp = np.matmul(theta.transpose(), x) + self.alpha * np.square(
                np.matmul(np.matmul(x.transpose(), np.linalg.inv(self.A[a])), x))
            p.append(p_temp[0])
        a_t = np.argmax(p)
        return a_t

    def update(self, x, a, r):
        """
        See Algorithm 1 from paper:
            "A Contextual-Bandit Approach to Personalized News Article Recommendation"

        Args:
            x: numpy array of user features
            a: integer, indicating the action your algorithm chose in range(0,sim.num_actions)
            r: the reward you recieved for that action
        Returns:
            Nothing

        Please implement the update step for Disjoint Linear Upper Confidence Bound Bandit algorithm.
        """
        x = x.reshape(-1, 1)
        self.A[a] += np.matmul(x, x.transpose())
        self.b[a] += r * x


    def add_arm_params(self):
        """
        Add a new A and b for the new arm we added.
        Initialize them in the same way you did in the __init__ method
        """
        self.A.append(np.eye(self.d))
        self.b.append(np.zeros((self.d, 1)))


############ RUNNER ############

def run(sim, learner, T):
    '''
    Runs the learnerfor T steps on the simulator
    '''
    correct = np.zeros(T, dtype=bool)
    u, x = sim.reset()
    for t in range(T):
        action = learner.choose(x)
        new_u, new_x, reward, arm_added = sim.step(u, action)
        learner.update(x, action, reward)
        if arm_added:
            learner.add_arm_params()
        x, u = new_x, new_u
        correct[t] = (reward == 0)

    return {
        'total_fraction_correct': np.mean(correct),
        'average_fraction_incorrect': np.mean([
            np.mean(~correct[:t]) for t in range(1, T)]),
        'fraction_incorrect_per_time': [
            np.mean(~correct[:t]) for t in range(1, T)],
    }


def main(args):
    frac_incorrect = defaultdict(list)
    frac_correct = defaultdict(list)
    for k, u in product(args.update_freq, args.update_arms_strat):
        stats = []
        print(f'Running LinUCB bandit with K={k} and U={u} with seeds {args.seeds}')
        for seed in args.seeds:
            np.random.seed(seed)
            sim = Simulator(num_users=args.num_users, num_arms=args.num_arms, num_features=args.num_features,
                            update_freq=k, update_arms_strategy=u)
            policy = LinUCB(args.num_arms, args.num_features, alpha=args.alpha)
            results = run(sim, policy, args.T)
            stats.append(results['total_fraction_correct'])
            frac_incorrect[f"LinUCB_K={k}_U={u}"].append(results["fraction_incorrect_per_time"])
            frac_correct[u].append((k, results['total_fraction_correct']))
        stats = np.asarray(stats)
        print(f'Total Fraction Correct: Mean: {stats.mean().round(3)}, Std: {stats.std().round(3)}')
        print('###########################################')

    if args.plot_u:
        plt.xlabel("Users seen")
        plt.ylabel("Fraction Incorrect")
        legend = []
        for name, values in frac_incorrect.items():
            legend.append(name)
            values = np.asarray(values)
            mean, std = values.mean(0)[10:], values.std(0)[10:]
            x = np.arange(len(mean))
            plt.plot(x, mean)
            plt.fill_between(x, mean + std, mean - std, alpha=0.1)
        plt.ylim(0.0, 1.0)
        plt.legend(legend)
        plt.savefig('fraction_incorrect.png')

    if args.plot_k:
        plt.xlabel("K")
        plt.ylabel("Total Fraction Correct")
        for name, values in frac_correct.items():
            x, y = [], {}
            for k, val in values:
                if k not in x:
                    x.append(k)
                    y[k] = [val]
                else:
                    y[k].append(val)
            mean = np.asarray([np.mean(y[k]) for k in x])
            std = np.asarray([np.std(y[k]) for k in x])
            plt.plot(x, mean, label=name)
            plt.fill_between(x, mean + std, mean - std, alpha=0.1)
        plt.ylim(0.0, 1.0)
        plt.legend()
        plt.savefig('k_analysis.png')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--seeds', '-s', nargs='+', type=int, default=[0])
    parser.add_argument('--num-users', type=int, default=25)
    parser.add_argument('--num-arms', type=int, default=10)
    parser.add_argument('--num-features', type=int, default=10)
    parser.add_argument('--update-freq', '-k', type=int, nargs='+', default=[1000])
    parser.add_argument('--update-arms-strat', '-u', type=str, nargs='+', default=['none'],
                        choices=['none', 'counterfactual', 'popular', 'corrective'])
    parser.add_argument('--alpha', type=float, default=1.)
    parser.add_argument('--T', type=int, default=10000)
    parser.add_argument('--plot-u', action='store_true')
    parser.add_argument('--plot-k', action='store_true')
    args = parser.parse_args()
    main(args)
