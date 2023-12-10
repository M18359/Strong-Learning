import numpy as np
import pandas as pd
from maze_env import Maze


class OffSarsaN(object):
    # n-step Off-policy Learning by Importance Sampling
    def __init__(self, action_space):
        self.nA = action_space
        self.actions = list(range(action_space))

        self.q_table = pd.DataFrame(columns=self.actions)

    def check_state_exist(self, s):
        if s not in self.q_table.index:
            self.q_table = self.q_table.append(
                pd.Series([0]*len(self.actions),
                          index=self.q_table.columns,
                          name=s)
            )

    def target_policy(self, s):
        # target_policy is the greedy policy
        self.check_state_exist(s)
        A = self.target_policy_probs(s)
        return np.random.choice(range(self.nA), p=A)

    def target_policy_probs(self, s, epsilon=.1):
        A = np.ones(self.nA, dtype=float) * epsilon / self.nA
        best_action = np.argmax(self.q_table.loc[s, :])
        A[best_action] += (1.0 - epsilon)
        return A

    def behaviour_policy(self, s):
        # behaviour policy is the epsilon-greedy
        self.check_state_exist(s)
        A = self.behaviour_policy_probs(s)
        return np.random.choice(range(self.nA), p=A)

    def behaviour_policy_probs(self, s, epsilon=.3):
        A = np.ones(self.nA, dtype=float) * epsilon / self.nA
        best_action = np.argmax(self.q_table.loc[s, :])
        A[best_action] += (1.0 - epsilon)
        return A


if __name__ == '__main__':
    env = Maze()
    action_space = env.n_actions
    RL = OffSarsaN(action_space)

    n = 3
    gamma = 0.9
    alpha = 0.01

    for episode in range(100):
        buffer_s = []
        buffer_a = []
        buffer_r = []
        state = env.reset()
        action = RL.behaviour_policy(str(state))

        buffer_s.append(str(state))
        buffer_a.append(action)

        T = 10000
        t = 0

        while True:
            if t < T:
                env.render()
                state_, reward, done = env.step(action)
                buffer_s.append(str(state_))
                buffer_r.append(reward)

                if state_ == 'terminal':
                    T = t + 1
                else:
                    action_ = RL.behaviour_policy(str(state_))
                    buffer_a.append(action_)
                    action = action_

            tao = t - n + 1

            if tao >= 0:
                rho = 1
                for i in range(tao+1, min(tao+n, T)):
                    rho *= RL.target_policy_probs(buffer_s[i])[buffer_a[i]] /\
                           RL.behaviour_policy_probs(buffer_s[i])[buffer_a[i]]
                G = 0
                for i in range(tao+1, min(tao+n, T)+1):
                    G += gamma**(i-tao-1) * buffer_r[i-1]

                if tao+n < T:
                    G += gamma**n * RL.q_table.loc[buffer_s[tao+n], buffer_a[tao+n]]

                RL.q_table.loc[buffer_s[tao], buffer_a[tao]] += \
                    alpha * rho * (G - RL.q_table.loc[buffer_s[tao], buffer_a[tao]])

            if tao == T-1:
                break

            t += 1

    print('game over')
    env.destroy()
