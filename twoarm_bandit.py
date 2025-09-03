import random
import numpy as np
import matplotlib.pyplot as plt

def plot_normal(mean, std):
    x = np.linspace(mean - 4*std, mean + 4*std, 500)
    y = (1/(std * np.sqrt(2*np.pi))) * np.exp(-0.5 * ((x - mean)/std)**2)
    
    plt.plot(x, y, color="blue")
    plt.title(f"Normal Distribution (μ={mean}, σ={std})")
    plt.xlabel("Value")
    plt.ylabel("Probability Density")
    
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.xticks(np.arange(mean - 4*std, mean + 5*std, std))   # ticks every 1 std
    plt.yticks(np.linspace(0, max(y), 6))                   # ~5 divisions on y-axis
    
    plt.show()


def plot_bandit_results(bandit, rfr, mu, sd):
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    # account 
    axs[0, 0].plot(bandit.N, bandit.account, color="blue")
    axs[0, 0].set_title("Account", fontsize=12, fontweight="bold")
    axs[0, 0].set_xlabel("Months")
    axs[0, 0].set_ylabel("Account Value")
    axs[0, 0].grid(True, linestyle="--", alpha=0.6)

    # rewards 
    axs[0, 1].plot(bandit.N, bandit.rewards, color="green")
    axs[0, 1].set_title("Rewards", fontsize=12, fontweight="bold")
    axs[0, 1].set_xlabel("Months")
    axs[0, 1].set_ylabel("Returns")
    axs[0, 1].grid(True, linestyle="--", alpha=0.6)

    # actions 
    axs[1, 0].step(bandit.N, bandit.actions, where="mid", color="purple")

    axs[1, 0].set_title("Actions", fontsize=12, fontweight="bold")
    axs[1, 0].set_xlabel("Months")
    axs[1, 0].set_ylabel("Action Taken")


    axs[1, 0].set_yticks([1, 2])
    axs[1, 0].set_yticklabels(["Arm 1", "Arm 2"])

    axs[1, 0].grid(True, linestyle="--", alpha=0.6)

    # p2 distribution 
    axs[1, 1].plot(bandit.p2, color="red")
    axs[1, 1].set_title("P2 Distribution", fontsize=12, fontweight="bold")
    axs[1, 1].set_xlabel("Index")
    axs[1, 1].set_ylabel("Value")
    axs[1, 1].grid(True, linestyle="--", alpha=0.6)


    fig.suptitle(f"Two Arm Bandit w/ RFR = {rfr}", fontsize=14, fontweight="bold")

 
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


    plot_normal(mu, sd)


class TwoArmBandit:
    def __init__(self, p1, p2, horizon, initial):
        self.p1 = p1
        self.p2 = p2
        self.initial = initial
        self.horizon = horizon
        self.n = 0
        self.N = range(0, self.horizon, 1 )
        self.account = np.zeros(self.horizon)
        self.account[0] = self.initial
        self.rewards = np.zeros(self.horizon)
        self.actions = np.zeros(self.horizon)
        self.history = []
        # self.regrets = np.zeros(self.horizon)
        # self.best_arm = np.argmax([self.p1, self.p2])
        # self.best_reward = max(self.p1, self.p2)
        self.total_reward = 0
        # self.total_regret = 0

    def pull(self, arm):
        if self.n < self.horizon:
            if arm == 1:
                self.rewards[self.n] = self.p1
                if self.n == 0:
                    self.account[self.n] = self.initial * (1 + self.rewards[self.n])
                else:
                    self.account[self.n] = self.account[self.n - 1] * (1 + self.rewards[self.n])
                self.actions[self.n] = 1
            if arm == 2:
                self.rewards[self.n] = self.p2[self.n]
                if self.n == 0:
                    self.account[self.n] = self.initial * (1 + self.rewards[self.n])
                else:
                    self.account[self.n] = self.account[self.n - 1] * (1 + self.rewards[self.n])    
                self.actions[self.n] = 2
            # self.history[self.n] = (self.n, self.rewards[self.n], arm)
            self.n += 1
        elif self.n >= self.horizon:
            pass
        else:
            pass

rfr = .12
# p1 = (1+rfr ** 1/12) / 100
p1 = .949/100

horizon = 12
mu = p1 
sd = p1*2
initial = 1000

p2 = np.random.normal(mu, sd, horizon)

bandit = TwoArmBandit(p1, p2, horizon, initial)

for _ in range(0, horizon):
    bandit.pull(2)

plot_bandit_results(bandit, rfr, mu, sd)




