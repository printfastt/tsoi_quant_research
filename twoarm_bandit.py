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
    plt.xticks(np.arange(mean - 4*std, mean + 5*std, std))
    plt.yticks(np.linspace(0, max(y), 6))
    
    plt.show()


def plot_bandit_results(bandit, rfr, mu, sd, strategies=None):
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))

    axs[0, 0].plot(bandit.N, bandit.account, color="blue")
    axs[0, 0].set_title("Account", fontsize=12, fontweight="bold")
    axs[0, 0].set_xlabel("Months")
    axs[0, 0].set_ylabel("Account Value")
    axs[0, 0].grid(True, linestyle="--", alpha=0.6)

    axs[0, 1].scatter(bandit.N, bandit.rewards, color="green", s=5)
    axs[0, 1].set_title("Rewards", fontsize=12, fontweight="bold")
    axs[0, 1].set_xlabel("Months")
    axs[0, 1].set_ylabel("Returns")
    axs[0, 1].grid(True, linestyle="--", alpha=0.6)

    axs[0, 2].scatter(bandit.N, bandit.actions, color="purple")
    axs[0, 2].set_title("Actions", fontsize=12, fontweight="bold")
    axs[0, 2].set_xlabel("Months")
    axs[0, 2].set_ylabel("Action Taken")
    axs[0, 2].set_yticks([1, 2])
    axs[0, 2].set_yticklabels(["Arm 1", "Arm 2"])
    axs[0, 2].grid(True, linestyle="--", alpha=0.6)

    if hasattr(bandit, 'p1') and hasattr(bandit, 'p2'):
        axs[1, 0].scatter(bandit.N, bandit.p1, color="blue", s=5, alpha=0.6, label="P1")
        axs[1, 0].scatter(bandit.N, bandit.p2, color="red", s=5, alpha=0.6, label="P2")
        axs[1, 0].set_title("P1 & P2 Distributions", fontsize=12, fontweight="bold")
        axs[1, 0].legend()
    else:
        axs[1, 0].scatter(bandit.N, bandit.p2, color="red", s=5)
        axs[1, 0].set_title("P2 Distribution", fontsize=12, fontweight="bold")
    
    axs[1, 0].set_xlabel("Index")
    axs[1, 0].set_ylabel("Value")
    axs[1, 0].grid(True, linestyle="--", alpha=0.6)

    if hasattr(bandit, 'p1') and hasattr(bandit, 'p2'):
        axs[1, 1].plot(bandit.N, bandit.arm1_estimated_mean, color="blue", label="Arm 1 Est Mean")
        axs[1, 1].plot(bandit.N, bandit.arm2_estimated_mean, color="red", label="Arm 2 Est Mean")
        axs[1, 1].axhline(y=bandit.mu1_true, color="blue", linestyle="--", alpha=0.7, label="Arm 1 True Mean")
        axs[1, 1].axhline(y=bandit.mu2_true, color="red", linestyle="--", alpha=0.7, label="Arm 2 True Mean")
        axs[1, 1].set_title("Estimated vs True Means", fontsize=12, fontweight="bold")
        axs[1, 1].legend()
    else:
        axs[1, 1].plot(bandit.N, bandit.arm2_estimated_mean, color="red", label="Arm 2 Est Mean")
        axs[1, 1].axhline(y=bandit.mu2_true, color="red", linestyle="--", alpha=0.7, label="Arm 2 True Mean")
        axs[1, 1].set_title("Estimated vs True Mean", fontsize=12, fontweight="bold")
        axs[1, 1].legend()
    
    axs[1, 1].set_xlabel("Index")
    axs[1, 1].set_ylabel("Mean")
    axs[1, 1].grid(True, linestyle="--", alpha=0.6)

    if hasattr(bandit, 'p1') and hasattr(bandit, 'p2'):
        axs[1, 2].plot(bandit.N, bandit.arm1_estimated_std, color="blue", label="Arm 1 Est Std")
        axs[1, 2].plot(bandit.N, bandit.arm2_estimated_std, color="red", label="Arm 2 Est Std")
        axs[1, 2].axhline(y=bandit.sd1_true, color="blue", linestyle="--", alpha=0.7, label="Arm 1 True Std")
        axs[1, 2].axhline(y=bandit.sd2_true, color="red", linestyle="--", alpha=0.7, label="Arm 2 True Std")
        axs[1, 2].set_title("Estimated vs True Standard Deviations", fontsize=12, fontweight="bold")
        axs[1, 2].legend()
    else:
        axs[1, 2].plot(bandit.N, bandit.arm2_estimated_std, color="red", label="Arm 2 Est Std")
        axs[1, 2].axhline(y=bandit.sd2_true, color="red", linestyle="--", alpha=0.7, label="Arm 2 True Std")
        axs[1, 2].set_title("Estimated vs True Standard Deviation", fontsize=12, fontweight="bold")
        axs[1, 2].legend()
    
    axs[1, 2].set_xlabel("Index")
    axs[1, 2].set_ylabel("Standard Deviation")
    axs[1, 2].grid(True, linestyle="--", alpha=0.6)

    fig.suptitle(f"Two Arm Bandit w/ RFR = {rfr}", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    plot_normal(mu, sd)

class KArmBandit:
    def __init__(self, horizon, initial):
        self.horizon = horizon
        self.initial = initial
        self.n = 0
        self.N = range(0, self.horizon, 1)
        self.account = np.zeros(self.horizon)
        self.account[0] = self.initial
        self.rewards = np.zeros(self.horizon)
        self.actions = np.zeros(self.horizon)
        self.history = []
        self.total_reward = 0
        self.arm1_estimated_mean = np.zeros(self.horizon)
        self.arm1_estimated_std = np.zeros(self.horizon)
        self.arm2_estimated_mean = np.zeros(self.horizon)
        self.arm2_estimated_std = np.zeros(self.horizon)
        self.arm1_pull_count = 0
        self.arm2_pull_count = 0
        self.arm1_rewards_history = []
        self.arm2_rewards_history = []

    def _update_estimates(self, arm, reward):
        if self.n == 0:
            return
        
        if arm == 1:
            self.arm1_rewards_history.append(reward)
            self.arm1_pull_count += 1
            if self.arm1_pull_count > 0:
                self.arm1_estimated_mean[self.n] = np.mean(self.arm1_rewards_history)
                if self.arm1_pull_count > 1:
                    self.arm1_estimated_std[self.n] = np.std(self.arm1_rewards_history, ddof=1)
                if self.n > 0:
                    self.arm2_estimated_mean[self.n] = self.arm2_estimated_mean[self.n-1]
                    self.arm2_estimated_std[self.n] = self.arm2_estimated_std[self.n-1]
        elif arm == 2:
            self.arm2_rewards_history.append(reward)
            self.arm2_pull_count += 1
            if self.arm2_pull_count > 0:
                self.arm2_estimated_mean[self.n] = np.mean(self.arm2_rewards_history)
                if self.arm2_pull_count > 1:
                    self.arm2_estimated_std[self.n] = np.std(self.arm2_rewards_history, ddof=1)
                if self.n > 0:
                    self.arm1_estimated_mean[self.n] = self.arm1_estimated_mean[self.n-1]
                    self.arm1_estimated_std[self.n] = self.arm1_estimated_std[self.n-1]

class OneArmBandit(KArmBandit):
    def __init__(self, rfr, p2, horizon, initial, mu2_true, sd2_true):
        self.rfr = rfr
        self.p2 = p2
        self.mu2_true = mu2_true
        self.sd2_true = sd2_true
        super().__init__(horizon, initial)

    def pull(self, arm):
        if self.n < self.horizon:
            if arm == 1:
                self.rewards[self.n] = self.rfr
                self.actions[self.n] = 1
            elif arm == 2:
                self.rewards[self.n] = self.p2[self.n]
                self.actions[self.n] = 2

            if self.n == 0:
                self.account[self.n] = self.initial * (1 + self.rewards[self.n])
            else:
                self.account[self.n] = self.account[self.n - 1] * (1 + self.rewards[self.n])

            if arm == 2:
                self._update_estimates(arm, self.rewards[self.n])
            self.history.append((self.n, self.rewards[self.n], self.actions[self.n]))
            self.n += 1
        elif self.n >= self.horizon:
            pass
        else:
            pass

class TwoArmBandit(KArmBandit):
    def __init__(self, p1, p2, horizon, initial, mu1_true, sd1_true, mu2_true, sd2_true):
        self.p1 = p1
        self.p2 = p2
        self.mu1_true = mu1_true
        self.sd1_true = sd1_true
        self.mu2_true = mu2_true
        self.sd2_true = sd2_true
        super().__init__(horizon, initial)

    def pull(self, arm):
        if self.n < self.horizon:
            if arm == 1:
                self.rewards[self.n] = self.p1[self.n]
                self.actions[self.n] = 1
            elif arm == 2:
                self.rewards[self.n] = self.p2[self.n]
                self.actions[self.n] = 2

            if self.n == 0:
                self.account[self.n] = self.initial * (1 + self.rewards[self.n])
            else:
                self.account[self.n] = self.account[self.n - 1] * (1 + self.rewards[self.n])

            self._update_estimates(arm, self.rewards[self.n])
            self.history.append((self.n, self.rewards[self.n], self.actions[self.n]))
            self.n += 1
        elif self.n >= self.horizon:
            pass
        else:
            pass

rfr_year = .12
rfr_month = (1+rfr_year)**(1/12) - 1
rfr_day = (1+rfr_year)**(1/365) - 1

horizon = 12
mu1 = rfr_day
mu2 = rfr_day * 1.5
sd1 = rfr_day * 2
sd2 = rfr_day * 3
initial = 1000

p1 = np.random.normal(mu1, sd1, horizon)
p2 = np.random.normal(mu2, sd2, horizon)
bandit = TwoArmBandit(p1, p2, horizon, initial, mu1, sd1, mu2, sd2)

for _ in range(0, horizon):
    randomint = random.random()
    if randomint < .5:
        bandit.pull(1)
    else:
        bandit.pull(2)

plot_bandit_results(bandit, rfr_year, mu1, sd1)
