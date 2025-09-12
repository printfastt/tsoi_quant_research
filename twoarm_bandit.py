import random
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

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


def plot_bandit_results(bandit, rfr, mu, sd, arm1_estimates=None, arm2_estimates=None):
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))

    # account
    axs[0, 0].plot(bandit.N, bandit.account, color="blue")
    axs[0, 0].set_title("Account", fontsize=12, fontweight="bold")
    axs[0, 0].set_xlabel("Months")
    axs[0, 0].set_ylabel("Account Value")
    axs[0, 0].grid(True, linestyle="--", alpha=0.6)

    # rewards 
    axs[0, 1].scatter(bandit.N, bandit.rewards, color="green", s=5)
    axs[0, 1].set_title("Rewards", fontsize=12, fontweight="bold")
    axs[0, 1].set_xlabel("Months")
    axs[0, 1].set_ylabel("Returns")
    axs[0, 1].grid(True, linestyle="--", alpha=0.6)

    # axs[1, 0].step(bandit.N, bandit.actions, where="mid", color="purple")
    axs[1, 0].scatter(bandit.N, bandit.actions, color="purple")


    axs[1, 0].set_title("Actions", fontsize=12, fontweight="bold")
    axs[1, 0].set_xlabel("Months")
    axs[1, 0].set_ylabel("Action Taken")


    axs[1, 0].set_yticks([1, 2])
    axs[1, 0].set_yticklabels(["Arm 1", "Arm 2"])

    axs[1, 0].grid(True, linestyle="--", alpha=0.6)

    # p2 distribution 
    axs[1, 1].scatter(bandit.N, bandit.p2, color="red", s=5)
    axs[1, 1].set_title("P2 Distribution", fontsize=12, fontweight="bold")
    axs[1, 1].set_xlabel("Index")
    axs[1, 1].set_ylabel("Value")
    axs[1, 1].grid(True, linestyle="--", alpha=0.6)

    # estimated values over time
    if arm1_estimates is not None and arm2_estimates is not None:
        axs[0, 2].plot(bandit.N, arm1_estimates, color="blue", label="Arm 1 (RFR)", linewidth=2)
        axs[0, 2].plot(bandit.N, arm2_estimates, color="red", label="Arm 2 (Risky)", linewidth=2)
        axs[0, 2].axhline(y=bandit.rfr, color="blue", linestyle="--", alpha=0.7, label="True RFR")
        axs[0, 2].axhline(y=mu, color="red", linestyle="--", alpha=0.7, label="True μ")
        axs[0, 2].set_title("Estimated Values Over Time", fontsize=12, fontweight="bold")
        axs[0, 2].set_xlabel("Time")
        axs[0, 2].set_ylabel("Estimated Value")
        axs[0, 2].legend()
        axs[0, 2].grid(True, linestyle="--", alpha=0.6)
    
    x = np.linspace(mu - 4*sd, mu + 4*sd, 500)
    y = (1/(sd * np.sqrt(2*np.pi))) * np.exp(-0.5 * ((x - mu)/sd)**2)
    axs[1, 2].plot(x, y, color="red", linewidth=2)
    axs[1, 2].set_title(f"True Distribution (μ={mu:.4f}, σ={sd:.4f})", fontsize=12, fontweight="bold")
    axs[1, 2].set_xlabel("Value")
    axs[1, 2].set_ylabel("Probability Density")
    axs[1, 2].grid(True, linestyle="--", alpha=0.6)


    fig.suptitle(f"Two Arm Bandit w/ RFR = {rfr}", fontsize=14, fontweight="bold")

 
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

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
        # self.regrets = np.zeros(self.horizon)
        # self.best_arm = np.argmax([self.p1, self.p2])
        # self.best_reward = max(self.p1, self.p2)
        # self.total_regret = 0

class EpsilonGreedy:
    def __init__(self, epsilon, num_arms, strategy="standard", horizon=None):
        self.epsilon = epsilon
        self.num_arms = num_arms
        self.strategy = strategy
        self.horizon = horizon
        self.counts = np.zeros(num_arms)
        self.values = np.zeros(num_arms)
        self.time_step = 0
        
        if strategy in ["epsilon_first", "epsilon_decreasing"] and horizon is None:
            raise ValueError(f"Strategy '{strategy}' requires horizon parameter")
    
    def select_arm(self):
        current_epsilon = self._get_current_epsilon()
        
        if random.random() > current_epsilon:
            return np.argmax(self.values)
        else:
            return random.randrange(self.num_arms)
    
    def _get_current_epsilon(self):
        if self.strategy == "standard":
            return self.epsilon
        elif self.strategy == "epsilon_first":
            exploration_steps = int(self.epsilon * self.horizon)
            return 1.0 if self.time_step < exploration_steps else 0.0
        elif self.strategy == "epsilon_decreasing":
            return self.epsilon * (1 - self.time_step / self.horizon)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] += 1
        n = self.counts[chosen_arm]
        value = self.values[chosen_arm]
        self.values[chosen_arm] = ((n - 1) / n) * value + (1 / n) * reward
        self.time_step += 1

class OneArmBandit(KArmBandit):
    def __init__(self, rfr, p2, horizon, initial):
        self.rfr = rfr
        self.p2 = p2
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

            self.history.append((self.n, self.rewards[self.n], self.actions[self.n]))
            self.n += 1
        elif self.n >= self.horizon:
            pass
        else:
            pass








#SPY returns
# SPYReturns = yf.download("SPY", period="1y", interval="1d")["Close"]
# SPYReturns = SPYReturns.pct_change()
# SPYReturns = SPYReturns.dropna()




rfr_year = .05
rfr_month = (1+rfr_year)**(1/12) - 1
rfr_day = (1+rfr_year)**(1/365) - 1

horizon = 365
mu = rfr_day*1.5
sd = rfr_day*2
initial = 1000
p2 = np.random.normal(mu, sd, horizon)
bandit = OneArmBandit(rfr_day, p2, horizon, initial)

epsilon = 0.1
strategy = EpsilonGreedy(epsilon, 2, strategy="epsilon_first", horizon=horizon)

arm1_estimates = []
arm2_estimates = []

for _ in range(0, horizon):
    arm = strategy.select_arm() + 1
    bandit.pull(arm)
    reward = bandit.rewards[bandit.n - 1]
    strategy.update(arm - 1, reward)
    
    arm1_estimates.append(strategy.values[0])
    arm2_estimates.append(strategy.values[1])

plot_bandit_results(bandit, rfr_year, mu, sd, arm1_estimates, arm2_estimates)




