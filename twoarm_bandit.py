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


def plot_bandit_results(bandit, rfr, mu, sd, strategies=None):
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

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

    # actions
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
        # self.regrets = np.zeros(self.horizon)
        # self.best_arm = np.argmax([self.p1, self.p2])
        # self.best_reward = max(self.p1, self.p2)
        # self.total_regret = 0

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




rfr_year = .12
rfr_month = (1+rfr_year)**(1/12) - 1
rfr_day = (1+rfr_year)**(1/365) - 1

horizon = 365
mu = rfr_day
sd = rfr_day*2
initial = 1000
p2 = np.random.normal(mu, sd, horizon)
bandit = OneArmBandit(rfr_day, p2, horizon, initial)



#strategy
for _ in range(0, horizon):
    randomint = random.random()
    if randomint < .5:
        bandit.pull(1)
    else:
        bandit.pull(2)

plot_bandit_results(bandit, rfr_year, mu, sd)




