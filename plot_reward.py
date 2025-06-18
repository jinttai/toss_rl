# plot_saved_episodes.py
import os
import glob
import numpy as np
import matplotlib.pyplot as plt

def plot_reward() :
    reward_npz = "rl_results/SelfTossing_V1_10_TD3_comp/training_results_td3_random.npz"
    data = np.load(reward_npz)
    rewards = data['avg_returns']
    episodes = np.arange(len(rewards))
    success_rate = data['success_rates']

    plt.figure(figsize=(8, 12))
    plt.subplot(2, 1, 1)
    plt.plot(episodes*10, rewards, label='Average Reward', color='blue')
    plt.title('Average Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.grid()
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(episodes*10, success_rate, label='Success Rate', color='green')
    plt.title('Success Rate per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Success Rate')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig('reward_plot.png')
    plt.show()

    save_dir = "rl_results/SelfTossing_V1_10_TD3_comp"
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'reward_plot.png'))



def main() -> None:
    plot_reward()


if __name__ == "__main__":

    main()
