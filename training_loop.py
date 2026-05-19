import gymnasium as gym
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from agent import GridWorldsAgent
from envs.cliff_v0 import Cliff
from envs.gridworld_v0 import GridWorld
from envs.windy_gridworld import WindyGridWorld

# Per-env hyperparameters. State spaces and reward scales differ, so train
# length and learning rate are tuned independently.
ENV_CONFIGS = {
    "GridWorld": {
        "env_cls": GridWorld,
        "n_episodes": 5_000,
        "learning_rate": 0.1,
        "discount_factor": 0.95,
        "start_epsilon": 1.0,
        "final_epsilon": 0.05,
    },
    "Cliff": {
        "env_cls": Cliff,
        "n_episodes": 5000,
        "learning_rate": 0.1,
        "discount_factor": 1.0,
        "start_epsilon": 1.0,
        "final_epsilon": 0.1,
    },
    "WindyGridWorld": {
        "env_cls": WindyGridWorld,
        "n_episodes": 5000,
        "learning_rate": 0.5,
        "discount_factor": 1.0,
        "start_epsilon": 1.0,
        "final_epsilon": 0.1,
    },
}


def train(env, agent, n_episodes):
    for _ in tqdm(range(n_episodes)):
        obs, _ = env.reset()
        obs = tuple(obs)
        done = False
        while not done:
            action = agent.get_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_obs = tuple(next_obs)
            agent.update(obs, action, reward, terminated, next_obs)
            obs = next_obs
            done = terminated or truncated
        agent.decay_epsilon()


def get_moving_avgs(arr, window, convolution_mode):
    """Compute moving average to smooth noisy data."""
    return np.convolve(
        np.array(arr).flatten(),
        np.ones(window),
        mode=convolution_mode,
    ) / window


def plot_training(env, agent, title, rolling_length=500):
    fig, axs = plt.subplots(ncols=3, figsize=(12, 5))
    fig.suptitle(title)

    axs[0].set_title("Episode rewards")
    reward_ma = get_moving_avgs(env.return_queue, rolling_length, "valid")
    axs[0].plot(range(len(reward_ma)), reward_ma)
    axs[0].set_ylabel("Average Reward")
    axs[0].set_xlabel("Episode")

    axs[1].set_title("Episode lengths")
    length_ma = get_moving_avgs(env.length_queue, rolling_length, "valid")
    axs[1].plot(range(len(length_ma)), length_ma)
    axs[1].set_ylabel("Average Episode Length")
    axs[1].set_xlabel("Episode")

    axs[2].set_title("Training Error")
    td_ma = get_moving_avgs(agent.training_error, rolling_length, "same")
    axs[2].plot(range(len(td_ma)), td_ma)
    axs[2].set_ylabel("Temporal Difference Error")
    axs[2].set_xlabel("Step")

    plt.tight_layout()
    plt.show()

# Test the trained agent
def test_agent(agent, env, num_episodes=1000):
    """Test agent performance without learning or exploration."""
    total_rewards = []
    episode_lengths = []

    old_epsilon = agent.epsilon
    agent.epsilon = 0.0
    try:
        for _ in range(num_episodes):
            obs, _ = env.reset()
            obs = tuple(obs)
            episode_reward = 0
            steps = 0
            done = False
            while not done:
                action = agent.get_action(obs)
                next_obs, reward, terminated, truncated, _ = env.step(action)
                obs = tuple(next_obs)
                episode_reward += reward
                steps += 1
                done = terminated or truncated
            total_rewards.append(episode_reward)
            episode_lengths.append(steps)
    finally:
        agent.epsilon = old_epsilon

    print(f"Test Results over {num_episodes} episodes:")
    print(f"Average Reward: {np.mean(total_rewards):.3f}")
    print(f"Average Length: {np.mean(episode_lengths):.2f}")
    print(f"Standard Deviation: {np.std(total_rewards):.3f}")


def plot_greedy_path(env, agent, title, num_episodes=300):
    """Visualize state-visitation under the greedy policy + per-cell arrows.

    Works for envs with `width`, `height`, `_terminal_state`, and an
    `_action_to_direction` mapping where obs is (col, row) and direction is (dx, dy).
    """
    base = env.unwrapped
    width, height = base.width, base.height
    visits = np.zeros((height, width), dtype=int)

    old_epsilon = agent.epsilon
    agent.epsilon = 0.0
    try:
        start_obs, _ = env.reset()
        max_steps = width * height * 20
        for _ in range(num_episodes):
            obs, _ = env.reset()
            obs = tuple(obs)
            done = False
            steps = 0
            while not done and steps < max_steps:
                c, r = obs
                visits[r, c] += 1
                action = agent.get_action(obs)
                next_obs, _, terminated, truncated, _ = env.step(action)
                obs = tuple(next_obs)
                done = terminated or truncated
                steps += 1
            c, r = obs
            visits[r, c] += 1
    finally:
        agent.epsilon = old_epsilon

    fig, ax = plt.subplots(figsize=(max(6, width * 0.6), max(4, height * 0.6) + 1))
    ax.set_title(f"{title}: greedy path visitation ({num_episodes} rollouts)")
    im = ax.imshow(visits, origin="lower", cmap="Blues", aspect="equal")
    fig.colorbar(im, ax=ax, label="visits")

    for r in range(height):
        for c in range(width):
            if visits[r, c] == 0:
                continue
            qs = agent.q_values.get((c, r))
            if qs is None:
                continue
            dx, dy = base._action_to_direction[int(np.argmax(qs))]
            ax.arrow(c, r, dx * 0.3, dy * 0.3,
                     head_width=0.15, head_length=0.15,
                     fc="black", ec="black", length_includes_head=True)

    sx, sy = start_obs
    gx, gy = base._terminal_state
    ax.scatter([sx], [sy], marker="o", s=180, c="lime",
               edgecolors="black", linewidths=1.5, label="start", zorder=3)
    ax.scatter([gx], [gy], marker="*", s=300, c="gold",
               edgecolors="black", linewidths=1.5, label="goal", zorder=3)

    if hasattr(base, "_cliff"):
        cliff_cols = list(base._cliff)
        ax.scatter(cliff_cols, [0] * len(cliff_cols), marker="X", s=180, c="red",
                   edgecolors="black", linewidths=1.0, label="cliff", zorder=2)

    ax.set_xticks(range(width))
    ax.set_yticks(range(height))
    ax.set_xlim(-0.5, width - 0.5)
    ax.set_ylim(-0.5, height - 0.5)
    ax.grid(True, color="gray", linewidth=0.3, alpha=0.5)
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


trained_agents = {}
for name, cfg in ENV_CONFIGS.items():
    print(f"\nTraining on {name}")
    n_episodes = cfg["n_episodes"]
    env = gym.wrappers.RecordEpisodeStatistics(cfg["env_cls"](), buffer_length=n_episodes)

    agent = GridWorldsAgent(
        env=env,
        learning_rate=cfg["learning_rate"],
        initial_epsilon=cfg["start_epsilon"],
        epsilon_decay=cfg["start_epsilon"] / (n_episodes / 2),
        final_epsilon=cfg["final_epsilon"],
        discount_factor=cfg["discount_factor"],
    )
    train(env, agent, n_episodes)
    trained_agents[name] = (env, agent)
    plot_training(env, agent, title=name)
    test_agent(agent, env, n_episodes)
    if name in ("Cliff", "WindyGridWorld"):
        plot_greedy_path(env, agent, title=name)


