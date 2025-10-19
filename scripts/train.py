from ray.rllib.algorithms.impala import IMPALAConfig
from tqdm import tqdm

# Configure IMPALA (local mode for Windows compatibility)
config = (
    IMPALAConfig()
    .environment("CartPole-v1")
    .learners(num_learners=0)  # Local mode (no distributed learners)
    .env_runners(num_env_runners=4)
    .training(
        gamma=0.99,
        lr=0.0005,
        train_batch_size=512,
    )
)

# Build and train for multiple iterations
algo = config.build_algo()
print("Training IMPALA on CartPole-v1...")

for i in tqdm(range(100), desc="Training", unit="iter"):
    result = algo.train()
    episodes = result['env_runners']['num_episodes']
    reward = result['env_runners']['episode_return_mean']
    length = result['env_runners']['episode_len_mean']
    tqdm.write(f"Iter {i+1:2d} | Episodes: {episodes:6.0f} | Reward: {reward:6.2f} | Length: {length:6.2f}")