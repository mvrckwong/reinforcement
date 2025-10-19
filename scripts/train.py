from ray.rllib.algorithms.impala import IMPALAConfig
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

# Configure IMPALA
config = (
    IMPALAConfig()
    .environment("CartPole-v1")
    .learners(num_learners=0)
    .env_runners(num_env_runners=4)
    .training(
        gamma=0.99,
        lr=0.0005,
        train_batch_size=512,
    )
    .callbacks(
        # Use new callback-based loggers instead of deprecated ones
        callbacks_class=None  # Disable default callbacks
    )
    .reporting(
        # Disable the deprecated UnifiedLogger by providing custom loggers
        min_time_s_per_iteration=0,
        min_sample_timesteps_per_iteration=0,
    )
    .resources(num_gpus=0)
    .debugging(log_level="ERROR")
)

# Build algorithm
algo = config.build_algo()
print("Training IMPALA on CartPole-v1...")

for i in tqdm(range(10), desc="Training", unit="iter"):
    result = algo.train()
    episodes = result['env_runners']['num_episodes']
    reward = result['env_runners']['episode_return_mean']
    length = result['env_runners']['episode_len_mean']
    tqdm.write(f"Iter {i+1:2d} | Episodes: {episodes:6.0f} | Reward: {reward:6.2f} | Length: {length:6.2f}")

algo.stop()