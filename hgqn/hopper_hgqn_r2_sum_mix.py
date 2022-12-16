import sys

# sys.path.append('/Users/puyuan/code/tianshou')
# sys.path.append('/home/puyuan/tianshou')
sys.path.append('/mnt/nfs/puyuan/tianshou')
# sys.path.append('/mnt/lustre/puyuan/tianshou')

import argparse
import datetime
import os
import pprint

import gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import ContinuousToDiscrete, SubprocVectorEnv
from tianshou.policy import HGQNr2SumMixPolicy

from tianshou.trainer import offpolicy_trainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import HypergraphNet


def get_args():
    parser = argparse.ArgumentParser()
    # task
    parser.add_argument("--task", type=str, default="Hopper-v3")
    # network architecture
    parser.add_argument(
        "--common-hidden-sizes", type=int, nargs="*", default=[512, 256]
    )
    parser.add_argument("--action-hidden-sizes", type=int, nargs="*", default=[128])
    parser.add_argument("--value-hidden-sizes", type=int, nargs="*", default=[128])
    parser.add_argument("--action-per-branch", type=int, default=5)
    # training hyperparameters
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eps-test", type=float, default=0.)
    parser.add_argument("--eps-train", type=float, default=0.73)
    parser.add_argument("--eps-decay", type=float, default=5e-6)
    parser.add_argument("--buffer-size", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--target-update-freq", type=int, default=1000)
    # parser.add_argument("--epoch", type=int, default=1000)
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--step-per-epoch", type=int, default=80000)
    parser.add_argument("--step-per-collect", type=int, default=16)
    parser.add_argument("--update-per-step", type=float, default=0.0625)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--training-num", type=int, default=20)
    parser.add_argument("--test-num", type=int, default=10)
    # parser.add_argument("--training-num", type=int, default=1)
    # parser.add_argument("--test-num", type=int, default=1)
    # other
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--render", type=float, default=0.)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def train(seed):
    args = get_args()
    args.seed = seed
    args.logdir = f"log_hopper_r2_sum_mix_seed{seed}"

    env = gym.make(args.task)
    env = ContinuousToDiscrete(env, args.action_per_branch)

    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    # args.num_branches = args.action_shape if isinstance(args.action_shape,
    #                                                     int) else args.action_shape[0]

    # r2 related
    action_shape = args.action_shape if isinstance(args.action_shape, int) else args.action_shape[0]
    num_branch_pairs = action_shape * args.action_per_branch + int(
        action_shape * (action_shape - 1) / 2) * args.action_per_branch ** 2  # 3*5 + 3*2/2* 5**2 = 15+75=90
    args.num_branches = num_branch_pairs
    # e.g., n=3,
    # rank-1, 1, 2, 3
    # rank-2: 3*2/2=2, (1,2), (2,3), (1,3)
    print("Observations shape:", args.state_shape)
    print("Num branches:", args.num_branches)
    print("Actions per branch:", args.action_per_branch)

    # train_envs = ContinuousToDiscrete(gym.make(args.task), args.action_per_branch)
    # you can also use tianshou.env.SubprocVectorEnv
    train_envs = SubprocVectorEnv(
        [
            lambda: ContinuousToDiscrete(gym.make(args.task), args.action_per_branch)
            for _ in range(args.training_num)
        ]
    )
    # test_envs = ContinuousToDiscrete(gym.make(args.task), args.action_per_branch)
    test_envs = SubprocVectorEnv(
        [
            lambda: ContinuousToDiscrete(gym.make(args.task), args.action_per_branch)
            for _ in range(args.test_num)
        ]
    )
    print("=======env========")

    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)
    # model
    net = HypergraphNet(
        args.state_shape,
        action_shape,
        args.action_per_branch,
        args.common_hidden_sizes,
        args.value_hidden_sizes,
        args.action_hidden_sizes,
        device=args.device,
        mix_type='sum_mix',
    ).to(args.device)
    print("=======net========")

    optim = torch.optim.Adam(net.parameters(), lr=args.lr)
    policy = HGQNr2SumMixPolicy(
        net, optim, args.gamma, target_update_freq=args.target_update_freq, original_action_dim=action_shape,
    )
    # collector
    train_collector = Collector(
        policy,
        train_envs,
        VectorReplayBuffer(args.buffer_size, len(train_envs)),
        exploration_noise=True
    )
    print("=======train_collector========")

    test_collector = Collector(policy, test_envs, exploration_noise=False)
    # policy.set_eps(1)
    train_collector.collect(n_step=args.batch_size * args.training_num)
    # log
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = os.path.join(args.logdir, "bdq", args.task, current_time)
    writer = SummaryWriter(log_path)
    logger = TensorboardLogger(writer)

    def save_best_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, "policy.pth"))

    def stop_fn(mean_rewards):
        return mean_rewards >= getattr(env.spec.reward_threshold)

    def train_fn(epoch, env_step):  # exp decay
        eps = max(args.eps_train * (1 - args.eps_decay) ** env_step, args.eps_test)
        policy.set_eps(eps)

    def test_fn(epoch, env_step):
        policy.set_eps(args.eps_test)

    print("=======trainer begin========")

    # trainer
    result = offpolicy_trainer(
        policy,
        train_collector,
        test_collector,
        args.epoch,
        args.step_per_epoch,
        args.step_per_collect,
        args.test_num,
        args.batch_size,
        update_per_step=args.update_per_step,
        # stop_fn=stop_fn,
        train_fn=train_fn,
        test_fn=test_fn,
        save_best_fn=save_best_fn,
        logger=logger
    )

    print("=======trainer end========")

    # assert stop_fn(result["best_reward"])
    pprint.pprint(result)
    # Let's watch its performance!
    policy.eval()
    policy.set_eps(args.eps_test)
    test_envs.seed(args.seed)
    test_collector.reset()
    result = test_collector.collect(n_episode=args.test_num, render=args.render)
    rews, lens = result["rews"], result["lens"]
    print(f"Final reward: {rews.mean()}, length: {lens.mean()}")


if __name__ == "__main__":
    for seed in [0, 1, 2]:
        train(seed)
