"""Reward monitoring callback for SB3 training."""

import os

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class RewardMonitorCallback(BaseCallback):
    """Track and export reward statistics during training."""

    def __init__(self, verbose=1, log_freq=10, output_path=None):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.rewards = []
        self.episode_rewards = []
        self.current_episode_rewards = []
        self.output_path = output_path

    def _on_step(self) -> bool:
        if len(self.model.ep_info_buffer) > 0:
            latest_rewards = [ep_info["r"] for ep_info in self.model.ep_info_buffer]
            if len(latest_rewards) > 0:
                for reward in latest_rewards:
                    if reward not in self.rewards:
                        self.rewards.append(reward)
                        self.episode_rewards.append(reward)

        if hasattr(self.training_env, "buf_rews"):
            step_rewards = self.training_env.buf_rews
            self.current_episode_rewards.extend(step_rewards)

            if self.num_timesteps % self.log_freq == 0:
                if len(step_rewards) > 0:
                    min_rew = np.min(step_rewards)
                    max_rew = np.max(step_rewards)
                    mean_rew = np.mean(step_rewards)
                    print(f"步骤 {self.num_timesteps}:")
                    print(f"  当前步奖励 - 最小: {min_rew:.4f}, 最大: {max_rew:.4f}, 平均: {mean_rew:.4f}")

                if len(self.episode_rewards) > 0:
                    min_ep_rew = np.min(self.episode_rewards)
                    max_ep_rew = np.max(self.episode_rewards)
                    mean_ep_rew = np.mean(self.episode_rewards)
                    print(
                        f"  完整回合奖励 - 最小: {min_ep_rew:.4f}, 最大: {max_ep_rew:.4f}, 平均: {mean_ep_rew:.4f}"
                    )
                    self.episode_rewards = []

        return True

    def on_training_end(self) -> None:
        if len(self.rewards) > 0:
            print("\n训练过程奖励统计:")
            print(f"  最小奖励: {np.min(self.rewards):.4f}")
            print(f"  最大奖励: {np.max(self.rewards):.4f}")
            print(f"  平均奖励: {np.mean(self.rewards):.4f}")
            print(f"  奖励标准差: {np.std(self.rewards):.4f}")

            if np.max(np.abs(self.rewards)) > 1000:
                print("警告: 检测到非常大的奖励值(>1000)，这可能导致训练不稳定。建议考虑奖励缩放。")

            if self.output_path:
                self.export_rewards(self.output_path)

    def export_rewards(self, filepath):
        try:
            import pandas as pd

            df = pd.DataFrame({"step": range(len(self.rewards)), "reward": self.rewards})
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            df.to_csv(filepath, index=False)
            print(f"奖励数据已成功导出到: {filepath}")
        except Exception as exc:
            print(f"导出奖励数据时发生错误: {exc}")

