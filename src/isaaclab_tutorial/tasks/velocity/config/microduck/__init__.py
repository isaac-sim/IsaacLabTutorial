"""MicroDuck configuration for the velocity-tracking task."""

import gymnasium as gym


def register_task(name, env_cfg, agent_cfg):
    gym.register(
        id=f"Course-Microduck-{name}-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "default_agent": "rsl_rl",
            "env_cfg_entry_point": f"{__name__}.env_cfg:{env_cfg}",
            "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:{agent_cfg}",
        },
    )


register_task("Inspect", "InspectEnvCfg", "BalancePPORunnerCfg")
register_task("Passive", "BaseEnvCfg", "BalancePPORunnerCfg")

register_task("Balance", "BalanceEnvCfg", "BalancePPORunnerCfg")

register_task("BalanceHead", "BalanceHeadEnvCfg", "BalanceHeadPPORunnerCfg")
