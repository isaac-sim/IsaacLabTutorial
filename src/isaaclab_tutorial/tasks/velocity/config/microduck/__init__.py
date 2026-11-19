"""MicroDuck configuration for the velocity-tracking task."""

import gymnasium as gym

from .schedules import install_config_resolution


def register_task(name, env_cfg, agent_cfg, *, symmetry_agent_cfg=None):
    runners = f"{__name__}.agents.rsl_rl_ppo_cfg"
    kwargs = {
        "default_agent": "rsl_rl",
        "env_cfg_entry_point": f"{__name__}.env_cfg:{env_cfg}",
        "rsl_rl_cfg_entry_point": f"{runners}:{agent_cfg}",
    }
    if symmetry_agent_cfg is not None:
        kwargs["rsl_rl_with_symmetry_cfg_entry_point"] = f"{runners}:{symmetry_agent_cfg}"

    gym.register(
        id=f"Course-Microduck-{name}-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs=kwargs,
    )


register_task("Inspect", "InspectEnvCfg", "BalancePPORunnerCfg")
register_task("Passive", "BaseEnvCfg", "BalancePPORunnerCfg")

register_task("Balance", "BalanceEnvCfg", "BalancePPORunnerCfg")

register_task("BalanceHead", "BalanceHeadEnvCfg", "BalanceHeadPPORunnerCfg")

install_config_resolution()
register_task("Track", "TrackingEnvCfg", "WalkPPORunnerCfg")

register_task(
    "Walk",
    "WalkEnvCfg",
    "WalkPPORunnerCfg",
    symmetry_agent_cfg="SymmetricWalkPPORunnerCfg",
)
