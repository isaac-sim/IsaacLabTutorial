"""Numbered vial goals, target-aware stock rewards, and expanded reset bins."""

from copy import deepcopy
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch
from isaaclab.managers import ObservationTermCfg
from isaaclab.sim import MultiAssetSpawnerCfg
from isaaclab.sim.utils import clone
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply

from isaaclab_tutorial.tasks.place_vial.mdp.events import (
    ResetFromDataset,
    phase_balanced_row_weights,
)
from isaaclab_tutorial.tasks.place_vial.mdp.geometry import (
    RACK_HOLE_CENTERS,
    hole_relative_positions,
    symmetric_axial_keypoint_error,
)
from isaaclab_tutorial.tasks.place_vial.mdp.progress import placement_progress
from isaaclab_tutorial.tasks.place_vial.mdp.rewards import HELD_INSERTION_TARGET
from isaaclab_tutorial.tasks.place_vial.mdp.task import (
    RACK_RIM_HEIGHT,
    VIAL_AXIS_MAX,
    VIAL_AXIS_MIN,
    vial_in_rack,
    vial_upright,
)
from isaaclab_tutorial.tasks.place_vial.mdp.terminations import placement_success
from isaaclab_tutorial.tasks.place_vial.config.so101.env_cfg import (
    VialSpawnCfg,
    spawn_usd_with_contact_material,
)

from .sampling import PHASE_NAMES

NEAR_GOAL_DISTANCE = 0.045
CAP_LABEL_POSITION = (0.0, 0.0, 0.1002)
CAP_LABEL_SIZE = 0.022
RACK_LABEL_SIZE = 0.013
RACK_LABEL_OFFSETS = ((-0.023, -0.023), (0.023, -0.023), (0.023, 0.023), (-0.023, 0.023))


def _attach_label(stage, path, asset_path, position, size, rotation=0.0):
    from pxr import UsdGeom

    label = UsdGeom.Xform.Define(stage, path)
    label.GetPrim().GetReferences().AddReference(str(asset_path))
    label.AddTranslateOp().Set(position)
    label.AddRotateZOp().Set(rotation)
    label.AddScaleOp().Set((size, size, size))


@clone
def spawn_numbered(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """Add render-only decals; preserve every stock collision and contact parameter."""
    prim = spawn_usd_with_contact_material(prim_path, cfg, translation=translation, orientation=orientation, **kwargs)
    if cfg.numbered_kind == "vial":
        from pxr import Sdf

        prim.CreateAttribute("isaaclab_fovi:vialNumber", Sdf.ValueTypeNames.Int).Set(cfg.vial_number)
        _attach_label(
            prim.GetStage(),
            f"{prim_path}/Number",
            Path(cfg.number_label_assets) / f"number_{cfg.vial_number}.usda",
            CAP_LABEL_POSITION,
            CAP_LABEL_SIZE,
            90.0,
        )
    else:
        for i, (center, offset) in enumerate(zip(RACK_HOLE_CENTERS, RACK_LABEL_OFFSETS, strict=True)):
            _attach_label(
                prim.GetStage(),
                f"{prim_path}/Number{i}",
                Path(cfg.number_label_assets) / f"number_{i + 1}.usda",
                (center[0] + offset[0], center[1] + offset[1], RACK_RIM_HEIGHT + 0.0002),
                RACK_LABEL_SIZE,
                90.0,
            )
    return prim


@configclass
class NumberedSpawnCfg(VialSpawnCfg):
    func = spawn_numbered
    numbered_kind: str = "vial"
    number_label_assets: str = ""
    vial_number: int = 1


def balanced_shuffled_clones(weights, num_clones):
    """Keep equal variant counts, shuffled by Isaac Lab's seeded NumPy RNG."""
    from isaaclab.cloner.cloner_strategies import sequential

    layout = np.random.permutation(sequential(weights, num_clones))
    # OVRTX's camera source contract expects the first prototype at env_0.
    # Pin one representative there; all remaining assignments stay shuffled.
    first = int(np.flatnonzero(layout == layout.min())[0])
    layout[0], layout[first] = layout[first], layout[0]
    return layout


def scene_vial_numbers(env):
    """Read immutable cap identities from the actual native clone plan."""
    from isaaclab.cloner import path as cloner_path

    plan = env.scene.clone_plan
    topology = plan.topology
    paths = cloner_path.get_asset_prototype_paths(plan)
    numbers = {}
    for world in np.unique(topology.world_prototype_layout):
        start, end = topology.world_prototype_starts[world + 1 : world + 3]
        candidates = []
        for asset in topology.world_prototypes[start:end]:
            prim = env.scene.stage.GetPrimAtPath(paths[asset])
            number = prim.GetAttribute("isaaclab_fovi:vialNumber").Get()
            if number is not None:
                candidates.append(number)
        if len(candidates) != 1 or candidates[0] not in (1, 2, 3, 4):
            raise ValueError(f"Expected one numbered vial in world prototype {world}: {candidates}")
        numbers[world] = candidates[0]
    return torch.tensor(
        [numbers[world] for world in topology.world_prototype_layout],
        device=env.device,
        dtype=torch.long,
    )


def configure_numbered_scene(cfg):
    """Clone four prebuilt cap variants with unchanged stock physical properties."""
    for name in ("vial", "rack"):
        asset = getattr(cfg.scene, name)
        asset.spawn = NumberedSpawnCfg(
            **{
                field.name: deepcopy(getattr(asset.spawn, field.name))
                for field in fields(asset.spawn)
                if field.init and field.name != "func"
            },
            numbered_kind=name,
            number_label_assets=cfg.number_label_assets,
        )
    original = cfg.scene.vial.spawn
    variants = []
    for number in range(1, 5):
        variant = original.copy()
        variant.vial_number = number
        variants.append(variant)
    cfg.scene.vial.spawn = MultiAssetSpawnerCfg(
        assets_cfg=variants, activate_contact_sensors=original.activate_contact_sensors
    )
    cfg.scene.clone_cfg.clone_strategy = balanced_shuffled_clones
    cfg.events.reset_from_dataset.func = NumberedResetFromDataset
    cfg.terminations.success.func = TargetPlacementSuccess
    cfg.rewards.held_goal.func = held_target_goal
    cfg.observations.critic.placement.func = target_placement_features
    cfg.observations.critic.numbered_goal = ObservationTermCfg(func=critic_goal_state)


def expand_numbered_resets(states, rack_pose):
    """Cross every physical row with every target; split geometrically nearby starts.

    Each original phase retains its base probability mass, with uniform physical
    rows and target numbers before SGS adapts the far/target/non-target bins.
    Physical joint targets and vial poses remain byte-for-byte stock states.
    """
    from isaaclab_tutorial.tasks.place_vial.mdp.geometry import rack_local_position

    expanded = {k: v.repeat_interleave(4, dim=0) for k, v in states.items()}
    target = torch.arange(4, device=states["phase"].device).repeat(len(states["phase"]))
    rack = expanded["vial_pose"].new_tensor(rack_pose).expand(len(target), -1)
    local = rack_local_position(expanded["vial_pose"][:, :3], rack[:, :3], rack[:, 3:])
    distance, nearest = hole_relative_positions(local)[..., :2].norm(dim=-1).min(-1)
    near = distance <= NEAR_GOAL_DISTANCE
    names, bin_phases, bins = [], [], torch.empty_like(target)
    for phase, name in enumerate(PHASE_NAMES):
        masks = (
            (name, ~near),
            (f"{name}_target", near & (target == nearest)),
            (f"{name}_non_target", near & (target != nearest)),
        )
        for label, mask in masks:
            mask = mask & (expanded["phase"] == phase)
            if mask.any():
                bins[mask] = len(names)
                names.append(label)
                bin_phases.append(phase)
    expanded["target_hole"] = target
    # A released vial in the wrong opening needs a fresh pickup before goal fixation.
    regrasp = (expanded["phase"] == 7) & (target != nearest)
    expanded["grasped"] = expanded["grasped"] & ~regrasp
    return expanded, bins, tuple(names), torch.tensor(bin_phases, device=target.device)


def conditional_row_probabilities(row_weights, target_fractions):
    """Actual row probabilities under fixed cap assignments and conditional SGS."""
    weights = row_weights.reshape(-1, 4)
    return (weights / weights.sum(0).clamp_min(1e-30) * target_fractions).flatten()


def sample_matching_rows(row_weights, targets):
    """Sample a physical row for each fixed cap; expanded row order is row x cap."""
    weights = row_weights.reshape(-1, 4).T.contiguous()
    if (weights.sum(-1)[targets] <= 0).any():
        raise ValueError("No enabled reset rows for an assigned cap number")
    # Absent cap groups can have zero mass in targeted diagnostic sampling.
    weights = torch.where(weights.sum(-1, keepdim=True) > 0, weights, torch.ones_like(weights))
    draws = torch.multinomial(weights, len(targets), replacement=True)
    physical = draws[targets, torch.arange(len(targets), device=targets.device)]
    return 4 * physical + targets


class NumberedResetFromDataset(ResetFromDataset):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        rack = env.cfg.scene.rack.init_state
        self.states, self.bin_ids, self.bin_names, phases = expand_numbered_resets(self.states, (*rack.pos, *rack.rot))
        self.bin_phase_ids = phases
        self.row_weights = phase_balanced_row_weights(self.states["phase"], cfg.params["phase_weights"])
        self.sequential_rows = self.row_weights.nonzero().flatten()
        self.cursor = 0
        env.vial_number = scene_vial_numbers(env)
        env.numbered_target_hole = env.vial_number - 1
        env.hole_numbers = torch.arange(1, 5, device=env.device).expand(env.num_envs, -1).clone()
        self.target_fractions = torch.bincount(env.numbered_target_hole, minlength=4) / env.num_envs

    def sampling_row_probabilities(self):
        return conditional_row_probabilities(self.row_weights, self.target_fractions)

    def __call__(
        self,
        env,
        env_ids,
        dataset_path,
        phase_weights,
        sequential=False,
        home_position_noise=0.0,
        home_heading_noise=0.0,
        home_rack_clearance=0.0,
        home_rack_position_noise=0.0,
        home_rack_yaw_noise=0.0,
        support_height_range=(0.0, 0.0),
    ):
        ids = torch.arange(env.num_envs, device=env.device)[slice(None) if env_ids is None else env_ids]
        if not len(ids):
            return
        targets = env.numbered_target_hole[ids]
        saved_rows, saved_cursor = self.sequential_rows, self.cursor
        if sequential:
            rows = torch.empty_like(ids)
            for target in targets.unique():
                available = saved_rows[self.states["target_hole"][saved_rows] == target]
                if not len(available):
                    raise ValueError(f"No sequential reset rows for cap {int(target) + 1}")
                mask = targets == target
                ranks = torch.arange(int(mask.sum()), device=env.device) + saved_cursor
                rows[mask] = available[ranks % len(available)]
        else:
            rows = sample_matching_rows(self.row_weights, targets)
        # Delegate all physical restoration to the stock task via its explicit-row path.
        self.sequential_rows, self.cursor = rows, 0
        try:
            self._restore_rows(
                env,
                ids,
                dataset_path,
                phase_weights,
                home_position_noise,
                home_heading_noise,
                home_rack_clearance,
                home_rack_position_noise,
                home_rack_yaw_noise,
                support_height_range,
            )
        finally:
            self.sequential_rows = saved_rows
            self.cursor = saved_cursor + len(ids) if sequential else saved_cursor

    def _restore_rows(
        self,
        env,
        ids,
        dataset_path,
        phase_weights,
        home_position_noise,
        home_heading_noise,
        home_rack_clearance,
        home_rack_position_noise,
        home_rack_yaw_noise,
        support_height_range,
    ):
        super().__call__(
            env,
            ids,
            dataset_path,
            phase_weights,
            True,
            home_position_noise,
            home_heading_noise,
            home_rack_clearance,
            home_rack_position_noise,
            home_rack_yaw_noise,
            support_height_range,
        )


def target_relative_position(env):
    position, axis, lowest = vial_in_rack(env)
    center = position.new_tensor(RACK_HOLE_CENTERS)[env.numbered_target_hole]
    return position - center, axis, lowest


def selected_goal_center(env, label=False):
    rack = env.scene["rack"].data
    local = rack.root_pos_w.torch.new_tensor(RACK_HOLE_CENTERS)[env.numbered_target_hole].clone()
    local[:, 2] = RACK_RIM_HEIGHT
    if label:
        local[:, :2] += local.new_tensor(RACK_LABEL_OFFSETS)[env.numbered_target_hole]
        local[:, 2] += 0.0002
    return rack.root_pos_w.torch + quat_apply(rack.root_quat_w.torch, local)


def target_placement_features(env):
    position, _, lowest = target_relative_position(env)
    speed = env.scene["vial"].data.root_lin_vel_w.torch.norm(dim=-1)
    return torch.stack((position[:, :2].norm(dim=-1), lowest, vial_upright(env), speed), dim=-1)


def critic_goal_state(env):
    """Privileged target position and identity, excluded from actor groups."""
    local = env.scene["vial"].data.root_pos_w.torch.new_tensor(RACK_HOLE_CENTERS)[env.numbered_target_hole]
    number = torch.nn.functional.one_hot(env.vial_number - 1, 4)
    return torch.cat((local, number), dim=-1).float()


def held_target_goal(env, std=0.10):
    position, axis, _ = target_relative_position(env)
    error = symmetric_axial_keypoint_error(
        position,
        axis,
        position.new_tensor(HELD_INSERTION_TARGET),
        position.new_tensor((0, 0, 1)),
        VIAL_AXIS_MIN,
        VIAL_AXIS_MAX,
    )
    progress = placement_progress(env)
    return torch.where(progress.grasped & ~progress.inserted, torch.exp(-error / std), 0.0)


class TargetPlacementSuccess(placement_success):
    """Retain stock contact/stability logic, accepting only the selected opening.

    Stock success boxes and insertion cylinders are disjoint across openings.
    Outside the target opening's nearest-hole region, roll back only insertion
    and success updates; grasp/lift, contacts, and safety bookkeeping stay stock.
    """

    def __call__(self, env):
        position, _, _ = vial_in_rack(env)
        nearest = hole_relative_positions(position)[..., :2].square().sum(-1).argmin(-1)
        wrong = nearest != env.numbered_target_hole
        inserted, success, time = (
            v.clone() for v in (self.progress.inserted, self.progress.success, self.progress.time_to_success)
        )
        super().__call__(env)
        self.progress.inserted[wrong] = inserted[wrong]
        self.progress.success[wrong] = success[wrong]
        self.progress.time_to_success[wrong] = time[wrong]
        self.progress.stable_count[wrong] = 0
        success = self.progress.success.clone()
        log = env.extras.setdefault("log", {})
        log["Metrics/insertion_rate"] = self.progress.inserted.float().mean()
        log["Metrics/success_rate"] = success.float().mean()
        env._so101_terminal_progress[:, 2] = self.progress.inserted
        env._so101_terminal_time_to_success_s = self.progress.time_to_success.float() * env.step_dt
        env._so101_terminal_hole = torch.where(success, env.numbered_target_hole, -1)
        return success
