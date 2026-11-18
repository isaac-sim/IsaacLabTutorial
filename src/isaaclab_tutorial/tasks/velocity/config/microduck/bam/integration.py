# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""BAM schema authoring and lifecycle integration with Isaac Lab 3.0 GA."""

from __future__ import annotations

import importlib

import torch
import warp as wp
from newton import JointType
from newton.actuators import Actuator
from pxr import Sdf

from isaaclab.sim.schemas import schemas_actuators as schemas
from isaaclab.utils import to_dict, validate
from isaaclab.utils.string import resolve_matching_names, to_camel_case
from isaaclab_newton.assets.articulation.actuator_control import NewtonActuatorControl
from isaaclab_newton.physics import NewtonManager as SimulationManager

from .bam import DriveBam
from .bam_cfg import BAM_DRIVE_API, BAM_MODEL_FLAGS, BamActuatorCfg
from .mjwarp_actuator_bridge import MjWarpActuatorBridge

_original_author = schemas.author_actuator_prims
_original_supported = schemas._is_newton_native_actuator_cfg
_original_define = schemas.define_actuator_properties
_original_initialize = SimulationManager.initialize_solver.__func__
_pending_controls = []


def define_actuator_properties(prim_path, actuator_cfgs, stage=None):
    if any(isinstance(cfg, BamActuatorCfg) for cfg in actuator_cfgs.values()):
        from isaaclab.sim import SimulationContext
        from isaaclab_newton.physics import NewtonCfg

        context = SimulationContext.instance()
        if context is None or not isinstance(context.cfg.physics, NewtonCfg):
            raise ValueError("BAM requires the Newton backend with MJWarp.")
        if not context.cfg.use_newton_actuators:
            raise ValueError("BAM requires use_newton_actuators=True.")
    return _original_define(prim_path, actuator_cfgs, stage=stage)


def author_actuator_prims(stage, articulation_prim_path, actuator_cfgs):
    """Delegate GA actuators and author BAM's own USD drive schema."""
    ordinary = {name: cfg for name, cfg in actuator_cfgs.items() if not isinstance(cfg, BamActuatorCfg)}
    if ordinary:
        _original_author(stage, articulation_prim_path, ordinary)
    art_prim = stage.GetPrimAtPath(articulation_prim_path)
    inventory = schemas._collect_joint_prims(art_prim)
    for group, cfg in actuator_cfgs.items():
        if not isinstance(cfg, BamActuatorCfg):
            continue
        validate(cfg)
        _, names = resolve_matching_names(cfg.joint_names_expr, list(inventory))
        schemas._remove_actuator_prims_for_joints(art_prim, {inventory[name] for name in names})
        motor = to_dict(cfg.motor)
        del motor["model"], motor["friction_viscous"]
        attrs = {
            **{name: float(value) for name, value in motor.items()},
            "kp_fw": float(cfg.kp_fw),
            "vin": float(cfg.vin),
            "vin_min": float(cfg.vin_min),
            "delay_hold_prob": float(cfg.delay_hold_prob),
            **BAM_MODEL_FLAGS[cfg.motor.model],
            "has_backlash": int(cfg.backlash_joint_template is not None),
            "min_delay": cfg.min_delay,
            "max_delay": cfg.max_delay,
            "delay_update_period": cfg.delay_update_period,
            "delay_seed": 0,
            "kp_scale": 1.0,
            "kd_scale": 1.0,
            "friction_scale": 1.0,
            "sag_gain": 0.0,
        }
        voltage = max(cfg.vin_range) if cfg.vin_range is not None else cfg.vin
        limit = cfg.actuator_effort_limit
        if limit is None:
            limit = voltage * cfg.motor.kt / cfg.motor.resistance
        limits = schemas.resolve_per_dof(limit, names)
        for name in names:
            joint = stage.GetPrimAtPath(inventory[name])
            joint.CreateAttribute("newton:friction", Sdf.ValueTypeNames.Float).Set(1.0)
            prim = stage.DefinePrim(f"{articulation_prim_path}/{group}_{name}_actuator", "NewtonActuator")
            prim.SetActive(True)
            api = Sdf.TokenListOp()
            api.prependedItems = [BAM_DRIVE_API]
            prim.SetMetadata("apiSchemas", api)
            prim.CreateRelationship("newton:targets").SetTargets([Sdf.Path(inventory[name])])
            for key, value in {**attrs, "max_effort": float(limits[name])}.items():
                dtype = Sdf.ValueTypeNames.Int if isinstance(value, int) else Sdf.ValueTypeNames.Float
                prim.CreateAttribute("newton:" + to_camel_case(key), dtype).Set(value)


class BamActuatorControl(NewtonActuatorControl):
    """Extend GA's native-actuator binding with BAM's per-world state and solver channels."""

    def __init__(self, articulation):
        super().__init__(articulation)
        self._bam_cfgs = {}
        self._bam_group_dofs = {}

    def prepare_native_actuators(self, collection, actuator_cfgs):
        self._bam_model = SimulationManager.backend.model
        self._bam_cfgs = {name: cfg for name, cfg in actuator_cfgs.items() if isinstance(cfg, BamActuatorCfg)}
        if self._bam_cfgs and not self._articulation._sim_cfg.use_newton_actuators:
            raise ValueError("BAM requires use_newton_actuators=True with Newton MJWarp.")
        if self._bam_cfgs:
            for actuator in self._bam_model.actuators:
                if not isinstance(actuator.drive, DriveBam):
                    continue
                mapping = self._articulation._root_view._get_actuator_dof_mapping(actuator)
                if not (mapping.numpy() >= 0).any():
                    continue
                stride = len(actuator.indices) // self.num_instances
                if actuator.drive.env_dof_stride == stride:
                    continue
                if actuator.drive.startup_settings is not None:
                    raise ValueError("Articulations sharing a BAM drive must agree on its environment DOF stride.")
                actuator.drive.env_dof_stride = stride
                # A previous articulation may have created the shared adapter before
                # this drive's owner ran. Recreate its initial delay states with the
                # correct grouping, before any simulation steps or graph capture.
                adapter = SimulationManager._adapter
                if adapter is not None:
                    index = next(i for i, item in enumerate(adapter.actuators) if item is actuator)
                    adapter._states_a[index] = actuator.state()
                    adapter._states_b[index] = actuator.state()
        return super().prepare_native_actuators(collection, actuator_cfgs)

    def finalize_native_actuators(self, collection):
        result = super().finalize_native_actuators(collection)
        if not self._bam_cfgs:
            return result
        first_dof = self._joint_dof_offset()
        for name, cfg in self._bam_cfgs.items():
            ids = collection._group_joint_indices[name]
            self._articulation.write_joint_viscous_friction_coefficient_to_sim_index(
                joint_viscous_friction_coeff=cfg.motor.friction_viscous, joint_ids=ids
            )
            ids = list(range(self.num_joints))[ids] if isinstance(ids, slice) else ids.tolist()
            self._bam_group_dofs[name] = {first_dof + j for j in self._articulation.map_joint_ids_to_backend(ids)}
        self._sample_bam_startup_parameters()
        _pending_controls.append(self)
        return result

    def _bind_bam_actuators(self):
        solver = SimulationManager._solver
        if not MjWarpActuatorBridge.is_available(solver):
            raise ValueError("BAM requires Newton MJWarp with use_mujoco_cpu=False.")
        model = SimulationManager.backend.model
        for actuator, names in self._bam_actuators():
            drive = actuator.drive
            if drive.external_torque is not None:
                continue
            cfg = self._bam_cfgs[names[0]]
            if drive.has_backlash:
                self._bind_bam_backlash(actuator, cfg.backlash_joint_template)
            bridge = MjWarpActuatorBridge(solver, model, actuator.indices, self.device)
            drive.external_torque = wp.zeros(actuator.num_actuators, dtype=wp.float32, device=self.device)
            if cfg.stiff_frictionloss:
                bridge.stiffen_friction_constraint()
            # The drive gathers the previous solve's load before computing and
            # publishes the new friction budget immediately after computing.
            drive._ga_bridge = bridge

    def _bam_actuators(self) -> list[tuple[Actuator, list[str]]]:
        """Return the Newton BAM actuators that drive this articulation, each with the BAM groups it drives.

        Newton merges structurally identical actuators across groups and articulations, so an
        actuator belongs here when any of its DOFs belongs to one of this articulation's BAM groups.
        """
        adapter = SimulationManager._adapter
        if adapter is None:
            return []
        owned = []
        for actuator in adapter.actuators:
            if not isinstance(actuator.drive, DriveBam):
                continue
            local_dofs = set((actuator.indices.numpy() % adapter.num_joints).tolist())
            names = [name for name, dofs in self._bam_group_dofs.items() if dofs & local_dofs]
            if names:
                owned.append((actuator, names))
        return owned

    def _sample_bam_startup_parameters(self) -> None:
        """Sample the per-environment BAM supply voltage and sag gain once per Newton actuator.

        A USD prim is shared by every clone, so these ranges are sampled here and held across resets.

        Raises:
            ValueError: If BAM groups or articulations sharing a Newton actuator disagree on settings
                that Newton does not use to group actuators.
        """
        for actuator, names in self._bam_actuators():
            drive = actuator.drive
            cfgs = [self._bam_cfgs[name] for name in names]
            settings = {
                (c.vin_range, c.vin_drop_gain_range, c.stiff_frictionloss, c.backlash_joint_template) for c in cfgs
            }
            if len(settings) > 1:
                raise ValueError(
                    f"BAM groups {names} share a Newton actuator and must agree on 'vin_range',"
                    " 'vin_drop_gain_range', 'stiff_frictionloss' and 'backlash_joint_template'."
                )
            (setting,) = settings
            if drive.startup_settings == setting:
                # Another articulation sharing this drive already sampled it.
                continue
            if drive.startup_settings is not None:
                raise ValueError(
                    "Articulations sharing a Newton BAM actuator must agree on 'vin_range', 'vin_drop_gain_range',"
                    f" 'stiff_frictionloss' and 'backlash_joint_template' (got {drive.startup_settings} and {setting})."
                )
            drive.startup_settings = setting
            for values, value_range in ((drive.vin, cfgs[0].vin_range), (drive.sag_gain, cfgs[0].vin_drop_gain_range)):
                if value_range is None:
                    continue
                per_env = wp.to_torch(values).view(-1, drive.env_dof_stride)
                samples = torch.empty(per_env.shape[0], 1, device=per_env.device, dtype=per_env.dtype)
                per_env.copy_(samples.uniform_(*value_range).expand_as(per_env))

    def _bind_bam_backlash(self, actuator: Actuator, template: str) -> None:
        """Resolve each servo's sibling play hinge to its model coordinate index.

        Raises:
            ValueError: If a servo or its play-hinge sibling is missing or not revolute.
        """
        model = SimulationManager.backend.model
        q_start = model.joint_q_start.numpy()
        qd_start = model.joint_qd_start.numpy()
        joint_types = model.joint_type.numpy()
        worlds = model.joint_world.numpy()
        joints = {(int(world), label): index for index, (world, label) in enumerate(zip(worlds, model.joint_label))}
        dof_to_joint = {int(start): index for index, start in enumerate(qd_start[:-1])}
        indices = []
        for dof in actuator.indices.numpy():
            joint = dof_to_joint[int(dof)]
            parent, _, name = model.joint_label[joint].rpartition("/")
            twin_label = f"{parent}/{template.format(name)}"
            twin = joints.get((int(worlds[joint]), twin_label))
            if joint_types[joint] != JointType.REVOLUTE:
                raise ValueError(f"BAM backlash requires a revolute servo joint: {model.joint_label[joint]}")
            if twin is None or joint_types[twin] != JointType.REVOLUTE:
                raise ValueError(f"BAM backlash requires a revolute sibling play hinge: {twin_label}")
            indices.append(int(q_start[twin]))
        actuator.drive.backlash_pos_indices = wp.array(indices, dtype=wp.uint32, device=self.device)


def initialize_solver(cls):
    _original_initialize(cls)
    pending = list(_pending_controls)
    _pending_controls.clear()
    for control in pending:
        if control._bam_model is cls.backend.model:
            control._bind_bam_actuators()


def install_hooks():
    schemas._is_newton_native_actuator_cfg = lambda cfg: isinstance(cfg, BamActuatorCfg) or _original_supported(cfg)
    schemas.author_actuator_prims = author_actuator_prims
    schemas.define_actuator_properties = define_actuator_properties
    module = importlib.import_module("isaaclab_newton.assets.articulation.articulation")
    module.NewtonActuatorControl = BamActuatorControl
    SimulationManager.initialize_solver = classmethod(initialize_solver)
