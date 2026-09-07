"""Build MS-PPO observations from Dobot LowerState feedback."""

import copy
import time

import numpy as np
import torch


# FL, FR, RL, RR; abad, thigh, calf within each leg.
ABS2HW = np.array([0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14])
MOTOR_OFFSET = np.array([
    -0.05, -0.5, 1.17, 0.0,
    0.05, -0.5, 1.17, 0.0,
    -0.05, 0.5, -1.17, 0.0,
    0.05, 0.5, -1.17, 0.0,
], dtype=np.float32)
# Logical K4 standing pose from the RegRep checkpoint's training configuration.
DEFAULT_JOINT_POS = np.array([
    0.1, 0.7, -1.5, 0.0,
    -0.1, 0.7, -1.5, 0.0,
    0.1, -0.7, 1.5, 0.0,
    -0.1, -0.7, 1.5, 0.0,
], dtype=np.float32)


def class_to_dict(obj) -> dict:
    if not hasattr(obj, "__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_") or key == "terrain":
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result


class FakeEnvAgent:
    """One robot with 58 observations in the MS-PPO training layout.

    Register ``lower_state_callback`` with
    ``middleware.subscribeLowerState("rt/lower/state", env.lower_state_callback)``.
    Alternatively, ``se`` may supply projected gravity and the twelve already
    offset-corrected joint positions/velocities in FL, FR, RL, RR order.
    An MS-PPO command profile supplies [vx, vy, yaw_rate, ...]; without a profile,
    use ``set_commands``. Reading observations does not advance policy time.
    For actions, pass a DDS ``middleware`` whose ``rt/lower/cmd`` writer has
    already been created. Policy actions have 12 entries; motor commands have 16.
    """

    def __init__(self, cfg, se=None, command_profile=None, middleware=None):
        if not isinstance(cfg, dict):
            cfg = class_to_dict(cfg)
        self.cfg = copy.deepcopy(cfg)
        self.se = se
        self.command_profile = command_profile
        self.middleware = middleware

        self.dt = self.cfg["control"]["decimation"] * self.cfg["sim"]["dt"]
        self.cfg["control"].setdefault("action_scale", 0.25)
        self.cfg["control"].setdefault("hip_scale_reduction", 0.5)
        self.timestep = 0
        self.time = time.time()

        self.num_obs = 58
        self.num_envs = 1
        self.num_privileged_obs = 2
        self.num_actions = 12
        self.num_motors = 16
        self.num_commands = 3
        self.device = 'cpu'

        env_cfg = self.cfg.setdefault("env", {})
        env_cfg.update(
            num_observations=58, num_scalar_observations=58,
            num_privileged_obs=2, num_actions=12,
            observe_command=True, observe_two_prev_actions=True,
            observe_clock_inputs=True, observe_timing_parameter=False,
            observe_vel=False, observe_only_lin_vel=False,
            observe_only_ang_vel=False, observe_yaw=False, observe_contact_states=False,
        )
        env_cfg.setdefault("num_observation_history", 30)

        # Effective observation scales and clipping from MS-PPO scripts/train.py.
        self.obs_scales = dict(lin_vel=2.0, ang_vel=0.25, dof_pos=1.0, dof_vel=0.05)
        self.cfg.setdefault("obs_scales", {}).update(self.obs_scales)
        self.cfg.setdefault("normalization", {}).update(clip_actions=10.0, clip_observations=100.0)
        self.commands_scale = np.array([2.0, 2.0, 0.25], dtype=np.float32)
        command_cfg = self.cfg.setdefault("commands", {})
        command_cfg["num_commands"] = self.num_commands
        self.cmd_indices = list(command_cfg.get("cmd_indices", [0, 1, 2]))
        if sorted(self.cmd_indices) != [0, 1, 2]:
            raise ValueError("commands.cmd_indices must be a permutation of [0, 1, 2]")
        command_cfg["cmd_indices"] = self.cmd_indices

        joint_names = [
            f"joint_{leg}_{joint}"
            for leg in ("front_left", "front_right", "rear_left", "rear_right")
            for joint in ("abad", "thigh_pitch", "calf_pitch")
        ]
        self.default_dof_pos = DEFAULT_JOINT_POS[ABS2HW].copy()
        self.cfg.setdefault("init_state", {})["default_joint_angles"] = dict(
            zip(joint_names, self.default_dof_pos.tolist())
        )

        self.p_gains = np.zeros(12)
        self.d_gains = np.zeros(12)
        for i in range(12):
            joint_name = joint_names[i]
            found = False
            for dof_name in self.cfg["control"]["stiffness"].keys():
                if dof_name in joint_name:
                    self.p_gains[i] = self.cfg["control"]["stiffness"][dof_name]
                    self.d_gains[i] = self.cfg["control"]["damping"][dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg["control"]["control_type"] in ["P", "V"]:
                    print(f"PD gain of joint {joint_name} were not defined, setting them to zero")

        self.commands = np.zeros((1, self.num_commands), dtype=np.float32)
        self.actions = torch.zeros(1, 12, dtype=torch.float32)
        self.last_actions = torch.zeros_like(self.actions)
        self.gravity_vector = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        self.dof_pos = self.default_dof_pos.copy()
        self.dof_vel = np.zeros(12, dtype=np.float32)
        self._state_snapshot = None
        self.body_linear_vel = np.zeros(3)
        self.body_angular_vel = np.zeros(3)
        self.joint_pos_target = np.zeros(12)
        self.joint_vel_target = np.zeros(12)
        self.motor_pos_target = np.zeros(self.num_motors, dtype=np.float32)
        self.torques = np.zeros(12)
        self.contact_state = np.ones(4)

        # Fixed training gait; these are independent of the three velocity commands.
        self.gait_frequency = 3.0
        self.gait_phase = 0.5
        self.gait_offset = 0.0
        self.gait_bound = 0.0
        self.gait_duration = 0.5
        self.gait_indices = torch.zeros(self.num_envs, dtype=torch.float)
        self.foot_indices = torch.zeros(self.num_envs, 4, dtype=torch.float)
        self.clock_inputs = torch.zeros(self.num_envs, 4, dtype=torch.float)

        self.is_currently_probing = False

    def set_probing(self, is_currently_probing):
        self.is_currently_probing = is_currently_probing

    def lower_state_callback(self, state):
        """Cache every DDS sample, with no printing throttle or motor commands.

        The SDK quaternion is [w, x, y, z]. Treat it as the body-to-world
        orientation, with IMU axes aligned to the policy's body axes.
        """
        quat = np.asarray(state.imu_state().quaternion(), dtype=np.float64)
        if quat.shape != (4,) or not np.all(np.isfinite(quat)):
            raise ValueError("IMU quaternion must contain four finite values in wxyz order")
        norm = np.linalg.norm(quat)
        if norm < 1e-8:
            raise ValueError("IMU quaternion must have nonzero norm")
        w, x, y, z = quat / norm
        # R(body->world).T @ [0, 0, -1]: unit projected gravity, not acceleration.
        gravity = np.array([
            2 * (w * y - x * z),
            -2 * (w * x + y * z),
            2 * (x * x + y * y) - 1,
        ], dtype=np.float32)
        motors = state.motor_state()
        dof_pos = np.array([motors[hw].q() for hw in ABS2HW], dtype=np.float32)
        dof_pos -= MOTOR_OFFSET[ABS2HW]
        dof_vel = np.array([motors[hw].dq() for hw in ABS2HW], dtype=np.float32)
        if not np.all(np.isfinite(dof_pos)) or not np.all(np.isfinite(dof_vel)):
            raise ValueError("Joint positions and velocities must be finite")
        # Swap a complete sample so get_obs cannot mix two callback updates.
        self._state_snapshot = (gravity, dof_pos, dof_vel)

    def set_commands(self, vx, vy, yaw_rate):
        """Set physical velocity commands when no command profile is supplied."""
        self.commands[0] = (vx, vy, yaw_rate)

    def get_obs(self):
        """Return float32 (1, 58): gravity, commands, q-q0, dq, a[-1], a[-2], clocks."""
        snapshot = self._state_snapshot
        if snapshot is not None:
            self.gravity_vector, self.dof_pos, self.dof_vel = snapshot
        elif self.se is not None:
            self.gravity_vector = np.asarray(self.se.get_gravity_vector(), dtype=np.float32)
            self.dof_pos = np.asarray(self.se.get_dof_pos(), dtype=np.float32)
            self.dof_vel = np.asarray(self.se.get_dof_vel(), dtype=np.float32)
        else:
            raise RuntimeError("No LowerState received; register lower_state_callback and wait for feedback")

        if self.command_profile is not None:
            cmds, reset_timer = self.command_profile.get_command(
                self.timestep * self.dt, probe=self.is_currently_probing
            )
            cmds = torch.as_tensor(cmds).detach().cpu().numpy().reshape(-1)
            if cmds.size < 3:
                raise ValueError("Command profile must provide vx, vy, and yaw_rate")
            self.commands[0] = cmds[:3]
            if reset_timer:
                self.reset_gait_indices()

        clip_actions = self.cfg["normalization"]["clip_actions"]
        ob = np.concatenate((
            self.gravity_vector.reshape(1, 3),
            (self.commands * self.commands_scale)[:, self.cmd_indices],
            (self.dof_pos - self.default_dof_pos).reshape(1, 12) * self.obs_scales["dof_pos"],
            self.dof_vel.reshape(1, 12) * self.obs_scales["dof_vel"],
            self.actions.clamp(-clip_actions, clip_actions).detach().cpu().numpy().reshape(1, 12),
            self.last_actions.clamp(-clip_actions, clip_actions).detach().cpu().numpy().reshape(1, 12),
            self.clock_inputs.cpu().numpy(),
        ), axis=1)
        clip_obs = self.cfg["normalization"]["clip_observations"]
        return torch.as_tensor(np.clip(ob, -clip_obs, clip_obs), dtype=torch.float32,
                               device=self.device)

    def get_observations(self):
        return self.get_obs()

    def get_privileged_observations(self):
        return None

    def _prepare_action(self, action):
        action = torch.as_tensor(action, dtype=torch.float32, device=self.device).detach()
        if action.shape not in ((12,), (1, 12)):
            raise ValueError("Policy action must have shape (12,) or (1, 12)")
        if not torch.isfinite(action).all():
            raise ValueError("Policy action must contain finite values")
        clip_actions = self.cfg["normalization"]["clip_actions"]
        return action.reshape(1, 12).clamp(-clip_actions, clip_actions)

    def action_to_motor_targets(self, action):
        """Scale 12 policy offsets and expand to 16 hardware position targets.

        Hardware slots 3, 7, 11 and 15 are unused on the legged robot and stay zero.
        This only builds targets; it does not publish or advance action history.
        """
        action = self._prepare_action(action).cpu().numpy().reshape(12)
        self.joint_pos_target = action * self.cfg["control"]["action_scale"]
        self.joint_pos_target[[0, 3, 6, 9]] *= self.cfg["control"]["hip_scale_reduction"]
        self.joint_pos_target += self.default_dof_pos
        self.joint_vel_target = np.zeros(12, dtype=np.float32)
        self.motor_pos_target = np.zeros(self.num_motors, dtype=np.float32)
        self.motor_pos_target[ABS2HW] = self.joint_pos_target + MOTOR_OFFSET[ABS2HW]
        return self.motor_pos_target.copy()

    def build_motor_command(self, action):
        """Build a 16-slot Dobot LowerCmd without sending it to the robot."""
        import dds_middleware_python as dds

        targets = self.action_to_motor_targets(action)
        kp = np.zeros(self.num_motors)
        kd = np.zeros(self.num_motors)
        kp[ABS2HW] = self.p_gains
        kd[ABS2HW] = self.d_gains
        cmd = dds.LowerCmd()
        for hw in range(self.num_motors):
            cmd[hw].mode(0)
            cmd[hw].q(float(targets[hw]))
            cmd[hw].dq(0.0)
            cmd[hw].tau(0.0)
            cmd[hw].kp(float(kp[hw]))
            cmd[hw].kd(float(kd[hw]))
        return cmd

    def publish_action(self, action, hard_reset=False):
        if hard_reset:
            raise NotImplementedError("Dobot LowerCmd does not support the legacy LCM hard_reset")
        if self.middleware is None:
            raise RuntimeError("Pass a DDS middleware with an rt/lower/cmd writer to publish actions")
        cmd = self.build_motor_command(action)
        self.middleware.publishLowerCmd(cmd)
        self.torques = ((self.joint_pos_target - self.dof_pos) * self.p_gains
                        + (self.joint_vel_target - self.dof_vel) * self.d_gains)

    def reset(self):
        self.actions.zero_()
        self.last_actions.zero_()
        self.reset_gait_indices()
        self.time = time.time()
        self.timestep = 0
        return self.get_obs()

    def reset_idx(self, env_ids):
        """Support HistoryWrapper's indexed reset for the single hardware robot."""
        env_ids = torch.as_tensor(env_ids, device=self.device).reshape(-1)
        if env_ids.numel() == 0:
            return None
        if torch.any(env_ids != 0):
            raise ValueError("FakeEnvAgent has only one environment, with index 0")
        return self.reset()

    def reset_gait_indices(self):
        self.gait_indices.zero_()
        self.foot_indices.zero_()
        self.clock_inputs.zero_()

    def _advance_gait(self):
        """Advance once per policy step, before constructing its returned observation."""
        self.gait_indices = torch.remainder(
            self.gait_indices + self.dt * self.gait_frequency, 1.0
        )
        phase, offset, bound = self.gait_phase, self.gait_offset, self.gait_bound
        foot_offsets = [phase + offset + bound, offset, bound, phase]
        if self.cfg["commands"].get("pacing_offset", False):
            foot_offsets[1], foot_offsets[2] = foot_offsets[2], foot_offsets[1]
        self.foot_indices = torch.remainder(
            self.gait_indices[:, None] + torch.tensor(foot_offsets, dtype=torch.float32), 1.0
        )
        duration = self.gait_duration
        clock_phase = torch.where(
            self.foot_indices < duration,
            self.foot_indices * (0.5 / duration),
            0.5 + (self.foot_indices - duration) * (0.5 / (1.0 - duration)),
        )
        self.clock_inputs = torch.sin(2 * torch.pi * clock_phase)

    def step(self, actions, hard_reset=False):
        actions = self._prepare_action(actions)
        self.publish_action(actions, hard_reset=hard_reset)
        self.last_actions = self.actions.clone()
        self.actions = actions
        time.sleep(max(self.dt - (time.time() - self.time), 0))
        if self.timestep % 100 == 0: print(f'frq: {1 / (time.time() - self.time)} Hz')
        self.time = time.time()
        self.timestep += 1
        self._advance_gait()
        obs = self.get_obs()

        infos = {"joint_pos": self.dof_pos[np.newaxis, :],
                 "joint_vel": self.dof_vel[np.newaxis, :],
                 "joint_pos_target": self.joint_pos_target[np.newaxis, :],
                 "joint_vel_target": self.joint_vel_target[np.newaxis, :],
                 "motor_pos_target": self.motor_pos_target[np.newaxis, :],
                 "body_linear_vel": self.body_linear_vel[np.newaxis, :],
                 "body_angular_vel": self.body_angular_vel[np.newaxis, :],
                 "contact_state": self.contact_state[np.newaxis, :],
                 "clock_inputs": self.clock_inputs[np.newaxis, :],
                 "body_linear_vel_cmd": self.commands[:, 0:2],
                 "body_angular_vel_cmd": self.commands[:, 2:],
                 "privileged_obs": None,
                 }

        return obs, None, None, infos
