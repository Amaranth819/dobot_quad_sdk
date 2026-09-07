#!/usr/bin/env python3
"""Deploy a Dobot policy after standing and waiting for Enter."""

import dds_middleware_python as dds
import time
import math
import os
import pickle
import select
import sys
from pathlib import Path
from threading import Event

import torch

if __package__:
    from .fake_env import DEFAULT_JOINT_POS, FakeEnvAgent
    from .history_wrapper import HistoryWrapper
else:
    from fake_env import DEFAULT_JOINT_POS, FakeEnvAgent
    from history_wrapper import HistoryWrapper

# Configuration parameters
NUM_MOTORS = 12
ABS2HW = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14]
MOTOR_OFFSET = [
    -0.05, -0.5, 1.17, 0.0, 0.05, -0.5, 1.17, 0.0, -0.05, 0.5, -1.17, 0.0, 0.05, 0.5, -1.17, 0.0
]

# Logical joint positions in hardware index order.
q_init = [0.0] * 16
q_init_count = 0
q_current = None


def stand_position(is_c2):
    if is_c2:
        # C2
        return [0.1, 0.7, -1.5, 0.0, 0.1, 0.7, -1.5, 0.0, 0.1, 0.7, -1.5, 0.0, 0.1, 0.7, -1.5, 0.0]
    else:
        # K4 standing pose from the RegRep training configuration.
        return [0.1, 0.7, -1.5, 0.0, -0.1, 0.7, -1.5, 0.0, 0.1, -0.7, 1.5, 0.0, -0.1, -0.7, 1.5, 0.0]
    
    
def laydown_position(is_c2):
    if is_c2:
        # C2
        return [0.1, 1.2, -2.4, 0.0, -0.1, 1.2, -2.4, 0.0, 0.1, 1.2, -2.4, 0.0, -0.1, 1.2, -2.4, 0.0]
    else:
        # K4
        return [0.1, 1.2, -2.4, 0.0, -0.1, 1.2, -2.4, 0.0, 0.1, -1.2, 2.4, 0.0, -0.1, -1.2, 2.4, 0.0]



def lower_state_callback(state):
    """Update current positions and collect the swing reference at startup."""
    global q_init_count, q_init, q_current

    motor_states = state.motor_state()
    positions = [0.0] * 16
    for hw in ABS2HW:
        # Subtract offset when reading to get real joint angle.
        positions[hw] = motor_states[hw].q() - MOTOR_OFFSET[hw]
    # Replace the whole snapshot so readers cannot see partially updated joints.
    q_current = positions

    if q_init_count < 10:
        q_init = positions.copy()
        q_init_count += 1
        if q_init_count == 10:
            print("Initial position collection completed: ", end="")
            for i in range(NUM_MOTORS):
                print(f"{q_init[ABS2HW[i]]:.4f} ", end="")
            print()


def create_damp_cmd():
    """Create damping mode command - using index accessor"""
    cmd = dds.LowerCmd()
    for i in range(NUM_MOTORS):
        hw = ABS2HW[i]
        cmd[hw].mode(0)
        cmd[hw].q(0.0 + MOTOR_OFFSET[hw])
        cmd[hw].dq(0.0)
        cmd[hw].tau(0.0)
        cmd[hw].kp(0.0)
        cmd[hw].kd(0.5)
    return cmd


def create_swing_cmd(elapsed, swing_frequency=1.0):
    """Create a swing sample at ``elapsed`` seconds from the start of swinging.

    ``swing_frequency`` sets the sinusoid's cycles per second; the publishing
    loop independently controls how often commands are sent (50 Hz in main).
    """
    if not math.isfinite(elapsed) or elapsed < 0.0:
        raise ValueError("elapsed must be a nonnegative, finite number of seconds")
    if not math.isfinite(swing_frequency) or swing_frequency <= 0.0:
        raise ValueError("swing_frequency must be a positive, finite number of Hz")

    displacement = math.sin(2.0 * math.pi * swing_frequency * elapsed) * 0.2
    cmd = dds.LowerCmd()
    for i in range(NUM_MOTORS):
        hw = ABS2HW[i]
        qdes = q_init[hw] + displacement + MOTOR_OFFSET[hw]
        cmd[hw].mode(0)
        cmd[hw].q(qdes)
        cmd[hw].dq(0.0)
        cmd[hw].tau(0.0)
        cmd[hw].kp(30.0)
        cmd[hw].kd(1.2)
    return cmd


def create_init_cmd(duration=2.0, frequency=50.0):
    """Return a trajectory from the latest measured pose to DEFAULT_JOINT_POS.

    Every controlled motor uses kp=25.0 and kd=0.6.
    See ``create_position_cmds`` for the command sampling schedule.
    """
    return create_position_cmds(DEFAULT_JOINT_POS, duration, frequency, kp=25.0, kd=0.6)


def create_laydown_cmd(duration=2.0, frequency=50.0, is_c2=False):
    """Return a command trajectory from the latest measured pose to lying down."""
    return create_position_cmds(laydown_position(is_c2), duration, frequency)


def create_position_cmds(q_target, duration=2.0, frequency=50.0, *, kp=30.0, kd=1.2):
    """Return a trajectory to logical joint targets in hardware index order.

    Snapshot the current joint positions once, then sample a smooth trajectory
    at ``frequency`` Hz. Publish command ``i`` at elapsed time
    ``min(i / frequency, duration)``. Both endpoints are included, so the list
    has ``ceil(duration * frequency) + 1`` commands; the last interval may be
    shorter than one period. ``kp`` and ``kd`` apply to the twelve controlled
    motors; unused slots have zero gains. This function only builds commands.
    """
    if not math.isfinite(duration) or duration <= 0.0:
        raise ValueError("duration must be a positive, finite number of seconds")
    if not math.isfinite(frequency) or frequency <= 0.0:
        raise ValueError("frequency must be a positive, finite number of Hz")
    if not math.isfinite(kp) or kp < 0.0 or not math.isfinite(kd) or kd < 0.0:
        raise ValueError("kp and kd must be finite and nonnegative")
    if q_current is None:
        raise RuntimeError("Wait for a joint state before generating the trajectory")

    q_start = q_current.copy()
    if not all(math.isfinite(q_start[hw]) for hw in ABS2HW):
        raise ValueError("Current joint positions must be finite")
    q_target = list(q_target)
    if len(q_target) != 16 or not all(math.isfinite(q_target[hw]) for hw in ABS2HW):
        raise ValueError("Target pose must contain 16 entries with finite motor positions")
    num_intervals = max(1, math.ceil(duration * frequency))
    commands = []

    for step in range(num_intervals + 1):
        elapsed = min(step / frequency, duration)
        progress = elapsed / duration
        # Cubic interpolation starts and ends with zero desired velocity.
        blend = progress * progress * (3.0 - 2.0 * progress)
        blend_rate = 6.0 * progress * (1.0 - progress) / duration

        cmd = dds.LowerCmd()
        for hw in (3, 7, 11, 15):
            cmd[hw].mode(0)
            cmd[hw].q(0.0)
            cmd[hw].dq(0.0)
            cmd[hw].tau(0.0)
            cmd[hw].kp(0.0)
            cmd[hw].kd(0.0)
        for hw in ABS2HW:
            qdes = (1.0 - blend) * q_start[hw] + blend * q_target[hw]
            dqdes = blend_rate * (q_target[hw] - q_start[hw])
            cmd[hw].mode(0)
            cmd[hw].q(qdes + MOTOR_OFFSET[hw])
            cmd[hw].dq(dqdes)
            cmd[hw].tau(0.0)
            cmd[hw].kp(kp)
            cmd[hw].kd(kd)
        commands.append(cmd)
    return commands


def hold_position_until_enter(middleware, cmd, frequency=50.0):
    """Keep publishing the supplied standing targets and gains until terminal Enter."""
    if not math.isfinite(frequency) or frequency <= 0.0:
        raise ValueError("frequency must be a positive, finite number of Hz")

    print("Holding stand position. Press Enter to start the policy test.", flush=True)
    dt = 1.0 / frequency
    next_publish_time = time.monotonic()
    while True:
        if time.monotonic() >= next_publish_time:
            middleware.publishLowerCmd(cmd)
            next_publish_time += dt

        delay = max(0.0, next_publish_time - time.monotonic())
        readable, _, _ = select.select([sys.stdin], [], [], delay)
        if readable:
            # Read only available bytes; a partial input line must not block control.
            data = os.read(sys.stdin.fileno(), 4096)
            if not data:
                raise EOFError("Standard input closed while holding the stand position")
            if b"\n" in data:
                return


@torch.inference_mode()
def warm_up_actor(actor, num_obs_history, num_actions):
    """Finish initial TorchScript inference work before starting motor control."""
    history = torch.zeros(1, num_obs_history, dtype=torch.float32, device="cpu")
    for _ in range(5):
        actions = actor(history)
        if actions.shape != (1, num_actions) or not torch.isfinite(actions).all():
            raise RuntimeError("Policy warm-up must produce finite actions with shape (1, num_actions)")


def main():
    # Load the exported actor; resolve the path relative to this script.
    deploy_dir = Path(__file__).resolve().parent
    policy_dir = deploy_dir / "policy/regrep"
    policy_path = policy_dir / "checkpoints/actor_jit_004999.pt"
    print(f"Loading policy from {policy_path}")
    actor = torch.jit.load(str(policy_path), map_location="cpu").eval()

    @torch.inference_mode()
    def policy(obs, info=None):
        history = obs["obs_history"].to(device="cpu", dtype=torch.float32)
        return actor(history)

    # Use the environment configuration saved alongside this checkpoint.
    with (policy_dir / "parameters_cpu.pkl").open("rb") as file:
        cfg = pickle.load(file)["Cfg"]

    print("Warming up policy before motor control...")
    warm_up_actor(
        actor,
        cfg["env"]["num_observations"] * cfg["env"]["num_observation_history"],
        cfg["env"]["num_actions"],
    )

    # Create DDS middleware
    middleware = dds.PyDDSMiddleware(str(deploy_dir / "config/dds_config.yaml"))

    # QoS configuration
    qos_config = {
        "reliability": "reliable",
        "history_kind": "keep_last",
        "history_depth": 1,
        "durability": "volatile"
    }
    
    # Initialize the environment with its command writer and training history length.
    middleware.createLowerCmdWriter("rt/lower/cmd", qos_config)
    fake_env = FakeEnvAgent(cfg, middleware=middleware)
    env = HistoryWrapper(fake_env, history_length=cfg["env"]["num_observation_history"])
    dt = env.dt
    frequency = 1.0 / dt

    state_ready = Event()

    def on_lower_state(state):
        env.lower_state_callback(state)
        lower_state_callback(state)
        state_ready.set()

    middleware.subscribeLowerState("rt/lower/state", on_lower_state)
    print("Waiting for the first robot state...")
    state_ready.wait()

    # Move to the initial standing position over 2 seconds.
    print("Moving to initial standing position over 2 seconds.")
    stand_duration = 2.0
    stand_cmds = create_init_cmd(stand_duration, frequency=frequency)
    stand_start_time = time.monotonic()
    for idx, cmd in enumerate(stand_cmds):
        publish_time = stand_start_time + min(idx * dt, stand_duration)
        delay = publish_time - time.monotonic()
        if delay > 0.0:
            time.sleep(delay)
        middleware.publishLowerCmd(cmd)

    # Apply the final standing target for one control period before policy startup.
    time.sleep(dt)
    hold_position_until_enter(middleware, stand_cmds[-1], frequency=frequency)

    env.set_commands(vx=0.0, vy=0.0, yaw_rate=0.0)
    env.reset()
    # Zero policy actions hold DEFAULT_JOINT_POS with the same standing gains.
    print("Collecting standing observations before starting the policy...")
    standing_action = torch.zeros(env.num_envs, env.num_actions, device=env.device)
    for _ in range(env.obs_history_length):
        observations, _, _, _ = env.step(standing_action)

    print("Starting policy gradually toward vx=0.3 m/s, vy=0, yaw_rate=0...")
    startup_duration = 1.0
    clip_actions = fake_env.cfg["normalization"]["clip_actions"]
    for step in range(250):
        progress = min(step * dt / startup_duration, 1.0)
        blend = progress * progress * (3.0 - 2.0 * progress)
        env.set_commands(vx=0.3 * blend, vy=0.0, yaw_rate=0.0)
        # Refresh the newest frame with this command without shifting history twice.
        observations = env.get_obs()
        actions = policy(observations).clamp(-clip_actions, clip_actions) * blend
        # FakeEnv publishes the motor command and enforces the control period.
        # Passing blended actions also keeps the two previous-action observations accurate.
        observations, _, _, _ = env.step(actions)

    print("Policy test completed. Moving to laydown position over 2 seconds.")
    laydown_duration = 2.0
    laydown_cmds = create_laydown_cmd(laydown_duration, frequency=frequency)
    laydown_start_time = time.monotonic()
    for idx, cmd in enumerate(laydown_cmds):
        publish_time = laydown_start_time + min(idx * dt, laydown_duration)
        delay = publish_time - time.monotonic()
        if delay > 0.0:
            time.sleep(delay)
        middleware.publishLowerCmd(cmd)

    print("Control sequence completed. Maintaining damping mode; press Ctrl+C to stop.")
    damp_cmd = create_damp_cmd()
    # Give the final laydown target one control period before switching to damping.
    next_publish_time = laydown_start_time + laydown_duration + dt
    try:
        while True:
            delay = next_publish_time - time.monotonic()
            if delay > 0.0:
                time.sleep(delay)
            middleware.publishLowerCmd(damp_cmd)
            next_publish_time += dt
    except KeyboardInterrupt:
        print("Damping publisher stopped.")


if __name__ == "__main__":
    main()
