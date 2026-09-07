"""Collect FakeEnv observations into the flattened history used by MS-PPO."""

from numbers import Integral

import torch


class HistoryWrapper:
    """Wrap ``FakeEnvAgent`` and return obs, privileged_obs and obs_history.

    History is ordered from oldest to newest. Each successful step appends one
    frame. Repeated observation reads at the same FakeEnv timestep refresh the
    newest frame without shifting history. Reset returns all-zero history, as in
    MS-PPO training; the first subsequent read or step adds the first frame.
    ``history_length`` sets the number of frames and defaults to 30.
    """

    def __init__(self, env, history_length=30):
        self.env = env

        self.obs_history_length = history_length
        if not isinstance(self.obs_history_length, Integral) or self.obs_history_length < 1:
            raise ValueError("history_length must be a positive integer")
        self.num_obs_history = self.obs_history_length * self.env.num_obs
        self.obs_history = torch.zeros(
            self.env.num_envs, self.num_obs_history, dtype=torch.float32,
            device=self.env.device, requires_grad=False,
        )
        self.num_privileged_obs = self.env.num_privileged_obs
        self._last_history_timestep = None

    def _observations(self, obs, privileged_obs, append):
        expected_shape = (self.env.num_envs, self.env.num_obs)
        if tuple(obs.shape) != expected_shape:
            raise ValueError(f"Expected observation shape {expected_shape}, got {tuple(obs.shape)}")
        obs = obs.detach().to(device=self.env.device, dtype=torch.float32)
        if append:
            retained = self.obs_history[:, self.env.num_obs:]
        else:
            retained = self.obs_history[:, :-self.env.num_obs]
        # Allocate a new tensor so previously returned histories remain snapshots.
        self.obs_history = torch.cat((retained, obs), dim=-1)
        self._last_history_timestep = getattr(self.env, "timestep", None)
        return {"obs": obs, "privileged_obs": privileged_obs, "obs_history": self.obs_history}

    def step(self, action, **kwargs):
        obs, rew, done, info = self.env.step(action, **kwargs)
        privileged_obs = info["privileged_obs"]
        return self._observations(obs, privileged_obs, append=True), rew, done, info

    def get_observations(self):
        return self.get_obs()

    def get_obs(self):
        obs = self.env.get_obs()
        privileged_obs = self.env.get_privileged_observations()
        timestep = getattr(self.env, "timestep", None)
        append = timestep is None or timestep != self._last_history_timestep
        return self._observations(obs, privileged_obs, append=append)

    def reset_idx(self, env_ids):
        ret = self.env.reset_idx(env_ids)
        self.obs_history = self.obs_history.clone()
        self.obs_history[env_ids, :] = 0
        if len(env_ids):
            self._last_history_timestep = None
        return ret

    def reset(self):
        ret = self.env.reset()
        privileged_obs = self.env.get_privileged_observations()
        self.obs_history = torch.zeros_like(self.obs_history)
        self._last_history_timestep = None
        return {"obs": ret, "privileged_obs": privileged_obs, "obs_history": self.obs_history}

    def __getattr__(self, name):
        return getattr(self.env, name)
