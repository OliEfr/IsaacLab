# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Action manager for processing actions sent to the environment."""

from __future__ import annotations

import inspect
import torch
import weakref
from abc import abstractmethod
from collections.abc import Sequence
from prettytable import PrettyTable
from typing import TYPE_CHECKING

import omni.kit.app

from omni.isaac.lab.assets import AssetBase

from .manager_base import ManagerBase, ManagerTermBase
from .manager_term_cfg import ActionTermCfg

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedEnv


class ActionTerm(ManagerTermBase):
    """Base class for action terms.

    The action term is responsible for processing the raw actions sent to the environment
    and applying them to the asset managed by the term. The action term is comprised of two
    operations:

    * Processing of actions: This operation is performed once per **environment step** and
      is responsible for pre-processing the raw actions sent to the environment.
    * Applying actions: This operation is performed once per **simulation step** and is
      responsible for applying the processed actions to the asset managed by the term.
    """

    def __init__(self, cfg: ActionTermCfg, env: ManagerBasedEnv):
        """Initialize the action term.

        Args:
            cfg: The configuration object.
            env: The environment instance.
        """
        # call the base class constructor
        super().__init__(cfg, env)
        # parse config to obtain asset to which the term is applied
        self._asset: AssetBase = self._env.scene[self.cfg.asset_name]

        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self._debug_vis_handle = None
        # set initial state of debug visualization
        self.set_debug_vis(self.cfg.debug_vis)

    def __del__(self):
        """Unsubscribe from the callbacks."""
        if self._debug_vis_handle:
            self._debug_vis_handle.unsubscribe()
            self._debug_vis_handle = None

    """
    Properties.
    """

    @property
    @abstractmethod
    def action_dim(self) -> int:
        """Dimension of the action term."""
        raise NotImplementedError

    @property
    @abstractmethod
    def raw_actions(self) -> torch.Tensor:
        """The input/raw actions sent to the term."""
        raise NotImplementedError

    @property
    @abstractmethod
    def processed_actions(self) -> torch.Tensor:
        """The actions computed by the term after applying any processing."""
        raise NotImplementedError

    @property
    def has_debug_vis_implementation(self) -> bool:
        """Whether the action term has a debug visualization implemented."""
        # check if function raises NotImplementedError
        source_code = inspect.getsource(self._set_debug_vis_impl)
        return "NotImplementedError" not in source_code

    """
    Operations.
    """

    def set_debug_vis(self, debug_vis: bool) -> bool:
        """Sets whether to visualize the action term data.
        Args:
            debug_vis: Whether to visualize the action term data.
        Returns:
            Whether the debug visualization was successfully set. False if the action term does
            not support debug visualization.
        """
        # check if debug visualization is supported
        if not self.has_debug_vis_implementation:
            return False
        # toggle debug visualization objects
        self._set_debug_vis_impl(debug_vis)
        # toggle debug visualization handles
        if debug_vis:
            # create a subscriber for the post update event if it doesn't exist
            if self._debug_vis_handle is None:
                app_interface = omni.kit.app.get_app_interface()
                self._debug_vis_handle = app_interface.get_post_update_event_stream().create_subscription_to_pop(
                    lambda event, obj=weakref.proxy(self): obj._debug_vis_callback(event)
                )
        else:
            # remove the subscriber if it exists
            if self._debug_vis_handle is not None:
                self._debug_vis_handle.unsubscribe()
                self._debug_vis_handle = None
        # return success
        return True

    @abstractmethod
    def process_actions(self, actions: torch.Tensor):
        """Processes the actions sent to the environment.

        Note:
            This function is called once per environment step by the manager.

        Args:
            actions: The actions to process.
        """
        raise NotImplementedError

    @abstractmethod
    def apply_actions(self):
        """Applies the actions to the asset managed by the term.

        Note:
            This is called at every simulation step by the manager.
        """
        raise NotImplementedError

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Set debug visualization into visualization objects.
        This function is responsible for creating the visualization objects if they don't exist
        and input ``debug_vis`` is True. If the visualization objects exist, the function should
        set their visibility into the stage.
        """
        raise NotImplementedError(f"Debug visualization is not implemented for {self.__class__.__name__}.")

    def _debug_vis_callback(self, event):
        """Callback for debug visualization.
        This function calls the visualization objects and sets the data to visualize into them.
        """
        raise NotImplementedError(f"Debug visualization is not implemented for {self.__class__.__name__}.")


class ActionManager(ManagerBase):
    """Manager for processing and applying actions for a given world.

    The action manager handles the interpretation and application of user-defined
    actions on a given world. It is comprised of different action terms that decide
    the dimension of the expected actions.

    The action manager performs operations at two stages:

    * processing of actions: It splits the input actions to each term and performs any
      pre-processing needed. This should be called once at every environment step.
    * apply actions: This operation typically sets the processed actions into the assets in the
      scene (such as robots). It should be called before every simulation step.
    """

    def __init__(self, cfg: object, env: ManagerBasedEnv):
        """Initialize the action manager.

        Args:
            cfg: The configuration object or dictionary (``dict[str, ActionTermCfg]``).
            env: The environment instance.

        Raises:
            ValueError: If the configuration is None.
        """
        # check if config is None
        if cfg is None:
            raise ValueError("Action manager configuration is None. Please provide a valid configuration.")

        # call the base class constructor (this prepares the terms)
        super().__init__(cfg, env)
        # create buffers to store actions
        self._action = torch.zeros((self.num_envs, self.total_action_dim), device=self.device)
        self._prev_action = torch.zeros_like(self._action)

        # check if any term has debug visualization implemented
        self.cfg.debug_vis = False
        for term in self._terms.values():
            self.cfg.debug_vis |= term.cfg.debug_vis

    def __str__(self) -> str:
        """Returns: A string representation for action manager."""
        msg = f"<ActionManager> contains {len(self._term_names)} active terms.\n"

        # create table for term information
        table = PrettyTable()
        table.title = f"Active Action Terms (shape: {self.total_action_dim})"
        table.field_names = ["Index", "Name", "Dimension"]
        # set alignment of table columns
        table.align["Name"] = "l"
        table.align["Dimension"] = "r"
        # add info on each term
        for index, (name, term) in enumerate(self._terms.items()):
            table.add_row([index, name, term.action_dim])
        # convert table to string
        msg += table.get_string()
        msg += "\n"

        return msg

    """
    Properties.
    """

    @property
    def total_action_dim(self) -> int:
        """Total dimension of actions."""
        return sum(self.action_term_dim)

    @property
    def active_terms(self) -> list[str]:
        """Name of active action terms."""
        return self._term_names

    @property
    def action_term_dim(self) -> list[int]:
        """Shape of each action term."""
        return [term.action_dim for term in self._terms.values()]

    @property
    def action(self) -> torch.Tensor:
        """The actions sent to the environment. Shape is (num_envs, total_action_dim)."""
        return self._action

    @property
    def prev_action(self) -> torch.Tensor:
        """The previous actions sent to the environment. Shape is (num_envs, total_action_dim)."""
        return self._prev_action

    @property
    def has_debug_vis_implementation(self) -> bool:
        """Whether the command terms have debug visualization implemented."""
        # check if function raises NotImplementedError
        has_debug_vis = False
        for term in self._terms.values():
            has_debug_vis |= term.has_debug_vis_implementation
        return has_debug_vis

    """
    Operations.
    """

    def set_debug_vis(self, debug_vis: bool) -> bool:
        """Sets whether to visualize the action data.
        Args:
            debug_vis: Whether to visualize the action data.
        Returns:
            Whether the debug visualization was successfully set. False if the action
            does not support debug visualization.
        """
        for term in self._terms.values():
            term.set_debug_vis(debug_vis)

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        """Resets the action history.

        Args:
            env_ids: The environment ids. Defaults to None, in which case
                all environments are considered.

        Returns:
            An empty dictionary.
        """
        # resolve environment ids
        if env_ids is None:
            env_ids = slice(None)
        # reset the action history
        self._prev_action[env_ids] = 0.0
        self._action[env_ids] = 0.0
        # reset all action terms
        for term in self._terms.values():
            term.reset(env_ids=env_ids)
        # nothing to log here
        return {}

    def process_action(self, action: torch.Tensor):
        """Processes the actions sent to the environment.

        Note:
            This function should be called once per environment step.

        Args:
            action: The actions to process.
        """
        # check if action dimension is valid
        if self.total_action_dim != action.shape[1]:
            raise ValueError(f"Invalid action shape, expected: {self.total_action_dim}, received: {action.shape[1]}.")
        # store the input actions
        self._prev_action[:] = self._action
        self._action[:] = action.to(self.device)

        # split the actions and apply to each tensor
        idx = 0
        for term in self._terms.values():
            term_actions = action[:, idx : idx + term.action_dim]
            term.process_actions(term_actions)
            idx += term.action_dim

    def apply_action(self) -> None:
        """Applies the actions to the environment/simulation.

        Note:
            This should be called at every simulation step.
        """
        for term in self._terms.values():
            term.apply_actions()

    def get_term(self, name: str) -> ActionTerm:
        """Returns the action term with the specified name.

        Args:
            name: The name of the action term.

        Returns:
            The action term with the specified name.
        """
        return self._terms[name]

    """
    Helper functions.
    """

    def _prepare_terms(self):
        # create buffers to parse and store terms
        self._term_names: list[str] = list()
        self._terms: dict[str, ActionTerm] = dict()

        # check if config is dict already
        if isinstance(self.cfg, dict):
            cfg_items = self.cfg.items()
        else:
            cfg_items = self.cfg.__dict__.items()
        # parse action terms from the config
        for term_name, term_cfg in cfg_items:
            # check if term config is None
            if term_cfg is None:
                continue
            # check valid type
            if not isinstance(term_cfg, ActionTermCfg):
                raise TypeError(
                    f"Configuration for the term '{term_name}' is not of type ActionTermCfg."
                    f" Received: '{type(term_cfg)}'."
                )
            # create the action term
            term = term_cfg.class_type(term_cfg, self._env)
            # sanity check if term is valid type
            if not isinstance(term, ActionTerm):
                raise TypeError(f"Returned object for the term '{term_name}' is not of type ActionType.")
            # add term name and parameters
            self._term_names.append(term_name)
            self._terms[term_name] = term

import yaml
constants_path = "source/constants.yaml"
with open(constants_path, "r") as file:
    constants = yaml.safe_load(file)
JOINT_UNITREE_TO_ISAAC_LAB_MAPPING = constants["JOINT_UNITREE_TO_ISAAC_LAB_MAPPING"]
JOINT_ISAAC_LAB_TO_UNITREE_MAPPING = constants["JOINT_ISAAC_LAB_TO_UNITREE_MAPPING"]

class LatentActionManager(ActionManager):

    def __init__(self, cfg: object, env: ManagerBasedEnv):
        self.robot_action_dim = 12
        self.latent_action_dim = 1 + 4 * 2 # main freq, 4 legs with freq and amp each
        self.residual_action_weight = 0.0

        super().__init__(cfg, env)

        # action buffers
        self._action = torch.zeros(
            (self.num_envs, self.robot_action_dim), device=self.device
        )
        self._prev_action = torch.zeros_like(self._action)

        self._residual_action = torch.zeros(
            (self.num_envs, self.robot_action_dim), device=self.device
        )
        self._prev_residual_action = torch.zeros_like(
            self._residual_action
        )
        self._latent_action = torch.zeros(
            (self.num_envs, self.latent_action_dim), device=self.device
        )
        self._prev_latent_action = torch.zeros_like(
            self._latent_action
        )

        self.projector = torch.jit.load("expert_projectors/temporal_spatial_prior.pt").to(self.device)

        self.mean_freq = None
        if self.projector.get_frequency() is not None: # temporal prior
            self.mean_freq = self.projector.get_frequency()
            self.mean_amp = 1.0

            self.range_leg_freq = 0.0 # 0.3
            self.range_main_freq = 0.0 # 1.0
            self.range_amp = 0.0 # 0.3

            self.phases = torch.zeros((self.num_envs, 4), device=self.device) # phase for each leg

            self.one_hot_vector = torch.tensor(torch.eye(4)).repeat(self.num_envs, 1).to(self.device)


            torch.pi = torch.acos(torch.zeros(1)).item() * 2
        else:
            raise NotImplementedError("Only temporal prior is supported")

    @property
    def action_term_dim(self) -> list[int]:
        """Shape of each action term."""
        return [self.robot_action_dim + self.latent_action_dim] # This is queried by the policy to get output dimension. Its seems save to use this variable

    def process_action(self, residual_and_latent_action: torch.Tensor):
        """Processes the actions sent to the environment.

        Note:
            This function should be called once per environment step.

        Args:
            action: The actions to process.
        """
        assert residual_and_latent_action.shape[1] == self.total_action_dim

        # TODO possibly clip actions [-1,1]
        residual_and_latent_action = torch.clamp(residual_and_latent_action, -1.0, 1.0)

        # get residual actions (should be same for all projectors)
        self._residual_action[:] = residual_and_latent_action[
            :, : self.robot_action_dim
        ].to(self.device)
        self._prev_residual_action[:] = self.residual_action

        self._latent_action[:] = residual_and_latent_action[
            :, self.robot_action_dim :
        ].to(self.device)
        self._prev_latent_action[:] = self.latent_action

        # get projected actions
        if self.mean_freq is not None:  # temporal prior
            main_freq = (
                self.latent_action[:, 0] * self.range_main_freq
                + self.mean_freq
            )
            # for each leg we have two parameters: freq and amp
            n = (self.latent_action.shape[1] - 1) // 2
            assert n == 4 # number of legs
            _params = self.latent_action[:, 1 :].reshape(-1, 2, n)
            amps = _params[:, 0] * self.range_amp + self.mean_amp
            leg_freqs = _params[:, 1] * self.range_leg_freq

            self.phases = self.phases + self._env.step_dt * 2 * torch.pi * (main_freq.unsqueeze(1) + leg_freqs)

            # reset phases for envs where episode length is 0
            self.phases[torch.where(self._env.episode_length_buf == 0, True, False)] = (
                0.0
            )

            sin_phase = amps * torch.sin(self.phases)
            cos_phase = amps * torch.cos(self.phases)
            sin_cos_phase = torch.cat(
                (sin_phase.flatten().unsqueeze(1), cos_phase.flatten().unsqueeze(1)),
                dim=1,
            )
            inputs = torch.column_stack((sin_cos_phase, self.one_hot_vector))
            with torch.no_grad():
                projected_action = self.projector(inputs).reshape(self.num_envs, -1)
                assert projected_action.shape[1] == self.robot_action_dim
        else:
            raise NotImplementedError("Only temporal prior is supported")

        action = (
            1 - self.residual_action_weight
        ) * projected_action + self.residual_action_weight * self.residual_action

        assert action.shape[1] == self.robot_action_dim

        # re-order joints
        action = action[:, JOINT_UNITREE_TO_ISAAC_LAB_MAPPING]

        self._action[:] = action.to(self.device)
        self._prev_action[:] = self.action

        # split the actions and apply to each tensor
        idx = 0
        for term in self._terms.values():
            term_actions = action[:, idx : idx + term.action_dim]
            term.process_actions(term_actions) # does rescaling, offset of joint targets
            idx += term.action_dim

    def apply_action(self) -> None:
        """Applies the actions to the environment/simulation.

        Note:
            This should be called at every simulation step.
        """
        for term in self._terms.values():
            term.apply_actions() # sets joint pos targets

    @property
    def latent_action(self) -> torch.Tensor:
        """The actions sent to the environment. Shape is (num_envs, total_action_dim)."""
        return self._latent_action

    @property
    def prev_latent_action(self) -> torch.Tensor:
        """The previous actions sent to the environment. Shape is (num_envs, total_action_dim)."""
        return self._prev_latent_action

    @property
    def residual_action(self) -> torch.Tensor:
        """The actions sent to the environment. Shape is (num_envs, total_action_dim)."""
        return self._residual_action

    @property
    def prev_residual_action(self) -> torch.Tensor:
        """The previous actions sent to the environment. Shape is (num_envs, total_action_dim)."""
        return self._prev_residual_action
