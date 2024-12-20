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


torch.pi = torch.acos(torch.zeros(1)).item() * 2
import yaml
import matplotlib.pyplot as plt
constants_path = "source/constants.yaml"
with open(constants_path, "r") as file:
    constants = yaml.safe_load(file)
JOINT_UNITREE_TO_ISAAC_LAB_MAPPING = constants["JOINT_UNITREE_TO_ISAAC_LAB_MAPPING"]
JOINT_ISAAC_LAB_TO_UNITREE_MAPPING = constants["JOINT_ISAAC_LAB_TO_UNITREE_MAPPING"]
DEFAULT_JOINT_POS_ISAAC_LAB = constants["DEFAULT_JOINT_POS_ISAAC_LAB"]


class LegwiseLatentActionManager(ActionManager):

    def __init__(self, cfg: object, env: ManagerBasedEnv):
        self.robot_action_dim = 12
        self.latent_action_dim = 1 + 4 * 2 # main freq, 4 legs with freq and amp each
        self.residual_action_weight = 0.05

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

        self.init_joint_pos_gait = torch.tensor([
            0.0735669806599617,
            0.9497295618057251,
            -1.594949722290039,
            -0.10308756679296494,
            0.7876842617988586,
            -1.908045768737793,
            0.008162420243024826,
            0.7421717643737793,
            -1.8416037559509277,
            -0.0335349403321743,
            0.897495448589325,
            -1.5823912620544434,
        ]) - torch.tensor(DEFAULT_JOINT_POS_ISAAC_LAB)[JOINT_ISAAC_LAB_TO_UNITREE_MAPPING]

        self.init_joint_pos_gait = torch.zeros_like(self.init_joint_pos_gait)

        self.init_joint_pos_gait = self.init_joint_pos_gait.repeat(self.num_envs, 1).to(self.device)


        self.live_print = False
        self.live_plot = False
        if self.live_plot:
            plt.ion()
            self._values_to_plot = {}
            for i in range(4):
                    self._values_to_plot.setdefault(f"leg_{i}_freq", [])
                    self._values_to_plot.setdefault(f"leg_{i}_amp", [])
                    self._values_to_plot.setdefault(f"leg_{i}_phase", [])
                    self._values_to_plot.setdefault(f"main_freq", [])
            for i in range(12):
                self._values_to_plot.setdefault(f"joint_{i}_pos", [])
            num_plots = len(self._values_to_plot)
            num_rows = (num_plots + 1) // 2
            self.figure, self.axs = plt.subplots(num_rows, 2)
            self.figure.set_size_inches(10, 5 * num_rows)
            self.axs = self.axs.flatten()
            


        self.has_freq = False
        if self.projector.get_frequency() is not None: # temporal prior

            self.has_freq = True

            self._demo_freq = self.projector.get_frequency()            
            self.mean_main_freq = 0.0 # self._demo_freq

            self.mean_amp = 1.0

            self.range_main_freq = 2 * self._demo_freq # * 1.2  # 0.3 # 1.0
            self.range_leg_freq = 0.0 # 0.3 # 0.3
            self.range_amp = 0.1 # 0.2 # 0.3

            self.phases = torch.zeros((self.num_envs, 4), device=self.device) # phase for each leg
            # sin cos phase for observations
            self.sin_cos_phases = torch.cat(
                (
                    self.mean_amp * torch.sin(self.phases),
                    self.mean_amp * torch.cos(self.phases),
                ),
                dim=1,
            ) # sin cos phase for leg; required for observations
            
            self.one_hot_vector = torch.eye(4).repeat(self.num_envs, 1).to(self.device)

            # buffers
            self._freqs = torch.zeros((self.num_envs, 4), device=self.device) # freq for each leg
            self._prev_freqs = torch.zeros_like(self._freqs)
            self._amps = torch.zeros((self.num_envs, 4), device=self.device) # amp for each leg
            self._prev_amps = torch.zeros_like(self._amps)

            self.i = 0

        else:
            raise NotImplementedError("Only temporal prior is supported")

    @property
    def action_term_dim(self) -> list[int]:
        """Shape of each action term."""
        return [self.robot_action_dim + self.latent_action_dim] # This is queried by the policy to get output dimension. Its seems save to modify this variable.

    def process_action(self, residual_and_latent_action: torch.Tensor):
        """Processes the actions sent to the environment.

        Note:
            This function should be called once per environment step.

        Args:
            action: The actions to process.
        """
        self.i += 1

        self._prev_action[:] = self.action
        self._prev_freqs = self.freqs
        self._prev_amps = self.amps

        assert residual_and_latent_action.shape[1] == self.total_action_dim

        # clip actions [-1,1]
        residual_and_latent_action = torch.clamp(residual_and_latent_action, -1.0, 1.0)

        # get residual actions (should be same for all projectors); these are the first 12 actions
        self._residual_action[:] = residual_and_latent_action[
            :, : self.robot_action_dim
        ].to(self.device)
        self._prev_residual_action[:] = self.residual_action

        # latent actions are remaining actions
        self._latent_action[:] = residual_and_latent_action[
            :, self.robot_action_dim :
        ].to(self.device)
        self._prev_latent_action[:] = self.latent_action

        # get projected actions
        if self.has_freq:  # temporal prior

            main_freq = (
                self.latent_action[:, 0] * self.range_main_freq + self.mean_main_freq
                # torch.where(
                #     self._env.unwrapped.command_manager.get_command('base_velocity')[:, 0] < 0,
                #     -self.mean_main_freq,
                #     self.mean_main_freq
                # )
            )
            # for each leg we have two parameters: freq and amp
            n = (self.latent_action.shape[1] - 1) // 2
            assert n == 4  # number of legs
            _params = self.latent_action[:, 1:].reshape(-1, 2, n)
            amps = _params[:, 0] * self.range_amp + self.mean_amp
            leg_freqs = _params[:, 1] * self.range_leg_freq

            self._freqs = main_freq.unsqueeze(1) + leg_freqs
            self._amps = amps

            self.phases = self.phases + self._env.step_dt * 2 * torch.pi * self.freqs

            if self.live_print:
                for i in range(10):
                    leg_freqs_str = ", ".join(
                        [
                            f"Leg {j+1} freq: {self.freqs[i][j].cpu().numpy().tolist():.5f}"
                            for j in range(4)
                        ]
                    )
                    print(
                        f"Target velocity: {self._env.unwrapped.command_manager.get_command('base_velocity')[i].tolist()}, Main freq: {main_freq[i].cpu().numpy().tolist():.5f}, {leg_freqs_str}"
                    )
                print("###############")

            if self.live_plot:

                self._values_to_plot.setdefault(f"main_freq", []).append(main_freq[0].cpu().numpy())
                for i in range(self._freqs.shape[1]): # for each leg
                    self._values_to_plot.setdefault(f"leg_{i}_freq", []).append(self.freqs[0][i].cpu().numpy())
                    self._values_to_plot.setdefault(f"leg_{i}_amp", []).append(amps[0][i].cpu().numpy())
                    self._values_to_plot.setdefault(f"leg_{i}_phase", []).append(self.phases[0][i].cpu().numpy())

                for i in range(12):
                    self._values_to_plot.setdefault(f"joint_{i}_pos", []).append(self._env.unwrapped.scene["robot"].data.joint_pos.cpu().numpy()[0][i])

                for i, (key, value) in enumerate(self._values_to_plot.items()):
                    self.axs[i].clear()
                    self.axs[i].plot(
                        [x * self._env.step_dt for x in range(len(value))], value
                    )
                    self.axs[i].set_title(key)

                
                
                plt.pause(0.001)
                plt.show()



            # reset phases for envs where episode length is 0; possible not required, as its handled by reset method of this class
            self.phases[torch.where(self._env.episode_length_buf == 0, True, False)] = (
                0.0
            )

            sin_phase = self.amps * torch.sin(self.phases)
            cos_phase = self.amps * torch.cos(self.phases)

            # sin cos phase for observations
            self.sin_cos_phases = torch.cat(
                (sin_phase, cos_phase),
                dim=1,
            )

            # sin cos phase for projector
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

        # if self.i < 150:
        #     action = self.init_joint_pos_gait
        # else:
        #     self.mean_main_freq = self._demo_freq

        assert action.shape[1] == self.robot_action_dim

        # re-order joints
        action = action[:, JOINT_UNITREE_TO_ISAAC_LAB_MAPPING]


        self._action[:] = action.to(self.device)

        # split the actions and apply to each tensor
        idx = 0
        for term in self._terms.values():
            term_actions = action[:, idx : idx + term.action_dim]
            term.process_actions(term_actions) # rescaling, offset
            idx += term.action_dim


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
        self._prev_residual_action[env_ids] = 0.0
        self._residual_action[env_ids] = 0.0
        self._prev_latent_action[env_ids] = 0.0
        self._latent_action[env_ids] = 0.0
        self._prev_freqs[env_ids] = 0.0
        self._freqs[env_ids] = 0.0
        self._prev_amps[env_ids] = 0.0
        self._amps[env_ids] = 0.0
        if self.has_freq:
            self.phases[env_ids] = 0.0
            self.sin_cos_phases = torch.cat(
                (
                    self.mean_amp * torch.sin(self.phases),
                    self.mean_amp * torch.cos(self.phases),
                ),
                dim=1,
            )

        # reset all action terms
        for term in self._terms.values():
            term.reset(env_ids=env_ids)
        # nothing to log here
        return {}

    @property
    def latent_action(self) -> torch.Tensor:
        return self._latent_action

    @property
    def prev_latent_action(self) -> torch.Tensor:
        return self._prev_latent_action

    @property
    def residual_action(self) -> torch.Tensor:
        return self._residual_action

    @property
    def prev_residual_action(self) -> torch.Tensor:
        return self._prev_residual_action
    
    @property
    def freqs(self) -> torch.Tensor:
        return self._freqs
    
    @property
    def prev_freqs(self) -> torch.Tensor:
        return self._prev_freqs
    
    @property
    def amps(self) -> torch.Tensor:
        return self._amps
    
    @property
    def prev_amps(self) -> torch.Tensor:
        return self._prev_amps
