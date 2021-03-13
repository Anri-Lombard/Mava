# python3
# Copyright 2021 InstaDeep Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Core Mava interfaces.
This file specifies and documents the notions of `Executor` and `Trainer` similar to the `Actor` and `Learner` in Acme.
"""

import abc
import itertools
from typing import Generic, List, Optional, Sequence, TypeVar, Dict

import acme
from acme import types

# Internal imports.
from acme.utils import metrics
import dm_env

T = TypeVar("T")


@metrics.record_class_usage
class Executor(acme.Actor):
    """Interface for a system that can execute agent policies.
    This interface defines an API for a System to interact with an EnvironmentLoop
    (see mava.environment_loop), e.g. a simple RL loop where each step is of the
    form:
      # Make the first observation.
      timestep = env.reset()
      system.observe_first(timestep.observation)
      # Take a step and observe.
      action = system.select_actions(timestep.observation)
      next_timestep = env.step(action)
      actor.observe(action, next_timestep)
      # Update the actor policy/parameters.
      system.update()
    """

    @abc.abstractmethod
    def select_action(
        self, agent: str, observation: types.NestedArray
    ) -> types.NestedArray:
        """Samples from the policy and returns an action."""

    @abc.abstractmethod
    def select_actions(
        self, observations: Dict[str, types.NestedArray]
    ) -> Dict[str, types.NestedArray]:
        """Samples from the policy and returns an action for each agent."""

    @abc.abstractmethod
    def observe(
        self,
        actions: Dict[str, types.NestedArray],
        next_timestep: dm_env.TimeStep,
    ):
        """Make an observation of timestep data from the environment.
        Args:
        action: action taken in the environment.
        next_timestep: timestep produced by the environment given the action.
        """


# Internal class.


class Trainer(acme.Learner):
    """Abstract learner object.
    This corresponds to an object which implements a learning loop. A single step
    of learning should be implemented via the `step` method and this step
    is generally interacted with via the `run` method which runs update
    continuously.
    All objects implementing this interface should also be able to take in an
    external dataset (see acme.datasets) and run updates using data from this
    dataset. This can be accomplished by explicitly running `learner.step()`
    inside a for/while loop or by using the `learner.run()` convenience function.
    Data will be read from this dataset asynchronously and this is primarily
    useful when the dataset is filled by an external process.
    """

    @abc.abstractmethod
    def get_variables(
        self, names: Dict[str, Sequence[str]]
    ) -> Dict[str, List[types.NestedArray]]:
        """Return the named variables as a collection of (nested) numpy arrays.
        Args:
          names: args where each name is a string identifying a predefined subset of
            the variables.
        Returns:
          A list of (nested) numpy arrays `variables` such that `variables[i]`
          corresponds to the collection named by `names[i]`.
        """