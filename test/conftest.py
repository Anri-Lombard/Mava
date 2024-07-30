# Copyright 2022 InstaDeep Ltd. All rights reserved.
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

from typing import Dict

import pytest


@pytest.fixture
def fast_config() -> Dict[str, Dict[str, bool | int | float]]:
    return {
        "system": {
            # common
            "num_updates": 2,
            "rollout_length": 1,
            "num_minibatches": 1,
            "update_batch_size": 1,
            # ppo:
            "ppo_epochs": 1,
            # sac:
            "explore_steps": 1,
            "epochs": 1,  # also for iql
            "policy_update_delay": 1,
            "buffer_size": 8,  # also for iql
            "batch_size": 1,
            # iql
            "min_buffer_size": 4,
            "sample_batch_size": 1,
            "sample_sequence_length": 1,
        },
        "arch": {
            "num_envs": 1,
            "num_eval_episodes": 1,
            "num_evaluation": 1,
            "absolute_metric": False,
        },
    }