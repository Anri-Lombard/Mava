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
"""Warehouse scenarios configurations."""

rware_scenario_configs = {
    "tiny-4ag": {
        "column_height": 8,
        "shelf_rows": 1,
        "shelf_columns": 3,
        "num_agents": 4,
        "sensor_range": 1,
        "request_queue_size": 4,
    },
    "tiny-4ag-easy": {
        "column_height": 8,
        "shelf_rows": 1,
        "shelf_columns": 3,
        "num_agents": 4,
        "sensor_range": 1,
        "request_queue_size": 8,
    },
    "tiny-2ag": {
        "column_height": 8,
        "shelf_rows": 1,
        "shelf_columns": 3,
        "num_agents": 2,
        "sensor_range": 1,
        "request_queue_size": 2,
    },
    "small-4ag": {
        "column_height": 8,
        "shelf_rows": 2,
        "shelf_columns": 3,
        "num_agents": 4,
        "sensor_range": 1,
        "request_queue_size": 4,
    },
}
