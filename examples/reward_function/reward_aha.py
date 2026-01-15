# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import json
import re
from typing import Any

# from mathruler.grader import extract_boxed_content, grade_answer


# Metadata
REWARD_NAME = "math"
REWARD_TYPE = "batch"


def format_reward(response: str) -> float:
    try:
        # Extract JSON content: find the substring between the first '{' and the last '}'
        start_idx = response.find("{")
        end_idx = response.rfind("}")

        if start_idx == -1 or end_idx == -1 or start_idx >= end_idx:
            return 0.0

        json_str = response[start_idx : end_idx + 1]
        data = json.loads(json_str)

        # Check if the keys match the expected structure
        required_keys = {"scene_description", "object_state", "reasoning", "task_completion"}
        # Check if all required keys are present in the parsed JSON
        if required_keys.issubset(data.keys()):
            return 1.0
        return 0.0
    except (json.JSONDecodeError, AttributeError, ValueError):
        return 0.0


def accuracy_reward(response: str, ground_truth: str) -> float:
    try:
        # Extract JSON from response
        start_idx = response.find("{")
        end_idx = response.rfind("}")
        if start_idx == -1 or end_idx == -1 or start_idx >= end_idx:
            return 0.0

        response_data = json.loads(response[start_idx : end_idx + 1])

        # Parse ground_truth (expected to be a JSON string)
        gt_data = json.loads(ground_truth)

        # Compare task_completion
        if response_data.get("task_completion") == gt_data.get("task_completion"):
            return 1.0
        return 0.0
    except (json.JSONDecodeError, AttributeError, ValueError, TypeError):
        return 0.0


def compute_score(reward_inputs: list[dict[str, Any]], format_weight: float = 0.1) -> list[dict[str, float]]:
    scores = []
    for reward_input in reward_inputs:
        # response = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])  # handle qwen2.5vl-32b format
        format_score = format_reward(reward_input["response"])
        accuracy_score = accuracy_reward(reward_input["response"], reward_input["ground_truth"])
        scores.append(
            {
                "overall": (1 - format_weight) * accuracy_score + format_weight * format_score,
                "format": format_score,
                "accuracy": accuracy_score,
            }
        )

    return scores
