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
import numpy as np
from typing import Any

# from mathruler.grader import extract_boxed_content, grade_answer


# Metadata
REWARD_NAME = "guba"
REWARD_TYPE = "batch"


def format_reward(response: str) -> float:
    """
    检查响应是否符合要求的 JSON 格式，包含交易决策。
    期望格式：{"position": [-1, 1]的连续值, ...}
    """
    try:
        # 尝试提取 JSON 内容
        start_idx = response.find("{")
        end_idx = response.rfind("}")

        if start_idx == -1 or end_idx == -1 or start_idx >= end_idx:
            return 0.0

        json_str = response[start_idx : end_idx + 1]
        data = json.loads(json_str)

        # 检查必需字段：position（交易决策）
        if "position" not in data:
            return 0.0

        position = data["position"]
        # 检查 position 是否为有效的连续交易决策值：[-1, 1] 范围内的浮点数
        # -1 表示完全做空，0 表示平仓，1 表示完全做多，中间值表示部分持仓
        if not isinstance(position, (int, float)) or not (-1.0 <= float(position) <= 1.0):
            return 0.0

        return 1.0
    except (json.JSONDecodeError, AttributeError, ValueError, TypeError):
        return 0.0


# 全局动作历史缓冲区（用于维护跨样本的状态）
# 注意：这个实现假设样本按照时间顺序处理，并且每个序列有唯一的标识
_action_history_buffer = {}

def accuracy_reward(response: str, ground_truth: str) -> float:
    """
    基于实际交易收益计算奖励。
    
    注意：在强化学习中，action_t_prev 和 action_t_prev_prev 应该是之前时间步的实际模型输出，
    而不是预先固定的值。当前实现会尝试从历史缓冲区获取实际的动作历史（如果 ground_truth 
    中包含 time_index 和 sequence_id），否则使用 ground_truth 中预设的历史动作（参考策略生成）。
    
    Args:
        response: 模型生成的响应，包含当前的交易决策 position
        ground_truth: JSON 字符串，包含：
            - time_index (可选): 时间索引，用于标识样本在序列中的位置
            - sequence_id (可选): 序列标识符，用于区分不同的交易序列
            - action_t_prev (上一个动作/持仓，如果无法从历史获取则使用此值)
            - action_t_prev_prev (上上个动作/持仓，如果无法从历史获取则使用此值)
            - price_t (当前价格)
            - price_t_prev (上一个价格)
            - volatility_t_prev (t-1 时刻的波动率)
            - volatility_t_prev_prev (t-2 时刻的波动率)
            - volatility_target (目标波动率)
            - transaction_cost_bp (交易成本，basis points)
    
    根据论文公式计算奖励：
    R_t = μ [ A_{t-1} * (σ_{tgt}/σ_{t-1}) * r_t - bp * p_{t-1} * |(σ_{tgt}/σ_{t-1}) * A_{t-1} - (σ_{tgt}/σ_{t-2}) * A_{t-2}| ]
    
    其中：
    - μ = 1 (固定)
    - A_{t-1} 是 t-1 时刻的动作（持仓）
    - r_t = p_t - p_{t-1} (收益率)
    - σ_{tgt} 是目标波动率
    - σ_{t-1} 是 t-1 时刻的波动率估计
    - bp 是交易成本率
    """
    try:
        # 解析响应
        start_idx = response.find("{")
        end_idx = response.rfind("}")
        if start_idx == -1 or end_idx == -1 or start_idx >= end_idx:
            return 0.0

        response_data = json.loads(response[start_idx : end_idx + 1])
        position = response_data.get("position")
        
        # 确保 position 在有效范围内
        if position is None:
            return 0.0
        # 将 position 规范化为 [-1, 1] 范围
        if isinstance(position, (int, float)):
            position = max(-1.0, min(1.0, float(position)))
        else:
            return 0.0

        # 解析 ground_truth
        gt_data = json.loads(ground_truth)
        
        # 尝试从历史缓冲区获取实际的历史动作
        # 注意：这要求样本按时间顺序处理，并且有正确的 time_index 和 sequence_id
        sequence_id = gt_data.get("sequence_id", "default")
        time_index = gt_data.get("time_index", None)
        
        # 如果提供了时间索引，尝试使用动态历史动作
        if time_index is not None:
            # 使用动态历史动作
            if sequence_id not in _action_history_buffer:
                _action_history_buffer[sequence_id] = {}
            
            history = _action_history_buffer[sequence_id]
            
            # 从历史中获取之前的动作（如果存在），否则使用 ground_truth 中的参考值
            action_t_prev = history.get(time_index - 1, gt_data.get("action_t_prev", 0.0))
            action_t_prev_prev = history.get(time_index - 2, gt_data.get("action_t_prev_prev", 0.0))
            
            # 更新历史缓冲区（将当前动作保存，供下一个时间步使用）
            history[time_index] = position
        else:
            print("没有时间索引")
            # 如果没有时间索引，使用 ground_truth 中预设的历史动作（参考策略生成）
            action_t_prev = gt_data.get("action_t_prev", 0.0)
            action_t_prev_prev = gt_data.get("action_t_prev_prev", 0.0)
        price_t = gt_data.get("price_t", 0.0)
        price_t_prev = gt_data.get("price_t_prev", 0.0)
        volatility_t_prev = gt_data.get("volatility_t_prev", 1.0)  # t-1 时刻的波动率
        volatility_t_prev_prev = gt_data.get("volatility_t_prev_prev", 1.0)  # t-2 时刻的波动率
        volatility_target = gt_data.get("volatility_target", 0.15)  # 目标波动率，默认15%
        transaction_cost_bp = gt_data.get("transaction_cost_bp", 2.0)  # 交易成本，默认2bp
        mu = gt_data.get("mu", 1.0)  # 固定系数，默认1

        # 验证数据有效性
        if price_t <= 0 or price_t_prev <= 0 or volatility_t_prev <= 0:
            return 0.0

        # 计算收益率 r_t = p_t - p_{t-1}（加法收益，用于固定合约数）
        r_t = price_t - price_t_prev

        # 计算波动率缩放因子
        scale_t_prev = volatility_target / (volatility_t_prev + 1e-8)
        scale_t_prev_prev = volatility_target / (volatility_t_prev_prev + 1e-8) if volatility_t_prev_prev > 0 else scale_t_prev

        # 计算收益部分：A_{t-1} * (σ_{tgt}/σ_{t-1}) * r_t
        profit = action_t_prev * scale_t_prev * r_t

        # 计算交易成本部分：bp * p_{t-1} * |(σ_{tgt}/σ_{t-1}) * A_{t-1} - (σ_{tgt}/σ_{t-2}) * A_{t-2}|
        position_change = abs(scale_t_prev * action_t_prev - scale_t_prev_prev * action_t_prev_prev)
        transaction_cost = transaction_cost_bp * 0.0001 * price_t_prev * position_change

        # 计算奖励
        reward = mu * (profit - transaction_cost)

        # 将奖励归一化到 [0, 1] 范围用于奖励函数
        # 使用 sigmoid 或 tanh 函数将奖励映射到合理范围
        # 这里我们假设奖励范围大致在 [-1, 1] 之间，使用 tanh 映射
        normalized_reward = (np.tanh(reward * 100) + 1) / 2  # 乘以100来放大信号，然后归一化到[0,1]

        return float(normalized_reward)

    except (json.JSONDecodeError, AttributeError, ValueError, TypeError, KeyError) as e:
        return 0.0


def compute_score(reward_inputs: list[dict[str, Any]], format_weight: float = 0.1) -> list[dict[str, float]]:
    """
    计算综合奖励分数。
    """
    scores = []
    for reward_input in reward_inputs:
        format_score = format_reward(reward_input["response"])
        accuracy_score = accuracy_reward(reward_input["response"], reward_input["ground_truth"])
        
        # 综合分数：格式权重 * 格式分数 + (1 - 格式权重) * 准确度分数
        overall_score = format_weight * format_score + (1 - format_weight) * accuracy_score
        
        scores.append(
            {
                "overall": overall_score,
                "format": format_score,
                "accuracy": accuracy_score,
            }
        )

    return scores