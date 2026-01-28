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
import random

# from mathruler.grader import extract_boxed_content, grade_answer


# Metadata
REWARD_NAME = "guba"
REWARD_TYPE = "batch"


def format_reward(response: str) -> float:
    """
    检查响应是否符合要求的 JSON 格式，包含交易决策。
    支持两种格式：
    1. 单步模式：{"position": [-1, 1]的连续值}
    2. 轨迹模式：{"positions": [[-1, 1], ...]}
    """
    try:
        # 尝试提取 JSON 内容
        start_idx = response.find("{")
        end_idx = response.rfind("}")

        if start_idx == -1 or end_idx == -1 or start_idx >= end_idx:
            return 0.0

        json_str = response[start_idx : end_idx + 1]
        data = json.loads(json_str)

        # 检查是否为轨迹模式 (positions 列表)
        if "positions" in data:
            positions = data["positions"]
            if not isinstance(positions, list) or not positions:
                return 0.0
            # 检查列表中每个元素是否有效
            for p in positions:
                if not isinstance(p, (int, float)) or not (-1.0 <= float(p) <= 1.0):
                    return 0.0
            return 1.0

        # 检查是否为单步模式 (position 值)
        if "position" in data:
            position = data["position"]
            if not isinstance(position, (int, float)) or not (-1.0 <= float(position) <= 1.0):
                return 0.0
            return 1.0

        return 0.0
    except (json.JSONDecodeError, AttributeError, ValueError, TypeError):
        return 0.0


# 全局动作历史缓冲区（用于维护跨样本的状态）
# 注意：这个实现假设样本按照时间顺序处理，并且每个序列有唯一的标识
_action_history_buffer = {}

def income_reward(response: str, ground_truth: str) -> float:
    """
    基于实际交易收益计算奖励。支持单步和轨迹两种模式。
    
    轨迹模式 (Trajectory Mode):
        如果 response 包含 "positions" 列表，则认为是对一段时间的动作序列预测。
        此时 ground_truth 必须包含对应的序列数据：
        - "prices": 价格序列 [p_0, p_1, ..., p_T] (长度为 T+1，用于计算 T 天的收益)
        - "volatilities": 波动率序列 [v_0, v_1, ..., v_{T-1}] (长度为 T，每个对应一天的决策)
        - "initial_action": 初始动作 (轨迹开始前的动作，默认为0)
        - "initial_volatility": 初始波动率 (用于计算第一步的调仓成本，默认为第一个波动率)
        
        可选参数（如果未提供则使用默认值）：
        - "volatility_target": 目标波动率 (默认 0.15)
        - "transaction_cost_bp": 交易成本 basis points (默认 2.0)
        - "mu": 奖励缩放因子 (默认 1.0)
    
    单步模式 (Single Step Mode):
        如果 response 包含 "position"，则执行原有的单步计算逻辑。
    """
    try:
        # 解析响应
        start_idx = response.find("{")
        end_idx = response.rfind("}")
        if start_idx == -1 or end_idx == -1 or start_idx >= end_idx:
            return 0.0

        response_data = json.loads(response[start_idx : end_idx + 1])
        gt_data = json.loads(ground_truth)

        # ==========================================
        # 模式 1: 轨迹模式 (Trajectory Mode)
        # ==========================================
        if "positions" in response_data and isinstance(response_data["positions"], list):
            positions = response_data["positions"]
            prices = gt_data.get("prices", [])
            volatilities = gt_data.get("volatilities", [])
            
            # 数据校验
            if not prices or not volatilities:
                # 如果没有序列数据，无法计算轨迹奖励
                return 0.0
                
            volatility_target = gt_data.get("volatility_target", 0.15)
            transaction_cost_bp = gt_data.get("transaction_cost_bp", 0.002)
            mu = gt_data.get("mu", 1.0)
            mse_weight = gt_data.get("mse_weight", 0.5) # MSE Loss 权重
            
            # 专家动作序列 (用于 MSE Loss)
            expert_positions = gt_data.get("expert_positions", [])

            # 初始状态
            # action_t_prev: 上一步的持仓，用于计算第一步的调仓成本
            action_t_prev = gt_data.get("initial_action", random.uniform(-1, 1)) 
            # scale_t_prev_prev: 上一步的波动率缩放因子
            initial_volatility = gt_data.get("initial_volatility", random.uniform(0.01, 0.2))
            scale_t_prev_prev = volatility_target / (initial_volatility + 1e-8)

            total_reward = 0.0
            
            # 确定计算步数
            # 动作序列 positions[i] 对应时间步 i 的决策
            # 收益计算依赖于 prices[i+1] - prices[i]
            # 所以 prices 长度至少需要 len(positions) + 1
            steps = len(positions)
            
            mse_loss = 0.0
            
            for i in range(steps):
                # 获取当前决策并规范化
                raw_pos = positions[i]
                if isinstance(raw_pos, (int, float)):
                    current_action = max(-1.0, min(1.0, float(raw_pos)))
                else:
                    current_action = 0.0
                
                # 计算 MSE Loss (如果提供了专家动作)
                if expert_positions and i < len(expert_positions):
                    expert_action = expert_positions[i]
                    mse_loss += (current_action - expert_action) ** 2

                # 获取环境数据
                price_t = prices[i+1]      # p_{t} 
                price_t_prev = prices[i]   # p_{t-1}
                vol_t_prev = volatilities[i] # sigma_{t-1} (决策时的波动率)
                
                # 计算波动率缩放因子
                scale_t_prev = volatility_target / (vol_t_prev + 1e-8)
                
                # 1. 计算收益 (使用收益率而非绝对价格，实现无量纲化，与 MSE 量级匹配)
                # r_t = (p_t - p_{t-1}) / p_{t-1}
                r_t = (price_t - price_t_prev) / (price_t_prev + 1e-8)
                profit = current_action * scale_t_prev * r_t
                
                # 2. 计算交易成本 (同样标准化为收益率形式)
                # Cost = bp * |change| (不再乘以价格)
                # change = | scale_t * A_t - scale_{t-1} * A_{t-1} |2.5*16
                position_change = abs(scale_t_prev * current_action - scale_t_prev_prev * action_t_prev)
                transaction_cost = transaction_cost_bp * position_change
                
                # 累计单步奖励
                step_reward = mu * (profit - transaction_cost)
                total_reward += step_reward
                
                # 更新状态用于下一步
                action_t_prev = current_action
                scale_t_prev_prev = scale_t_prev

            # 计算平均 MSE
            if steps > 0:
                mse_loss = mse_loss / steps
            
            # 减去 MSE Loss 惩罚
            # total_reward -= mse_weight * mse_loss

            # 轨迹奖励归一化
            # GRPO 后续会进行组内归一化 (r - mean) / std，因此这里不需要做非线性变换 (tanh)
            # 直接返回原始的累计奖励，保持收益的线性差异，以便模型能够区分“好”与“更好”
            return float(total_reward), float(mse_loss)
        else:
            return -10.0, 0.0

       

    except (json.JSONDecodeError, AttributeError, ValueError, TypeError, KeyError) as e:
        return -10, 0.0


def compute_score(reward_inputs: list[dict[str, Any]], format_weight: float = 0.1) -> list[dict[str, float]]:
    """
    计算综合奖励分数。
    """
    # import ipdb; ipdb.set_trace()
    scores = []
    for reward_input in reward_inputs:
        format_score = format_reward(reward_input["response"])
        income_score, mse_loss = income_reward(reward_input["response"], reward_input["ground_truth"])
        
        # 格式错误惩罚：如果格式错误（format_score=0），强制 overall 为一个较大的负值
        # 避免模型因为 income 为负（亏损）而故意输出错误格式来获得 0 分
        if format_score == 0.0:
            overall_score = -10.0  # 或者更小的负值，例如 -10.0，取决于 income 的量级
        else:
            # 格式正确时，综合分数：格式权重 * 格式分数 + (1 - 格式权重) * 准确度分数
            overall_score = format_weight * format_score + (1 - format_weight) * income_score - 0.5 * mse_loss
        
        scores.append(
            {
                "overall": overall_score,
                "format": format_score,
                "income": income_score,
                "mse_loss": mse_loss,
            }
        )

    return scores
