"""
计算 Oracle (专家策略) 在测试集上的理论收益上限。
逻辑与 EasyR1/tests/test_guba.py 保持一致，但使用基于未来价格的专家策略代替模型输出。
"""

import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import groupby
from tqdm import tqdm

# Add project root to sys.path to allow importing from guba
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from guba import gen_sft_data

def load_test_data(test_file: str):
    """加载测试数据集"""
    samples = []
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples

def calculate_oracle_actions(prices):
    """
    根据未来价格计算专家动作
    prices: 列表，长度为 trajectory_length + 1
    """
    if not prices:
        return []
        
    # 构造 DataFrame 以复用 guba.gen_sft_data 中的逻辑
    df = pd.DataFrame({'收盘价(元)': prices})
    
    # 复用预处理逻辑 (计算 Clean_Ret)
    # 注意：这里使用与 gen_sft_data.py 一致的 cost_rate (0.002)
    df = gen_sft_data.preprocess_cost_only(df, price_col='收盘价(元)', cost_rate=0.002)
    
    # 复用计算专家仓位逻辑
    # trajectory_length 为 prices 长度减 1 (因为需要 next_price)
    # start_idx 为 0
    actions = gen_sft_data.calculate_expert_positions(
        df, 
        start_idx=0, 
        trajectory_length=len(prices)-1,
        scale_factor=80.0
    )
    
    return actions

def run_oracle_backtest(test_file: str):
    print(f"加载测试数据: {test_file}")
    raw_samples = load_test_data(test_file)
    print(f"共加载 {len(raw_samples)} 个原始样本")

    # 解析 GT 并排序
    parsed_samples = []
    for s in raw_samples:
        gt = s["answer"]
        if isinstance(gt, str):
            gt = json.loads(gt)
        parsed_samples.append({
            "gt": gt,
            "sequence_id": gt.get("sequence_id", "default"),
            "start_idx": gt.get("trajectory_start_idx", 0)
        })
    
    # 按序列和时间排序
    parsed_samples.sort(key=lambda x: (x["sequence_id"], x["start_idx"]))
    
    # 回测参数
    initial_wealth = 100000.0
    volatility_target = 0.15
    transaction_cost_bp = 0.002
    mu = 1.0
    
    cumulative_wealth = initial_wealth
    total_reward = 0.0
    
    rewards = []
    positions = []
    
    print("\n开始 Oracle 回测...")
    
    pbar = tqdm(total=len(parsed_samples), desc="Processing")
    processed_count = 0
    
    for seq_id, group in groupby(parsed_samples, key=lambda x: x["sequence_id"]):
        items = list(group)
        
        # 初始化序列状态
        first_gt = items[0]["gt"]
        prev_action = first_gt.get("initial_action", 0.0)
        prev_vol = first_gt.get("initial_volatility", 0.15)
        prev_scale = volatility_target / (prev_vol + 1e-8)
        
        last_idx = items[0]["start_idx"] - 1
        
        item_idx = 0
        while item_idx < len(items):
            item = items[item_idx]
            current_idx = item["start_idx"]
            
            # 连续性检查
            if current_idx != last_idx + 1:
                prev_action = item["gt"].get("initial_action", 0.0)
                prev_vol = item["gt"].get("initial_volatility", item["gt"].get("volatilities", [0.15])[0])
                prev_scale = volatility_target / (prev_vol + 1e-8)
            
            last_idx = current_idx
            
            # 1. 获取 Oracle 动作
            gt_prices = item["gt"].get("prices", [])
            actions = calculate_oracle_actions(gt_prices)
            
            # 2. 回测逻辑
            num_actions = len(actions)
            max_steps = min(num_actions, len(items) - item_idx)
            
            if max_steps == 0:
                max_steps = 1
                
            for step_idx in range(max_steps):
                if item_idx + step_idx >= len(items):
                    break
                
                # 获取当前步骤的动作（Oracle 动作）
                action_t = actions[step_idx]
                positions.append(action_t)
                
                # 获取环境数据
                current_item = items[item_idx + step_idx]
                gt_prices_step = current_item["gt"].get("prices", [])
                gt_vols_step = current_item["gt"].get("volatilities", [])
                
                if len(gt_prices_step) > 0:
                    price_t = gt_prices_step[0]
                else:
                    price_t = 0.0
                    
                if len(gt_prices_step) > 1:
                    price_next = gt_prices_step[1]
                elif item_idx + step_idx + 1 < len(items):
                    # 尝试从下一个样本获取
                    next_item = items[item_idx + step_idx + 1]
                    next_prices = next_item["gt"].get("prices", [])
                    price_next = next_prices[0] if len(next_prices) > 0 else price_t
                else:
                    price_next = price_t
                
                vol_t = gt_vols_step[0] if len(gt_vols_step) > 0 else 0.15
                scale_t = volatility_target / (vol_t + 1e-8)
                
                # 计算收益
                r_t_next = price_next - price_t
                profit = action_t * scale_t * r_t_next
                
                # 计算成本
                position_change = abs(scale_t * action_t - prev_scale * prev_action)
                cost = transaction_cost_bp * price_t * position_change
                
                step_reward = mu * (profit - cost)
                
                rewards.append(step_reward)
                total_reward += step_reward
                cumulative_wealth += step_reward
                
                # 更新状态
                prev_action = action_t
                prev_scale = scale_t
                processed_count += 1
                
                # 更新 last_idx 以匹配当前处理到的位置
                last_idx = current_item["start_idx"]
            
            # 跳过已处理的样本
            item_idx += max_steps
            pbar.update(max_steps)
            
    pbar.close()
    
    # 统计结果
    rewards_np = np.array(rewards)
    positions_np = np.array(positions)
    
    stats = {
        "initial_wealth": initial_wealth,
        "final_wealth": cumulative_wealth,
        "total_reward": total_reward,
        "return_rate": (total_reward / initial_wealth) * 100,
        "mean_reward": np.mean(rewards_np) if len(rewards_np) > 0 else 0.0,
        "sharpe_ratio": 0.0,
        "max_drawdown": 0.0,
        "avg_position": np.mean(np.abs(positions_np)) if len(positions_np) > 0 else 0.0,
        "long_count": int(np.sum(positions_np > 0.01)),
        "short_count": int(np.sum(positions_np < -0.01)),
        "neutral_count": int(np.sum(np.abs(positions_np) <= 0.01))
    }
    
    if len(rewards) > 1 and np.std(rewards) > 1e-6:
        stats["sharpe_ratio"] = (np.mean(rewards) / np.std(rewards)) * np.sqrt(252)
        
    if len(rewards) > 0:
        cum_returns = np.cumsum(rewards)
        running_max = np.maximum.accumulate(cum_returns)
        drawdown = running_max - cum_returns
        stats["max_drawdown"] = np.max(drawdown)

    print("\n" + "="*60)
    print("Oracle (Expert) Strategy Evaluation Results")
    print("="*60)
    print(f"初始财富: {stats['initial_wealth']:.2f}")
    print(f"最终财富: {stats['final_wealth']:.2f}")
    print(f"累计收益: {stats['total_reward']:.2f}")
    print(f"收益率: {stats['return_rate']:.2f}%")
    print(f"夏普比率: {stats['sharpe_ratio']:.4f}")
    print(f"最大回撤: {stats['max_drawdown']:.2f}")
    print(f"平均每步收益: {stats['mean_reward']:.2f}")
    print("-" * 30)
    print(f"持仓分布: 多 {stats['long_count']} | 空 {stats['short_count']} | 平 {stats['neutral_count']}")
    print(f"平均绝对持仓: {stats['avg_position']:.4f}")
    print("="*60)

if __name__ == "__main__":
    TEST_FILE = "/home/tione/notebook/workspace/xiaoyangchen/work/data/guba/guba_eval_verl.jsonl"
    if not Path(TEST_FILE).exists():
        print(f"File not found: {TEST_FILE}")
    else:
        run_oracle_backtest(TEST_FILE)

