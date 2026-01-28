"""
测试训练好的期货交易模型在测试集上的表现，计算实际收益
根据 EasyR1/examples/reward_function/reward_guba.py 和 guba/gen_data.py 的逻辑进行重写。
使用滑动窗口为1的测试数据进行回测，使用模型预测轨迹的所有动作（positions列表的所有值）。
改为串行推理，以确保正确的状态依赖（交易成本计算依赖于上一步的实际模型输出）。
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from itertools import groupby

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoProcessor, AutoModelForImageTextToText
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_test_data(test_file: str) -> List[Dict]:
    """加载测试数据集"""
    samples = []
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def parse_model_response(response: str) -> List[float]:
    """
    解析模型输出，提取 positions 列表的所有值作为动作序列。
    期望格式: {"positions": [0.5, 0.2, ...], "reasoning": "..."}
    返回: positions 列表，每个值都被限制在 [-1.0, 1.0] 范围内
    """
    try:
        # 尝试提取 JSON 内容
        start_idx = response.find("{")
        end_idx = response.rfind("}")
        
        if start_idx == -1 or end_idx == -1 or start_idx >= end_idx:
            return [0.0]
        
        json_str = response[start_idx : end_idx + 1]
        data = json.loads(json_str)
        
        # 提取 positions（所有值）
        if "positions" in data and isinstance(data["positions"], list) and len(data["positions"]) > 0:
            print(data["positions"])
            actions = []
            for action in data["positions"]:
                if isinstance(action, (int, float)):
                    actions.append(max(-1.0, min(1.0, float(action))))
                else:
                    actions.append(0.0)
            return actions if actions else [0.0]
        
        # 兼容旧格式或单步格式
        if "position" in data:
             pos = data["position"]
             if isinstance(pos, (int, float)):
                 return [max(-1.0, min(1.0, float(pos)))]

        return [0.0]
    except (json.JSONDecodeError, AttributeError, ValueError, TypeError, KeyError):
        return [0.0]


def evaluate_model(
    model_path: str,
    test_file: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    use_fast_generation: bool = True,
):
    """
    评估模型：串行推理并回测
    """
    
    
    
    processor = AutoProcessor.from_pretrained(model_path, device_map="cuda")
    model = AutoModelForImageTextToText.from_pretrained(model_path, device_map="cuda")    


   
    model.eval()

    # 加载并预处理数据
    print(f"加载测试数据: {test_file}")
    raw_samples = load_test_data(test_file)
    print(f"共加载 {len(raw_samples)} 个原始样本")

    # 解析 GT 并排序
    parsed_samples = []
    for s in raw_samples:
        gt = json.loads(s["answer"])
        parsed_samples.append({
            "problem": s["problem"],
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
    valid_responses = 0
    
    print("\n开始串行回测...")
    
    # 使用 tqdm 监控总进度
    pbar = tqdm(total=len(parsed_samples), desc="Processing")
    
    processed_count = 0
    # import ipdb; ipdb.set_trace()
    for seq_id, group in groupby(parsed_samples, key=lambda x: x["sequence_id"]):
        items = list(group)
        # import ipdb; ipdb.set_trace()
        
        # 初始化序列状态
        first_gt = items[0]["gt"]
        prev_action = first_gt.get("initial_action", 0.0)
        prev_vol = first_gt.get("initial_volatility")
        prev_scale = volatility_target / (prev_vol + 1e-8)
        
        last_idx = items[0]["start_idx"] - 1
        
        # 使用索引遍历，以便在需要时跳过样本
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
            
            # 1. 推理
            messages = [{"role": "user", "content": [{"type": "text", "text": item["problem"]}]}]
            inputs = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=2048)
           
            response = processor.decode(outputs[0][inputs["input_ids"].shape[-1]:])
            print(response)

            # 2. 解析所有动作
            actions = parse_model_response(response)
            
            if any(a != 0.0 for a in actions) or "positions" in response:
                valid_responses += 1
            
            # 3. 对每个动作计算奖励
            # 每个动作应该使用对应位置的数据样本
            num_actions = len(actions)
            
            # 确定实际可使用的动作数量（受限于剩余的 items 数量）
            max_steps = min(num_actions, len(items) - item_idx)
            
            if max_steps == 0:
                # 如果没有可用动作或没有剩余样本，至少处理第一个
                max_steps = 1
            
            # 遍历所有动作并计算奖励
            for step_idx in range(max_steps):
                if item_idx + step_idx >= len(items):
                    break
                    
                action_t = actions[step_idx]
                positions.append(action_t)
                
                # 获取对应位置的数据样本
                current_item = items[item_idx + step_idx]
                gt_prices = current_item["gt"].get("prices", [])
                gt_vols = current_item["gt"].get("volatilities", [])
                
                # 获取当前价格
                if len(gt_prices) > 0:
                    price_t = gt_prices[0]
                else:
                    price_t = 0.0
                
                # 获取下一个价格（优先使用当前 item 的 prices[1]，否则使用下一个 item 的 prices[0]）
                if len(gt_prices) > 1:
                    price_next = gt_prices[1]
                elif item_idx + step_idx + 1 < len(items):
                    next_item = items[item_idx + step_idx + 1]
                    next_prices = next_item["gt"].get("prices", [])
                    price_next = next_prices[0] if len(next_prices) > 0 else price_t
                else:
                    price_next = price_t
                
                # 获取波动率
                vol_t = gt_vols[0] if len(gt_vols) > 0 else 0.15
                
                scale_t = volatility_target / (vol_t + 1e-8)
                
                # 收益 (Profit)
                r_t_next = price_next - price_t
                profit = action_t * scale_t * r_t_next
                
                # 交易成本 (Cost)
                position_change = abs(scale_t * action_t - prev_scale * prev_action)
                cost = transaction_cost_bp * price_t * position_change
                
                step_reward = mu * (profit - cost)
                
                rewards.append(step_reward)
                total_reward += step_reward
                cumulative_wealth += step_reward
                
                # 打印中间结果
                tqdm.write(f"Step {processed_count}[{step_idx}/{max_steps-1}]: ItemIdx={item_idx+step_idx}, Action={action_t:.4f}, PrevAction={prev_action:.4f}, PriceChange={r_t_next:.2f}, Profit={profit:.2f}, Cost={cost:.2f}, Reward={step_reward:.2f}")
                
                # 更新状态
                prev_action = action_t
                prev_scale = scale_t
                processed_count += 1
                
                pbar.set_postfix({"Wealth": f"{cumulative_wealth:.0f}", "LastReward": f"{step_reward:.2f}"})
            
            # 检查可能的解析错误
            if max_steps == 0 or all(a == 0.0 for a in actions):
                if "positions" not in response and "position" not in response:
                    tqdm.write(f"  [Warning] Possible Parse Error. Response: {response[:100]}...")
            
            # 关键：跳过已使用的样本
            # 如果使用了 N 个动作，需要跳过 N 个样本（当前样本 + 接下来的 N-1 个）
            # 但实际跳过的步数取决于实际使用的动作数量
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
        "valid_response_rate": valid_responses / len(parsed_samples) if parsed_samples else 0,
        "processed_samples": processed_count,
        "mean_reward": np.mean(rewards_np) if len(rewards_np) > 0 else 0.0,
        "std_reward": np.std(rewards_np) if len(rewards_np) > 0 else 0.0,
        "max_drawdown": 0.0,
        "sharpe_ratio": 0.0,
        "avg_position": np.mean(np.abs(positions_np)) if len(positions_np) > 0 else 0.0,
        "long_count": int(np.sum(positions_np > 0.01)),
        "short_count": int(np.sum(positions_np < -0.01)),
        "neutral_count": int(np.sum(np.abs(positions_np) <= 0.01))
    }
    
    # 计算最大回撤
    if len(rewards) > 0:
        cum_returns = np.cumsum(rewards)
        running_max = np.maximum.accumulate(cum_returns)
        drawdown = running_max - cum_returns
        stats["max_drawdown"] = np.max(drawdown)
    
    # 计算夏普比率
    if len(rewards) > 1 and np.std(rewards) > 1e-6:
        stats["sharpe_ratio"] = (np.mean(rewards) / np.std(rewards)) * np.sqrt(252)
        
    # 打印结果
    print("\n" + "="*60)
    print("评估结果 (Sequential Backtest)")
    print("="*60)
    print(f"初始财富: {stats['initial_wealth']:.2f}")
    print(f"最终财富: {stats['final_wealth']:.2f}")
    print(f"累计收益: {stats['total_reward']:.2f}")
    print(f"收益率: {stats['return_rate']:.2f}%")
    print(f"夏普比率: {stats['sharpe_ratio']:.4f}")
    print(f"最大回撤: {stats['max_drawdown']:.2f}")
    print(f"平均每步收益: {stats['mean_reward']:.2f}")
    print("-" * 30)
    print(f"总样本数: {stats['processed_samples']}")
    print(f"有效响应率: {stats['valid_response_rate']*100:.1f}%")
    print(f"持仓分布: 多 {stats['long_count']} | 空 {stats['short_count']} | 平 {stats['neutral_count']}")
    print(f"平均绝对持仓: {stats['avg_position']:.4f}")
    print("="*60)
    
    return stats


if __name__ == "__main__":
    # 配置
  
    # model_path = "Qwen/Qwen3-VL-4B-Instruct"  # 可以改为本地 checkpoint 路径或纯文本模型名称
    model_path = "/home/tione/notebook/workspace/xiaoyangchen/work/EasyR1/checkpoints/easy_r1/qwen3_4b_instruct_guba_grpo_mse/global_step_125/actor/huggingface"  # 可以改为本地 checkpoint 路径或纯文本模型名称
    # model_path = "/home/tione/notebook/workspace/xiaoyangchen/work/LLaMA-Factory/output/qwen3vl_guba_lora_sft"
   
    
    test_file = "/home/tione/notebook/workspace/xiaoyangchen/work/data/guba/guba_eval_verl.jsonl"
    
    if not os.path.exists(test_file):
        print(f"错误: 找不到测试文件: {test_file}")
        sys.exit(1)
        
    print(f"使用模型路径: {model_path}")
    
    results = evaluate_model(
        model_path,
        test_file,
        use_fast_generation=True
    )
    
    # 保存结果
    output_file = Path(model_path).parent / "evaluation_results_sequential_rl_mse_eval.json" if model_path else Path("evaluation_results_sequential.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        # 转换 numpy 类型为 python 类型以便 json 序列化
        def convert(o):
            if isinstance(o, np.generic): return o.item()
            raise TypeError
        json.dump(results, f, indent=2, ensure_ascii=False, default=convert)
    print(f"\n结果已保存到: {output_file}")
