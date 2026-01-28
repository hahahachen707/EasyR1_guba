import numpy as np
import pandas as pd

class TradingMetrics:
    def __init__(self, returns, annualization=252):
        """
        初始化评价指标计算器
        
        参数:
            returns (pd.Series or np.array): 时间序列的奖励/回报序列 (R_t)
            annualization (int): 年化因子，日线数据通常为 252
        """
        if isinstance(returns, list):
            returns = pd.Series(returns)
        self.returns = pd.Series(returns)
        self.freq = annualization
        
    def get_metrics_summary(self):
        """返回包含所有论文指标的字典"""
        return {
            "E(R)": self.expected_return(),
            "Std(R)": self.standard_deviation(),
            "DD": self.downside_deviation(),
            "Sharpe": self.sharpe_ratio(),
            "Sortino": self.sortino_ratio(),
            "MDD": self.max_drawdown(),
            "Calmar": self.calmar_ratio(),
            "% +ve Returns": self.win_rate(),
            "Ave. P / Ave. L": self.profit_loss_ratio()
        }

    def expected_return(self):
        """1. E(R): 年化期望回报"""
        # 论文公式: mean(R) * 252
        return self.returns.mean() * self.freq

    def standard_deviation(self):
        """2. Std(R): 年化标准差"""
        # 论文公式: std(R) * sqrt(252)
        return self.returns.std() * np.sqrt(self.freq)

    def downside_deviation(self):
        """3. DD: 下行偏差 (Downside Deviation)"""
        # 论文定义: "annualised standard deviation of trade returns that are negative"
        # 提取负收益部分
        negative_returns = self.returns[self.returns < 0]
        
        if len(negative_returns) == 0:
            return 0.0
            
        # 计算负收益的标准差并年化
        # 注意：这里严格遵循论文文字描述 "std of negative returns"
        # 标准金融定义通常是 sqrt(mean(min(0, r)^2))，但为了复现论文，我们计算负样本的std
        return negative_returns.std() * np.sqrt(self.freq)

    def sharpe_ratio(self):
        """4. Sharpe: 年化夏普比率"""
        # 论文公式: E(R) / Std(R)
        er = self.expected_return()
        std = self.standard_deviation()
        
        if std == 0: 
            return 0.0
        return er / std

    def sortino_ratio(self):
        """5. Sortino: 索提诺比率"""
        # 论文公式: E(R) / Downside Deviation
        er = self.expected_return()
        dd = self.downside_deviation()
        
        if dd == 0: 
            return 0.0
        return er / dd

    def max_drawdown(self):
        """6. MDD: 最大回撤"""
        # 因为论文使用的是 Additive Profits (加法利润)，回撤是绝对数值的下降
        # 计算累计财富曲线
        cumulative_wealth = self.returns.cumsum()
        
        # 计算截止目前的最高点 (High Water Mark)
        peak = cumulative_wealth.cummax()
        
        # 计算当前值与最高点的差值
        drawdown = peak - cumulative_wealth
        
        # 获取最大回撤值
        mdd = drawdown.max()
        return mdd

    def calmar_ratio(self):
        """7. Calmar: 卡尔玛比率"""
        # 论文公式: Expected Annual Return / MDD
        er = self.expected_return()
        mdd = self.max_drawdown()
        
        if mdd == 0: 
            return 0.0
        return er / mdd

    def win_rate(self):
        """8. % +ve Returns: 胜率"""
        # 论文公式: 正收益次数 / 总次数
        positive_count = (self.returns > 0).sum()
        total_count = len(self.returns)
        
        if total_count == 0: 
            return 0.0
        return positive_count / total_count

    def profit_loss_ratio(self):
        """9. Ave. P / Ave. L: 盈亏比"""
        # 论文公式: 平均正收益 / abs(平均负收益)
        positive_returns = self.returns[self.returns > 0]
        negative_returns = self.returns[self.returns < 0]
        
        if len(positive_returns) == 0:
            avg_positive = 0.0
        else:
            avg_positive = positive_returns.mean()
            
        if len(negative_returns) == 0:
            avg_negative = 0.0
        else:
            avg_negative = negative_returns.mean()
        
        # 处理边界情况
        if avg_positive == 0.0 or avg_negative == 0.0: 
            return 0.0
        
        return avg_positive / abs(avg_negative)

# ==========================================
# 测试代码 (模拟论文 Table 2 的数据格式)
# ==========================================
if __name__ == "__main__":
    # 1. 创建模拟数据
    # 假设我们有 1000 天的交易数据，平均每天赚 0.5，波动率为 2.0
    np.random.seed(42)
    daily_rewards = np.random.normal(loc=0.05, scale=2.0, size=1000)
    
    # 转换为 Series
    rewards_series = pd.Series(daily_rewards)
    
    # 2. 计算指标
    metrics = TradingMetrics(rewards_series, annualization=252)
    results = metrics.get_metrics_summary()
    
    # 3. 打印结果
    print("-" * 30)
    print("Paper Replication Metrics:")
    print("-" * 30)
    df_res = pd.DataFrame(results, index=['Value']).T
    print(df_res)
    
    print("-" * 30)
    print("Interpretation:")
    print(f"E(R) (年化收益): {results['E(R)']:.4f}")
    print(f"Sharpe (夏普比率): {results['Sharpe']:.4f}")
    print(f"MDD (最大回撤-绝对值): {results['MDD']:.4f}")
    
