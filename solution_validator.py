import numpy as np
import itertools
# 修改tqdm导入，创建无操作版本
from tqdm import tqdm as original_tqdm
# 创建禁用的tqdm函数
def disabled_tqdm(iterable, *args, **kwargs):
    return iterable
# 替换全局tqdm
tqdm = disabled_tqdm
import time
import math
import random
from scipy import stats
from functools import lru_cache

class SolutionValidator:
    """
    验证样本组合的有效性，并计算置信度
    """
    def __init__(self, samples, j, s, k, f):
        """
        初始化验证器
        samples: 所有样本列表
        j: j参数（子集大小）
        s: s参数（覆盖参数）
        k: k参数（组合大小）
        f: f参数（至少f个s样本）
        """
        self.samples = samples
        self.j = j
        self.s = s
        self.k = k
        self.f = f
        self.n = len(samples)
        
        # 预计算所有可能的s组合
        self.all_s_combinations = list(itertools.combinations(samples, s))
        self.total_s_combinations = len(self.all_s_combinations)
        
        # 缓存配置
        self._evaluation_cache = {}
        
        # 增强配置
        self.advanced_metrics = True       # 使用高级评估指标
        self.ensemble_weights = {          # 集成评分权重配置
            "basic": 0.25,                 # 基础覆盖率评分
            "statistical": 0.20,           # 统计分析评分
            "entropy": 0.15,               # 熵评分
            "bayesian": 0.15,              # 贝叶斯推断评分
            "optimization": 0.15,          # 优化度评分
            "minmax": 0.10                 # 最小最大分析评分
        }
        
        # 调试模式 - 当置信度为零时捕获详细信息
        self.debug_mode = True
        self.min_confidence_threshold = 0.05  # 最小置信度阈值，低于此值会触发调试信息
        
    def validate_solution(self, solution):
        """
        验证一个解决方案是否满足所有条件
        solution: 包含样本组合的列表
        返回: (是否有效, 置信度, 详细信息)
        """
        if not solution:
            return False, 0.0, {"error": "Empty solution"}
        
        # 查找缓存结果
        solution_key = self._get_solution_hash(solution)
        if solution_key in self._evaluation_cache:
            return self._evaluation_cache[solution_key]
        
        # 检查1：验证每个组合的大小是否为k
        size_check = all(len(group) == self.k for group in solution)
        if not size_check:
            invalid_groups = [i for i, group in enumerate(solution) if len(group) != self.k]
            result = (False, 0.0, {"error": f"Some combinations don't have size k: {invalid_groups}"})
            self._evaluation_cache[solution_key] = result
            return result
        
        # 检查2：验证所有样本都是有效的
        valid_samples = all(sample in self.samples for group in solution for sample in group)
        if not valid_samples:
            invalid_samples = [sample for group in solution for sample in group if sample not in self.samples]
            result = (False, 0.0, {"error": f"Solution contains invalid samples: {invalid_samples}"})
            self._evaluation_cache[solution_key] = result
            return result
        
        # 检查3：验证s覆盖约束 - 统计每个s组合被覆盖的次数
        s_coverage = self.calculate_s_coverage(solution)
        
        # 找出未满足覆盖次数的组合
        uncovered_combinations = [comb for comb, count in s_coverage.items() if count < self.f]
        coverage_satisfied = len(uncovered_combinations) == 0
        coverage_ratio = sum(1 for count in s_coverage.values() if count >= self.f) / max(1, self.total_s_combinations)
        
        # 检查整体有效性
        is_valid = size_check and valid_samples and coverage_satisfied
        
        # 计算置信度指标 - 如果启用高级评估，使用增强型评估方法
        try:
            if self.advanced_metrics:
                confidence, metrics = self.calculate_advanced_confidence(solution, s_coverage, coverage_ratio)
            else:
                confidence = self.calculate_confidence(solution, s_coverage, coverage_ratio)
                metrics = {"basic_confidence": confidence}
                
            # 调试低置信度情况
            if self.debug_mode and confidence < self.min_confidence_threshold:
                self._debug_low_confidence(solution, s_coverage, coverage_ratio, metrics)
                
            # 确保置信度不完全为零，除非解决方案无效
            if is_valid and confidence < self.min_confidence_threshold:
                confidence = max(confidence, self.min_confidence_threshold)
            
        except Exception as e:
            # 出错时提供一个基础置信度
            confidence = 0.1 if is_valid else 0.0
            metrics = {"error": str(e)}
        
        # 详细信息
        details = {
            "valid": is_valid,
            "size_check": size_check,
            "valid_samples": valid_samples, 
            "coverage_satisfied": coverage_satisfied,
            "coverage_ratio": coverage_ratio,
            "s_combinations_covered": sum(1 for count in s_coverage.values() if count >= self.f),
            "total_s_combinations": self.total_s_combinations,
            "solution_size": len(solution),
            "confidence": confidence,
            "metrics": metrics
        }
        
        result = (is_valid, confidence, details)
        self._evaluation_cache[solution_key] = result
        return result
    
    def _debug_low_confidence(self, solution, s_coverage, coverage_ratio, metrics):
        """收集低置信度情况的调试信息"""
        # 不记录日志，只收集调试信息
        debug_info = {
            "solution_size": len(solution),
            "coverage_ratio": coverage_ratio,
            "min_coverage": min(s_coverage.values()) if s_coverage else 0,
            "max_coverage": max(s_coverage.values()) if s_coverage else 0,
            "avg_coverage": sum(s_coverage.values()) / len(s_coverage) if s_coverage else 0,
            "metrics": metrics
        }
        # 调试信息已收集，但不记录日志
        
    def _get_solution_hash(self, solution):
        """生成解决方案的唯一哈希值，用于缓存"""
        try:
            return hash(frozenset(frozenset(group) for group in solution))
        except:
            # 降级处理
            solution_str = str(sorted([sorted(group) for group in solution]))
            return hash(solution_str)
    
    def calculate_s_coverage(self, solution):
        """计算每个s组合被覆盖的次数"""
        coverage = {s_comb: 0 for s_comb in self.all_s_combinations}
        
        for group in solution:
            # 获取该组合中所有可能的s大小子集
            group_s_combinations = list(itertools.combinations(group, self.s))
            
            # 更新覆盖计数
            for s_comb in group_s_combinations:
                if s_comb in coverage:
                    coverage[s_comb] += 1
        
        return coverage
    
    def calculate_advanced_confidence(self, solution, s_coverage, coverage_ratio):
        """
        高级置信度计算系统 - 使用多种评估方法的集成
        """
        metrics = {}
        
        try:
            # 1. 基础评分 (基本覆盖和冗余度)
            basic_score = self.calculate_basic_score(solution, s_coverage, coverage_ratio)
            metrics["basic_score"] = basic_score
            
            # 2. 统计分析评分 (分布分析和变异系数)
            statistical_score = self.calculate_statistical_score(s_coverage)
            metrics["statistical_score"] = statistical_score
            
            # 3. 熵评分 (信息理论度量)
            entropy_score = self.calculate_entropy_score(solution, s_coverage)
            metrics["entropy_score"] = entropy_score
            
            # 4. 贝叶斯推断评分
            bayesian_score = self.calculate_bayesian_score(solution, s_coverage)
            metrics["bayesian_score"] = bayesian_score
            
            # 5. 优化度评分 (理论最优解分析)
            optimization_score = self.calculate_optimization_score(solution)
            metrics["optimization_score"] = optimization_score
            
            # 6. 最小最大分析 (最坏情况分析)
            minmax_score = self.calculate_minmax_score(solution, s_coverage)
            metrics["minmax_score"] = minmax_score
            
            # 7. 集成所有评分 (加权平均)
            ensemble_score = (
                self.ensemble_weights["basic"] * basic_score +
                self.ensemble_weights["statistical"] * statistical_score +
                self.ensemble_weights["entropy"] * entropy_score + 
                self.ensemble_weights["bayesian"] * bayesian_score +
                self.ensemble_weights["optimization"] * optimization_score +
                self.ensemble_weights["minmax"] * minmax_score
            )
            
            # 确保最终分数在[0,1]范围内
            final_score = max(0.01, min(1.0, ensemble_score))  # 保证至少0.01
            
            # 添加所有评分到指标字典
            metrics["ensemble_score"] = ensemble_score
            metrics["final_score"] = final_score
            
            return final_score, metrics
            
        except Exception as e:
            # 发生错误时回退到基础评分
            basic_score = self.calculate_basic_score(solution, s_coverage, coverage_ratio)
            return max(0.01, basic_score), {"basic_score": basic_score, "error": str(e)}
    
    def calculate_basic_score(self, solution, s_coverage, coverage_ratio):
        """基础评分计算 - 包括覆盖率和冗余度"""
        try:
            # 指标1: 覆盖率 (60%)
            coverage_score = coverage_ratio
            
            # 指标2: 最小冗余度 (20%)
            # 检查是否有过度覆盖（有些s组合被覆盖次数远高于f）
            if not s_coverage:
                redundancy_score = 0
            else:
                # 理想情况下，每个s组合被覆盖f次
                coverage_values = np.array(list(s_coverage.values()))
                target_coverage = self.f
                deviation = np.abs(coverage_values - target_coverage)
                max_deviation = max(self.n, target_coverage)  # 最大可能偏差
                redundancy_score = 1 - (np.mean(deviation) / max_deviation)
            
            # 指标3: 组合有效性 (20%)
            # 每个组合应该尽可能地覆盖不同的s组合，避免冗余
            if not solution:
                efficiency_score = 0
            else:
                # 计算每个组合覆盖的唯一s组合数量
                unique_coverage = []
                
                for group in solution:
                    group_s_combs = set(itertools.combinations(group, self.s))
                    unique_coverage.append(len(group_s_combs))
                
                # 理想情况下，每个组合应该覆盖C(k,s)个不同的s组合
                max_possible = min(self.total_s_combinations, len(self.all_s_combinations))
                efficiency_score = sum(unique_coverage) / (len(solution) * max_possible) if max_possible > 0 else 0
            
            # 加权计算最终置信度
            score = (0.6 * coverage_score) + (0.2 * redundancy_score) + (0.2 * efficiency_score)
            
            # 确保置信度在0-1范围内，且有效解至少有最小置信度
            return max(0.01, min(1, score))
            
        except Exception as e:
            # 出错时返回一个最小置信度
            return 0.01
    
    def calculate_confidence(self, solution, s_coverage, coverage_ratio):
        """
        原始置信度计算方法 - 保留向后兼容性
        """
        return self.calculate_basic_score(solution, s_coverage, coverage_ratio)
    
    def calculate_statistical_score(self, s_coverage):
        """
        统计分析评分 - 分析覆盖次数的统计特性
        """
        if not s_coverage:
            return 0.0
            
        # 获取所有s组合的覆盖次数
        coverage_values = np.array(list(s_coverage.values()))
        
        # 计算统计指标
        mean = np.mean(coverage_values)
        median = np.median(coverage_values)
        std_dev = np.std(coverage_values)
        
        # 理想情况: 均值接近目标覆盖次数f，标准差较小
        target = self.f
        
        # 计算均值与目标的接近度 (0-1分，越接近1越好)
        mean_score = 1.0 - min(1.0, abs(mean - target) / max(1, target))
        
        # 计算变异系数 (标准差/均值)，评估覆盖均匀性
        # 变异系数越小越好，理想情况是0
        # 防止除零
        cv = std_dev / max(0.001, mean) if mean > 0 else float('inf')
        cv_score = 1.0 / (1.0 + cv)  # 转换为0-1分
        
        # 计算覆盖率分布的偏度
        # 如果偏度接近0，表示分布更均匀
        try:
            skewness = stats.skew(coverage_values)
            skewness_score = 1.0 / (1.0 + abs(skewness))
        except:
            skewness_score = 0.5  # 默认值
        
        # 综合评分 (加权平均)
        score = 0.5 * mean_score + 0.3 * cv_score + 0.2 * skewness_score
        
        return max(0.0, min(1.0, score))
    
    def calculate_entropy_score(self, solution, s_coverage):
        """
        信息熵评分 - 使用信息理论评估覆盖分布的均匀性
        """
        if not s_coverage or not solution:
            return 0.0
        
        # 计算覆盖次数的频率分布
        coverage_values = list(s_coverage.values())
        min_coverage = min(coverage_values)
        max_coverage = max(coverage_values)
        
        # 如果所有覆盖次数相同，直接给出完美分数
        if min_coverage == max_coverage:
            return 1.0
        
        # 计算频率分布
        freq = {}
        for val in coverage_values:
            if val in freq:
                freq[val] += 1
            else:
                freq[val] = 1
        
        # 计算概率分布
        total = len(coverage_values)
        probabilities = [count / total for count in freq.values()]
        
        # 计算熵
        entropy = -sum(p * math.log2(p) for p in probabilities if p > 0)
        
        # 计算最大可能熵 (均匀分布的熵)
        max_entropy = math.log2(len(freq))
        
        # 归一化熵 (0-1之间)
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        
        # 计算未覆盖率
        uncovered = sum(1 for count in coverage_values if count < self.f) / len(coverage_values)
        
        # 组合评分: 熵评分与覆盖率
        entropy_score = 0.6 * normalized_entropy + 0.4 * (1 - uncovered)
        
        return max(0.0, min(1.0, entropy_score))
    
    def calculate_bayesian_score(self, solution, s_coverage):
        """
        贝叶斯推断评分 - 基于贝叶斯概率模型
        """
        if not solution or not s_coverage:
            return 0.0
        
        # 获取覆盖计数
        coverage_values = np.array(list(s_coverage.values()))
        
        # 先验 - 假定理想情况下均匀覆盖f次
        prior_mean = self.f
        prior_std = max(1.0, self.f * 0.1)  # 确保标准差不为零
        
        # 似然 - 基于观察到的覆盖分布
        observed_mean = np.mean(coverage_values)
        # 确保标准差不为零，至少为0.1
        observed_std = max(0.1, np.std(coverage_values) if len(coverage_values) > 1 else 0.1)
        
        # 贝叶斯更新 (简化的贝叶斯更新)
        # 预先计算并检查分母是否为零
        prior_precision = 1.0 / (prior_std ** 2)
        observed_precision = 1.0 / (observed_std ** 2)
        denominator = prior_precision + observed_precision
        
        # 避免除以零
        if denominator <= 1e-10:  # 使用小的阈值而不是零
            posterior_mean = observed_mean  # 如果分母接近零，直接使用观察均值
        else:
            posterior_mean = (prior_mean * prior_precision + observed_mean * observed_precision) / denominator
        
        # 计算后验概率与目标f的接近度
        mean_error = abs(posterior_mean - self.f) / max(1, self.f)
        mean_score = 1.0 - min(1.0, mean_error)
        
        # 计算有效覆盖比例 (达到f次的s组合比例)
        effective_coverage = np.sum(coverage_values >= self.f) / len(coverage_values)
        
        # 计算最优覆盖比例 - 理论上的最优覆盖
        theoretical_coverage = min(1.0, len(solution) * self._combinations(self.k, self.s) / max(1, self.total_s_combinations))
        
        # 计算相对覆盖率
        relative_coverage = effective_coverage / theoretical_coverage if theoretical_coverage > 0 else 0
        
        # 综合评分
        bayesian_score = 0.4 * mean_score + 0.6 * relative_coverage
        
        return max(0.0, min(1.0, bayesian_score))
    
    def calculate_optimization_score(self, solution):
        """
        优化度评分 - 评估解决方案与理论最优解的接近程度
        """
        if not solution:
            return 0.0
        
        # 计算理论最小组合数
        min_combinations_theory = self.estimate_min_combinations()
        
        # 计算实际使用的组合数
        actual_combinations = len(solution)
        
        # 计算优化比率 (理论最小组合数 / 实际组合数)
        # 比率越接近1甚至超过1，说明效率越高
        optimization_ratio = min_combinations_theory / actual_combinations if actual_combinations > 0 else 0
        
        # 计算每个组合的平均覆盖效率
        max_s_per_k = self._combinations(self.k, self.s)  # 每个k组合最多可覆盖的s组合数
        
        # 评估每个组合实际覆盖的s组合比例
        coverage_counts = []
        for group in solution:
            group_s_combs = list(itertools.combinations(group, self.s))
            coverage_counts.append(len(group_s_combs))
        
        avg_coverage = np.mean(coverage_counts) if coverage_counts else 0
        efficiency_ratio = avg_coverage / max_s_per_k if max_s_per_k > 0 else 0
        
        # 组合两个评分
        optimization_score = 0.6 * min(1.0, optimization_ratio) + 0.4 * efficiency_ratio
        
        return max(0.0, min(1.0, optimization_score))
    
    def calculate_minmax_score(self, solution, s_coverage):
        """
        最小最大分析 - 评估最坏情况下的表现
        """
        if not solution or not s_coverage:
            return 0.0
        
        # 获取覆盖计数
        coverage_values = list(s_coverage.values())
        
        # 找出最小覆盖次数
        min_coverage = min(coverage_values) if coverage_values else 0
        
        # 计算最小覆盖比例 (最小覆盖次数 / 目标f)
        min_coverage_ratio = min_coverage / self.f if self.f > 0 else 0
        
        # 计算最小覆盖组合的比例
        min_coverage_count = sum(1 for v in coverage_values if v == min_coverage)
        min_count_ratio = min_coverage_count / len(coverage_values) if coverage_values else 1
        
        # 组合评分 - 惩罚最弱环节
        # 如果最小覆盖次数低于f，或者有大量组合是最小覆盖，得分会很低
        minmax_score = min_coverage_ratio * (1 - 0.5 * min_count_ratio)
        
        return max(0.0, min(1.0, minmax_score))
    
    @lru_cache(maxsize=128)
    def _combinations(self, n, k):
        """计算组合数 C(n,k)，使用缓存提高性能"""
        if k < 0 or k > n:
            return 0
        if k == 0 or k == n:
            return 1
            
        # 计算组合数
        result = 1
        for i in range(1, k + 1):
            result = result * (n - (i - 1)) // i
            
        return result
    
    def monte_carlo_validation(self, solution, iterations=100):
        """
        增强的蒙特卡洛方法验证 - 使用更复杂的随机模拟和对比分析
        """
        if not solution:
            return 0.05, {}  # 返回一个最小置信度而不是0
        
        # 获取当前解决方案的详细信息
        is_valid, curr_confidence, curr_details = self.validate_solution(solution)
        
        # 如果解决方案无效，返回较低但非零的置信度
        if not is_valid:
            return 0.05, {"warning": "Invalid solution", "details": curr_details}
        
        # 生成随机解决方案进行比较
        random_solutions = []
        random_confidences = []
        random_metrics = []
        
        # 使用进度条跟踪模拟进度
        for _ in tqdm(range(iterations), desc="Monte Carlo Simulation", leave=False):
            try:
                # 随机生成两种类型的解决方案
                if random.random() < 0.7:
                    # 70%概率: 小扰动基础上的随机方案 (更接近当前解)
                    random_sol = self._generate_perturbed_solution(solution)
                else:
                    # 30%概率: 完全随机方案
                    random_sol = self._generate_random_solution(len(solution))
                
                # 验证随机解决方案
                rand_valid, rand_confidence, rand_details = self.validate_solution(random_sol)
                
                # 只记录有效的随机解
                if rand_valid:
                    random_confidences.append(rand_confidence)
                    
                    if 'metrics' in rand_details:
                        random_metrics.append(rand_details['metrics'])
            except Exception:
                continue
        
        # 计算蒙特卡洛置信度 (当前解在随机解中的百分位数)
        if not random_confidences:
            monte_carlo_confidence = 1.0
        else:
            # 当前解决方案优于多少百分比的随机解决方案
            monte_carlo_confidence = sum(1 for rc in random_confidences if curr_confidence > rc) / len(random_confidences)
        
        # 计算当前解决方案相对于随机解决方案的各项指标优势
        metric_advantages = {}
        if random_metrics and 'metrics' in curr_details:
            curr_metrics = curr_details['metrics']
            for metric in curr_metrics:
                if all(metric in rm for rm in random_metrics):
                    random_values = [rm[metric] for rm in random_metrics]
                    # 计算百分位数
                    metric_advantages[f"{metric}_percentile"] = sum(1 for v in random_values if curr_metrics[metric] > v) / len(random_values)
        
        # 提取高级指标的统计数据
        metric_stats = {}
        if random_metrics:
            # 收集所有随机方案的指标值，计算各指标的统计特性
            for metric in random_metrics[0]:
                values = [rm[metric] for rm in random_metrics if metric in rm]
                if values:
                    metric_stats[metric] = {
                        "mean": np.mean(values),
                        "std": np.std(values),
                        "min": np.min(values),
                        "max": np.max(values),
                        "current": curr_details['metrics'].get(metric, 0) if 'metrics' in curr_details else 0
                    }
        
        details = {
            "standard_confidence": curr_confidence,
            "monte_carlo_confidence": monte_carlo_confidence,
            "metric_advantages": metric_advantages,
            "metric_stats": metric_stats,
            "details": curr_details
        }
        
        # 结合多种评估结果作为最终置信度
        # 使用高级集成方法融合多种评估指标
        final_confidence = self._calculate_ensemble_monte_carlo(curr_confidence, monte_carlo_confidence, metric_advantages)
        
        # 确保有效解决方案的最小置信度
        final_confidence = max(0.01, final_confidence)
        
        return final_confidence, details
    
    def _calculate_ensemble_monte_carlo(self, base_confidence, monte_carlo, advantages):
        """计算蒙特卡洛模拟的集成置信度"""
        # 基础置信度权重
        w_base = 0.5
        
        # 蒙特卡洛模拟权重
        w_monte = 0.3
        
        # 各指标优势权重
        w_adv = 0.2
        
        # 计算指标优势分数
        adv_score = 0
        if advantages:
            # 提取关键指标的优势
            key_advantages = [v for k, v in advantages.items() 
                             if "score" in k or "confidence" in k]
            adv_score = sum(key_advantages) / len(key_advantages) if key_advantages else 0
        
        # 计算加权和
        ensemble = w_base * base_confidence + w_monte * monte_carlo + w_adv * adv_score
        
        return max(0.0, min(1.0, ensemble))
    
    def _generate_perturbed_solution(self, base_solution):
        """生成基于当前解决方案的小扰动随机解"""
        if not base_solution:
            return self._generate_random_solution(3)  # 默认生成3个组合
            
        # 复制基础解决方案
        new_solution = [list(group) for group in base_solution]
        
        # 随机决定扰动程度
        perturbation_level = random.random()  # 0-1之间
        
        # 基于扰动程度确定修改的组数
        groups_to_modify = max(1, int(len(new_solution) * perturbation_level * 0.7))
        
        # 随机选择要修改的组
        for _ in range(groups_to_modify):
            # 随机选择一个组
            if not new_solution:  # 安全检查
                break
                
            group_idx = random.randrange(len(new_solution))
            
            # 确定操作类型: 0=修改组内样本，1=完全替换组，2=删除组，3=添加新组
            operation = random.randint(0, 3)
            
            if operation == 0:
                # 修改组内的样本
                if new_solution[group_idx]:
                    # 随机替换1-3个样本
                    elements_to_replace = random.randint(1, min(3, len(new_solution[group_idx])))
                    # 确保有足够的样本可供选择
                    if len(self.samples) < elements_to_replace:
                        elements_to_replace = len(self.samples)
                    
                    # 安全检查：确保新解的组不为空
                    if not elements_to_replace or len(new_solution[group_idx]) < elements_to_replace:
                        continue
                        
                    # 安全替换
                    try:
                        replacements = random.sample(self.samples, elements_to_replace)
                        positions = random.sample(range(len(new_solution[group_idx])), elements_to_replace)
                        for pos, new_elem in zip(positions, replacements):
                            new_solution[group_idx][pos] = new_elem
                    except ValueError:
                        # 如果采样出错，跳过此操作
                        continue
            
            elif operation == 1:
                # 完全替换组
                # 确保有足够的样本可供选择
                if len(self.samples) >= self.k:
                    try:
                        new_solution[group_idx] = random.sample(self.samples, self.k)
                    except ValueError:
                        # 如果采样出错，跳过此操作
                        continue
                
            elif operation == 2 and len(new_solution) > 1:
                # 删除组 (保留至少一个组)
                new_solution.pop(group_idx)
                
            elif operation == 3:
                # 添加新组
                if len(self.samples) >= self.k:
                    try:
                        new_group = random.sample(self.samples, self.k)
                        new_solution.append(new_group)
                    except ValueError:
                        # 如果采样出错，跳过此操作
                        continue
        
        # 确保返回的解不为空
        if not new_solution:
            return self._generate_random_solution(1)
            
        return new_solution
    
    def _generate_random_solution(self, size):
        """生成完全随机的解决方案"""
        random_sol = []
        for _ in range(size):
            # 每个组合包含k个随机选择的样本
            random_group = random.sample(self.samples, min(self.k, len(self.samples)))
            random_sol.append(random_group)
        return random_sol
    
    def benchmark_solution(self, solution):
        """
        增强的基准测试 - 使用高级评估指标
        """
        # 验证基本有效性
        is_valid, confidence, details = self.validate_solution(solution)
        
        if not is_valid:
            return {
                "valid": False,
                "confidence": 0.0,
                "details": details
            }
        
        try:
            # 进行蒙特卡洛验证
            start_time = time.time()
            monte_carlo_confidence, mc_details = self.monte_carlo_validation(solution, iterations=50)
            mc_time = time.time() - start_time
            
            # 计算解决方案的优化程度
            # 理论最小组合数
            min_combinations_theory = self.estimate_min_combinations()
            optimization_ratio = min_combinations_theory / len(solution) if len(solution) > 0 else 0
            
            # 提取高级评估指标
            advanced_metrics = {}
            if 'metrics' in details:
                advanced_metrics = details['metrics']
            
            # 组合结果
            benchmark_result = {
                "valid": True,
                "confidence": monte_carlo_confidence,
                "standard_confidence": confidence,
                "monte_carlo_confidence": mc_details["monte_carlo_confidence"],
                "monte_carlo_time": mc_time,
                "combinations_count": len(solution),
                "theoretical_min_combinations": min_combinations_theory,
                "optimization_ratio": optimization_ratio,
                "advanced_metrics": advanced_metrics,
                "metric_advantages": mc_details.get("metric_advantages", {}),
                "details": details
            }
            
            return benchmark_result
            
        except Exception as e:
            # 出错时返回简化的结果
            return {
                "valid": True,
                "confidence": max(0.01, confidence),
                "error": str(e),
                "combinations_count": len(solution),
                "details": details
            }
    
    def estimate_min_combinations(self):
        """估算理论上可能的最小组合数量"""
        # 每个k组合最多可以覆盖C(k,s)个s组合
        max_s_per_k = self._combinations(self.k, self.s)
        
        # 需要覆盖所有s组合至少f次
        total_coverage_needed = self.total_s_combinations * self.f
        
        # 理论最小组合数（向上取整）
        min_combinations = np.ceil(total_coverage_needed / max_s_per_k)
        
        return max(1, int(min_combinations))
    
    def compare_solutions(self, solutions_dict):
        """
        比较多个解决方案的置信度
        solutions_dict: {算法名称: 解决方案}
        返回: 比较结果字典
        """
        results = {}
        
        # 使用进度条跟踪评估进度
        for alg_name, solution in tqdm(solutions_dict.items(), desc="Evaluating Solutions"):
            try:
                # 进行基准测试
                benchmark = self.benchmark_solution(solution)
                
                # 保存结果
                results[alg_name] = {
                    "valid": benchmark["valid"],
                    "confidence": benchmark["confidence"],
                    "combinations_count": benchmark["combinations_count"] if "combinations_count" in benchmark else 0,
                    "optimization_ratio": benchmark["optimization_ratio"] if "optimization_ratio" in benchmark else 0,
                    "benchmark": benchmark,
                    "advanced_metrics": benchmark.get("advanced_metrics", {})
                }
                
            except Exception as e:
                # 出错时提供默认结果
                results[alg_name] = {
                    "valid": False,
                    "confidence": 0.0,
                    "error": str(e)
                }
        
        # 添加比较信息
        if results:
            # 找出最高置信度
            valid_results = [r for r in results.values() if r["valid"]]
            max_confidence = max(r["confidence"] for r in valid_results) if valid_results else 0.01
            
            # 更新相对置信度
            for alg_name in results:
                if results[alg_name]["valid"]:
                    results[alg_name]["relative_confidence"] = results[alg_name]["confidence"] / max_confidence if max_confidence > 0 else 0
                else:
                    results[alg_name]["relative_confidence"] = 0
            
            # 为每个算法添加排名信息
            self._add_ranking_data(results)
        
        return results
    
    def _add_ranking_data(self, results):
        """添加排名数据到结果字典"""
        if not results:
            return
            
        # 按不同指标对算法进行排名
        metrics_to_rank = [
            ("confidence", True),           # 置信度 (越高越好)
            ("optimization_ratio", True),   # 优化比率 (越高越好)
            ("combinations_count", False)   # 组合数量 (越少越好)
        ]
        
        # 添加高级指标（如果存在）
        advanced_metrics = set()
        for r in results.values():
            if "advanced_metrics" in r:
                advanced_metrics.update(r["advanced_metrics"].keys())
        
        for metric in advanced_metrics:
            metrics_to_rank.append((f"advanced_metrics.{metric}", True))
        
        # 计算每个指标的排名
        for metric_path, higher_is_better in metrics_to_rank:
            # 提取指标值
            metric_values = {}
            for alg_name, result in results.items():
                # 支持嵌套路径，如 advanced_metrics.entropy_score
                value = result
                for part in metric_path.split('.'):
                    if part in value:
                        value = value[part]
                    else:
                        value = 0
                        break
                
                if result["valid"]:  # 只对有效解决方案排名
                    metric_values[alg_name] = float(value)
            
            # 排序算法
            sorted_algs = sorted(
                metric_values.keys(),
                key=lambda a: metric_values[a],
                reverse=higher_is_better
            )
            
            # 添加排名
            for rank, alg_name in enumerate(sorted_algs, 1):
                # 创建排名字典（如果不存在）
                if "rankings" not in results[alg_name]:
                    results[alg_name]["rankings"] = {}
                
                # 添加该指标的排名
                results[alg_name]["rankings"][metric_path] = rank
        
        # 计算综合排名
        for alg_name, result in results.items():
            if "rankings" in result:
                # 简单平均所有排名
                avg_rank = sum(result["rankings"].values()) / len(result["rankings"])
                result["average_rank"] = avg_rank 