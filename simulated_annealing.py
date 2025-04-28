import itertools
import random
import math
import time
from collections import defaultdict
from typing import List, Tuple, Set

class SimulatedAnnealingOptimizer:
    def __init__(self, samples: List[str], j: int, s: int, k: int, f: int = 1):
        if not (3 <= s <= j <= k <= len(samples)):
            raise ValueError(f"Invalid parameters: 3≤s≤j≤k≤样本数, got s={s}, j={j}, k={k}, samples={len(samples)}")
        if f < 1:
            raise ValueError("覆盖次数f必须≥1")
        self.samples = samples
        self.j = j
        self.s = s
        self.k = k
        self.f = f
        self.progress_callback = None
        
        # 生成所有j大小的子集
        self.j_subsets = list(itertools.combinations(samples, j))
        print(f"Total j-subsets: {len(self.j_subsets)}")
    
    def set_progress_callback(self, callback):
        """Set progress callback function"""
        self.progress_callback = callback
    
    def _update_progress(self, progress_percent, message=None):
        """Update progress through callback if available"""
        if self.progress_callback:
            self.progress_callback(progress_percent, message)
    
    def init_solution(self) -> List[Tuple[str, ...]]:
        """初始化解：贪心方法生成初始k大小的组合集合"""
        # 初始化空解
        solution = []
        
        # 跟踪已覆盖的j子集
        covered_j_subsets = defaultdict(int)
        
        # 候选组合：所有可能的k大小组合
        candidate_groups = list(itertools.combinations(self.samples, self.k))
        
        # 计算每个候选组覆盖了多少个j子集
        while not self._all_covered(covered_j_subsets) and candidate_groups:
            # 计算每个候选组的贡献度（能新覆盖多少个j子集）
            contributions = []
            for group in candidate_groups:
                group_set = set(group)
                contribution = 0
                for j_sub in self.j_subsets:
                    # 如果该j子集已被充分覆盖，则跳过
                    if covered_j_subsets.get(j_sub, 0) >= self.f:
                        continue
                    
                    # 检查该组是否覆盖了这个j子集（即是否有至少s个样本重叠）
                    j_set = set(j_sub)
                    if len(j_set.intersection(group_set)) >= self.s:
                        contribution += 1
                
                contributions.append((group, contribution))
            
            # 如果没有贡献度大于0的组，则随机添加一个
            valid_contributions = [(g, c) for g, c in contributions if c > 0]
            if not valid_contributions:
                # 尝试随机几次寻找有贡献的组合
                for _ in range(10):  # 增加尝试次数
                    random_group = tuple(sorted(random.sample(self.samples, self.k)))
                    new_contribution = 0
                    for j_sub in self.j_subsets:
                        if covered_j_subsets.get(j_sub, 0) >= self.f:
                            continue
                        j_set = set(j_sub)
                        if len(j_set.intersection(set(random_group))) >= self.s:
                            new_contribution += 1
                    
                    if new_contribution > 0 and random_group not in solution:
                        solution.append(random_group)
                        # 更新覆盖情况
                        for j_sub in self.j_subsets:
                            j_set = set(j_sub)
                            if len(j_set.intersection(set(random_group))) >= self.s:
                                covered_j_subsets[j_sub] += 1
                        break
                break
            
            # 选取贡献度最高的组，如果有多个贡献度相同的组，随机选择
            sorted_contributions = sorted(valid_contributions, key=lambda x: x[1], reverse=True)
            max_contribution = sorted_contributions[0][1]
            best_groups = [g for g, c in sorted_contributions if c == max_contribution]
            best_group = random.choice(best_groups)
            
            # 添加到解中
            solution.append(best_group)
            
            # 更新覆盖情况
            for j_sub in self.j_subsets:
                j_set = set(j_sub)
                if len(j_set.intersection(set(best_group))) >= self.s:
                    covered_j_subsets[j_sub] += 1
            
            # 从候选中移除已选组
            candidate_groups.remove(best_group)
        
        # 确保我们有一个有效解
        if not solution:
            # 如果贪心算法失败，随机生成一些组合
            for _ in range(min(10, len(self.j_subsets))):
                new_group = tuple(sorted(random.sample(self.samples, self.k)))
                if new_group not in solution:
                    solution.append(new_group)
        
        return solution
    
    def _all_covered(self, covered_dict):
        """检查是否所有j子集都已被充分覆盖"""
        return all(covered_dict.get(j_sub, 0) >= self.f for j_sub in self.j_subsets)
    
    def evaluate_solution(self, solution: List[Tuple[str, ...]]) -> Tuple[float, float]:
        """评估解的能量（越低越好）和覆盖率"""
        # 跟踪每个j子集被覆盖的次数
        coverage_count = defaultdict(int)
        
        # 计算每个j子集被覆盖的次数
        for j_sub in self.j_subsets:
            j_set = set(j_sub)
            for group in solution:
                if len(j_set.intersection(set(group))) >= self.s:
                    coverage_count[j_sub] += 1
        
        # 计算完全覆盖的j子集数量
        covered_count = sum(1 for cnt in coverage_count.values() if cnt >= self.f)
        coverage_ratio = covered_count / len(self.j_subsets) if self.j_subsets else 1.0
        
        # 计算能量：首先优先覆盖率，其次是组合数量
        # 如果覆盖率低于100%，则主要考虑覆盖率
        if coverage_ratio < 1.0:
            energy = (1.0 - coverage_ratio) * 1000 + len(solution) * 0.1
        else:
            # 如果覆盖率为100%，则主要考虑组合数量
            energy = len(solution)
        
        return energy, coverage_ratio
    
    def generate_neighbor(self, solution: List[Tuple[str, ...]]) -> List[Tuple[str, ...]]:
        """生成邻居解"""
        # 复制当前解
        neighbor = list(solution)
        
        # 根据覆盖情况动态调整操作概率
        coverage_count = defaultdict(int)
        for j_sub in self.j_subsets:
            j_set = set(j_sub)
            for group in neighbor:
                if len(j_set.intersection(set(group))) >= self.s:
                    coverage_count[j_sub] += 1
        
        coverage_ratio = sum(1 for cnt in coverage_count.values() if cnt >= self.f) / len(self.j_subsets)
        
        # 如果覆盖率低，增加添加操作的概率；如果覆盖率高，增加删除操作的概率
        if coverage_ratio < 0.95:
            weights = [0.6, 0.2, 0.2]  # add, remove, replace
        elif coverage_ratio < 1.0:
            weights = [0.4, 0.3, 0.3]  # add, remove, replace
        else:
            weights = [0.1, 0.7, 0.2]  # add, remove, replace
        
        # 随机选择操作类型
        op_type = random.choices(
            ["add", "remove", "replace"],
            weights=weights,
            k=1
        )[0]
        
        # 操作1: 添加一个组合
        if op_type == "add":
            # 找出覆盖不足的j子集
            uncovered_j_subs = [j_sub for j_sub in self.j_subsets 
                              if coverage_count[j_sub] < self.f]
            
            # 如果有覆盖不足的子集，尝试添加一个能覆盖它们的组合
            if uncovered_j_subs and random.random() < 0.9:  # 增加针对性添加的概率
                # 随机选择一个覆盖不足的j子集
                target_j = random.choice(uncovered_j_subs)
                
                # 确保新组合至少包含这个j子集中的s个样本
                samples_from_j = random.sample(target_j, self.s)
                remaining_samples = list(set(self.samples) - set(samples_from_j))
                
                # 补齐剩下的k-s个样本
                if len(remaining_samples) >= self.k - self.s:
                    remaining = random.sample(remaining_samples, self.k - self.s)
                    new_group = tuple(sorted(samples_from_j + remaining))
                    
                    # 确保不重复添加
                    if new_group not in neighbor:
                        neighbor.append(new_group)
                        return neighbor
                
                # 如果上面的方法失败，尝试一个更加贪心的方法
                # 找出能覆盖最多未覆盖j子集的新组合
                best_coverage = 0
                best_group = None
                for _ in range(20):  # 尝试20次寻找一个好的组合
                    candidate = tuple(sorted(random.sample(self.samples, self.k)))
                    if candidate in neighbor:
                        continue
                    
                    coverage = 0
                    for j_sub in uncovered_j_subs:
                        j_set = set(j_sub)
                        if len(j_set.intersection(set(candidate))) >= self.s:
                            coverage += 1
                    
                    if coverage > best_coverage:
                        best_coverage = coverage
                        best_group = candidate
                
                if best_group and best_coverage > 0:
                    neighbor.append(best_group)
            else:
                # 随机添加一个组合
                attempts = 0
                while attempts < 10:  # 增加尝试次数
                    new_group = tuple(sorted(random.sample(self.samples, self.k)))
                    if new_group not in neighbor:
                        neighbor.append(new_group)
                        break
                    attempts += 1
        
        # 操作2: 移除一个组合
        elif op_type == "remove" and len(neighbor) > 1:  # 至少保留一个组合
            # 计算每个组合的贡献度
            contributions = []
            
            # 首先计算当前总覆盖情况
            full_coverage = defaultdict(int)
            for j_sub in self.j_subsets:
                j_set = set(j_sub)
                for group in neighbor:
                    if len(j_set.intersection(set(group))) >= self.s:
                        full_coverage[j_sub] += 1
            
            # 计算移除每个组合后的损失
            for group in neighbor:
                loss = 0
                critical_loss = 0  # 会导致覆盖率下降的损失
                group_set = set(group)
                
                # 计算这个组合对覆盖的贡献
                for j_sub in self.j_subsets:
                    j_set = set(j_sub)
                    # 如果这个组合覆盖了这个j子集
                    if len(j_set.intersection(group_set)) >= self.s:
                        loss += 1
                        # 如果这个j子集正好被覆盖f次，移除这个组合会导致覆盖不足
                        if full_coverage[j_sub] <= self.f:
                            critical_loss += 1
                
                contributions.append((group, loss, critical_loss))
            
            # 按关键损失和总损失排序，优先移除不会降低覆盖率且损失较低的
            sorted_contributions = sorted(contributions, key=lambda x: (x[2], x[1]))
            
            # 动态调整移除策略
            if coverage_ratio < 1.0:
                # 如果覆盖率尚未达到100%，只移除不会导致关键损失的组合
                safe_groups = [g for g, _, c in sorted_contributions if c == 0]
                if safe_groups:
                    to_remove = random.choice(safe_groups)
                    neighbor.remove(to_remove)
            else:
                # 如果已经达到100%覆盖，尝试更激进的移除
                # 90%概率移除最安全的组合，10%概率随机移除
                if random.random() < 0.9 and sorted_contributions:
                    min_critical_loss = sorted_contributions[0][2]
                    if min_critical_loss == 0:  # 只移除安全的
                        safe_groups = [g for g, _, c in sorted_contributions if c == 0]
                        if safe_groups:
                            to_remove = random.choice(safe_groups)
                            neighbor.remove(to_remove)
                else:
                    to_remove = random.choice(neighbor)
                    neighbor.remove(to_remove)
        
        # 操作3: 替换一个组合
        elif op_type == "replace" and neighbor:
            # 先找出对覆盖贡献最小的组合
            contributions = []
            
            # 计算每个组合覆盖了多少个j子集
            for group in neighbor:
                group_set = set(group)
                contribution = 0
                unique_contribution = 0  # 唯一贡献的j子集数
                
                for j_sub in self.j_subsets:
                    j_set = set(j_sub)
                    if len(j_set.intersection(group_set)) >= self.s:
                        contribution += 1
                        
                        # 检查这个j子集是否只被这个组合覆盖
                        covered_by_others = False
                        for other_group in neighbor:
                            if other_group == group:
                                continue
                            if len(j_set.intersection(set(other_group))) >= self.s:
                                covered_by_others = True
                                break
                        
                        if not covered_by_others:
                            unique_contribution += 1
                
                contributions.append((group, contribution, unique_contribution))
            
            # 按唯一贡献排序，优先替换贡献小且没有唯一贡献的组合
            sorted_contributions = sorted(contributions, key=lambda x: (x[2], x[1]))
            
            # 根据覆盖率和唯一贡献选择要替换的组合
            if coverage_ratio < 1.0:
                # 如果覆盖率未达到100%，避免替换有唯一贡献的组合
                replaceable = [g for g, _, u in sorted_contributions if u == 0]
                if replaceable:
                    to_replace = random.choice(replaceable)
                    idx = neighbor.index(to_replace)
                    
                    # 找出未覆盖的j子集
                    uncovered = [j_sub for j_sub in self.j_subsets if coverage_count[j_sub] < self.f]
                    
                    # 尝试找一个能提高覆盖率的替代品
                    best_new_coverage = 0
                    best_new_group = None
                    
                    for _ in range(20):  # 增加尝试次数
                        candidate = tuple(sorted(random.sample(self.samples, self.k)))
                        if candidate in neighbor:
                            continue
                        
                        new_coverage = 0
                        for j_sub in uncovered:
                            j_set = set(j_sub)
                            if len(j_set.intersection(set(candidate))) >= self.s:
                                new_coverage += 1
                        
                        if new_coverage > best_new_coverage:
                            best_new_coverage = new_coverage
                            best_new_group = candidate
                    
                    if best_new_group and best_new_coverage > 0:
                        neighbor[idx] = best_new_group
                    else:
                        # 随机替换
                        attempts = 0
                        while attempts < 10:
                            new_group = tuple(sorted(random.sample(self.samples, self.k)))
                            if new_group not in neighbor:
                                neighbor[idx] = new_group
                                break
                            attempts += 1
            else:
                # 如果已经100%覆盖，尝试任意替换不会导致覆盖率下降的组合
                if sorted_contributions and sorted_contributions[0][2] == 0:
                    to_replace = sorted_contributions[0][0]
                    idx = neighbor.index(to_replace)
                    
                    # 随机替换
                    attempts = 0
                    while attempts < 10:
                        new_group = tuple(sorted(random.sample(self.samples, self.k)))
                        if new_group not in neighbor:
                            old_neighbor = list(neighbor)
                            neighbor[idx] = new_group
                            
                            # 验证替换后覆盖率不会下降
                            new_coverage_count = defaultdict(int)
                            for j_sub in self.j_subsets:
                                j_set = set(j_sub)
                                for group in neighbor:
                                    if len(j_set.intersection(set(group))) >= self.s:
                                        new_coverage_count[j_sub] += 1
                            
                            new_coverage_ratio = sum(1 for cnt in new_coverage_count.values() if cnt >= self.f) / len(self.j_subsets)
                            
                            if new_coverage_ratio >= coverage_ratio:
                                break
                            else:
                                # 恢复原解
                                neighbor = old_neighbor
                            
                        attempts += 1
        
        return neighbor
    
    def acceptance_probability(self, current_energy: float, new_energy: float, temperature: float) -> float:
        """计算是否接受新解的概率"""
        if new_energy < current_energy:  # 如果新解更好，总是接受
            return 1.0
        else:
            # 否则，根据能量差和温度计算接受概率
            return math.exp((current_energy - new_energy) / temperature)
    
    def optimize(self, initial_temp=10.0, cooling_rate=0.995, 
                stopping_temp=0.0001, max_iterations=100000, verbose=True):
        """模拟退火算法主过程"""
        # 初始化解和能量
        current_solution = self.init_solution()
        current_energy, current_coverage = self.evaluate_solution(current_solution)
        
        # 跟踪最优解
        best_solution = list(current_solution)
        best_energy = current_energy
        best_coverage = current_coverage
        
        temperature = initial_temp
        iteration = 0
        
        # 没有改进的迭代次数
        no_improve = 0
        max_no_improve = 2000  # 如果2000次迭代没有改进，重启
        
        # 用于记录进度
        prev_report_time = time.time()
        report_interval = 2.0  # 每2秒报告一次进度
        
        # 重启计数器
        restarts = 0
        max_restarts = 5  # 最多重启5次
        
        while temperature > stopping_temp and iteration < max_iterations and restarts < max_restarts:
            # 计算进度百分比
            progress = min(100, int((1 - temperature/initial_temp) * 100))
            self._update_progress(progress, f"Temperature: {temperature:.6f}, Current energy: {current_energy:.4f}")
            
            # 生成邻居解
            new_solution = self.generate_neighbor(current_solution)
            new_energy, new_coverage = self.evaluate_solution(new_solution)
            
            # 决定是否接受新解
            accept_probability = self.acceptance_probability(current_energy, new_energy, temperature)
            if accept_probability > random.random():
                current_solution = new_solution
                current_energy = new_energy
                current_coverage = new_coverage
                
                # 检查是否为新的最优解
                if new_energy < best_energy or (new_energy == best_energy and new_coverage > best_coverage):
                    best_solution = list(new_solution)
                    best_energy = new_energy
                    best_coverage = new_coverage
                    no_improve = 0
                    
                    if verbose:
                        self._update_progress(progress, f"New best solution found! Coverage: {best_coverage:.4f}, Groups: {len(best_solution)}")
                else:
                    no_improve += 1
            else:
                no_improve += 1
            
            # 如果长时间没有改进，重启搜索
            if no_improve >= max_no_improve:
                if best_coverage >= 1.0:
                    # 如果已经找到了完全覆盖的解，进行优化重启
                    if verbose:
                        self._update_progress(progress, f"No improvement for {max_no_improve} iterations, optimizing restart ({restarts+1}/{max_restarts})")
                    
                    # 从最优解开始，尝试删除一个组合
                    if len(best_solution) > 1:
                        optimization_attempt = list(best_solution)
                        # 随机删除一个组合
                        optimization_attempt.pop(random.randrange(len(optimization_attempt)))
                        
                        # 验证覆盖率
                        _, optimization_coverage = self.evaluate_solution(optimization_attempt)
                        
                        if optimization_coverage >= 1.0:
                            # 如果删除后仍然完全覆盖，采用新解
                            current_solution = optimization_attempt
                            current_energy, current_coverage = self.evaluate_solution(current_solution)
                            
                            # 更新最优解
                            best_solution = list(current_solution)
                            best_energy = current_energy
                            best_coverage = current_coverage
                            
                            if verbose:
                                self._update_progress(progress, f"Optimization successful! New group count: {len(best_solution)}")
                        else:
                            # 否则重新初始化
                            current_solution = self.init_solution()
                            current_energy, current_coverage = self.evaluate_solution(current_solution)
                    else:
                        # 如果只有一个组合，无法进一步优化，结束算法
                        break
                else:
                    # 如果尚未找到完全覆盖的解，普通重启
                    if verbose:
                        self._update_progress(progress, f"No improvement for {max_no_improve} iterations, normal restart ({restarts+1}/{max_restarts})")
                    
                    # 重新初始化解
                    current_solution = self.init_solution()
                    current_energy, current_coverage = self.evaluate_solution(current_solution)
                
                no_improve = 0
                restarts += 1
                
                # 恢复更高的温度以进行更广泛的搜索
                temperature = initial_temp * (0.8 ** restarts)
            
            # 降低温度
            temperature *= cooling_rate
            iteration += 1
            
            # 定期报告进度
            current_time = time.time()
            if verbose and current_time - prev_report_time > report_interval:
                self._update_progress(progress, f"Iteration {iteration}: Temperature = {temperature:.6f}, Current energy = {current_energy:.4f}, "
                      f"Best energy = {best_energy:.4f}, Coverage = {best_coverage:.4f}, Groups = {len(best_solution)}")
                prev_report_time = current_time
            
            # 如果已找到完全覆盖且组数较小的解，提前终止
            if best_coverage >= 1.0 and len(best_solution) <= self.k and no_improve > 500:
                if verbose:
                    self._update_progress(100, "Ideal solution found, terminating early")
                break
        
        if verbose:
            self._update_progress(100, f"\nOptimization complete! Total iterations: {iteration}, Restarts: {restarts}")
            self._update_progress(100, f"Best solution coverage: {best_coverage:.4f}")
            self._update_progress(100, f"Best solution contains {len(best_solution)} groups")
        
        # 如果最终覆盖率未达到100%，尝试补充
        if best_coverage < 1.0 and verbose:
            self._update_progress(100, "Warning: Could not find 100% coverage solution, attempting to supplement...")
            
            # 找出未覆盖的j子集
            coverage_count = defaultdict(int)
            for j_sub in self.j_subsets:
                j_set = set(j_sub)
                for group in best_solution:
                    if len(j_set.intersection(set(group))) >= self.s:
                        coverage_count[j_sub] += 1
            
            uncovered = [j_sub for j_sub in self.j_subsets if coverage_count[j_sub] < self.f]
            self._update_progress(100, f"Uncovered j-subsets: {len(uncovered)}/{len(self.j_subsets)}")
            
            # 尝试贪心补充
            while uncovered and len(best_solution) < 100:  # 设置上限以防无限循环
                # 找到能覆盖最多未覆盖j子集的组合
                best_coverage_improvement = 0
                best_new_group = None
                
                for _ in range(100):  # 增加尝试次数
                    candidate = tuple(sorted(random.sample(self.samples, self.k)))
                    if candidate in best_solution:
                        continue
                    
                    coverage_improvement = 0
                    for j_sub in uncovered:
                        j_set = set(j_sub)
                        if len(j_set.intersection(set(candidate))) >= self.s:
                            coverage_improvement += 1
                    
                    if coverage_improvement > best_coverage_improvement:
                        best_coverage_improvement = coverage_improvement
                        best_new_group = candidate
                
                if best_new_group and best_coverage_improvement > 0:
                    best_solution.append(best_new_group)
                    
                    # 更新未覆盖列表
                    new_coverage_count = defaultdict(int)
                    for j_sub in self.j_subsets:
                        j_set = set(j_sub)
                        for group in best_solution:
                            if len(j_set.intersection(set(group))) >= self.s:
                                new_coverage_count[j_sub] += 1
                    
                    uncovered = [j_sub for j_sub in self.j_subsets if new_coverage_count[j_sub] < self.f]
                    
                    if verbose:
                        self._update_progress(100, f"After adding group, uncovered j-subsets: {len(uncovered)}/{len(self.j_subsets)}")
                else:
                    # 无法找到能改进覆盖的组合，退出循环
                    break
            
            # 最终检查覆盖情况
            _, final_coverage = self.evaluate_solution(best_solution)
            if verbose:
                self._update_progress(100, f"Final coverage after supplementation: {final_coverage:.4f}, Groups: {len(best_solution)}")
        
        return best_solution

def get_int(prompt, min_val, max_val):
    """获取验证过的整数输入"""
    while True:
        try:
            value = int(input(prompt))
            if (min_val is not None and value < min_val) or \
                    (max_val is not None and value > max_val):
                raise ValueError
            return value
        except ValueError:
            print(f"输入无效，请输入{min_val}-{max_val}之间的整数")

def select_samples(params):
    def random_samples(params):
        """生成随机样本"""
        all_samples = [f"{i:02d}" for i in range(1, params['m'] + 1)]
        selected = random.sample(all_samples, params['n'])
        print(f"\n随机选择的样本: {', '.join(sorted(selected))}")
        return selected
    
    choice = input("\n选择样本方式:\n1. 随机选择\n2. 手动输入\n请选择(1/2): ")
    
    def manual_input(params):
        """处理手动输入"""
        print(f"\n请输入{params['n']}个样本编号(01-{params['m']:02d}), 用逗号分隔")
        while True:
            try:
                inputs = input("输入样本: ").split(',')
                samples = [s.strip().zfill(2) for s in inputs]
                if len(samples) != params['n']:
                    raise ValueError
                if any(not s.isdigit() or int(s) > params['m'] for s in samples):
                    raise ValueError
                return sorted(list(set(samples)))  # 去重并排序
            except:
                print(f"输入无效，请确保输入{params['n']}个有效的样本编号")
    
    if choice == '1':
        return random_samples(params)
    elif choice == '2':
        return manual_input(params)
    else:
        print("无效选择，使用随机方式")
        return random_samples(params)

def show_results(solution):
    """显示结果"""
    print(f"\n最优解包含 {len(solution)} 个组合:")
    for i, group in enumerate(solution, 1):
        print(f"组合{i}: {', '.join(group)}")

def verify_coverage(solution, samples, j, s, f=1):
    """验证解是否满足覆盖要求"""
    # 生成所有j大小的子集
    j_subsets = list(itertools.combinations(samples, j))
    
    # 记录每个j子集被覆盖的次数
    coverage_count = defaultdict(int)
    
    # 计算覆盖情况
    for j_sub in j_subsets:
        j_set = set(j_sub)
        for group in solution:
            group_set = set(group)
            if len(j_set.intersection(group_set)) >= s:
                coverage_count[j_sub] += 1
    
    # 检查所有j子集是否都被覆盖至少f次
    covered = all(coverage_count[j_sub] >= f for j_sub in j_subsets)
    
    coverage_ratio = sum(1 for cnt in coverage_count.values() if cnt >= f) / len(j_subsets)
    
    return covered, coverage_ratio, len(j_subsets)

if __name__ == "__main__":
    print("基于模拟退火算法的最小覆盖组合优化")
    """获取用户参数输入"""
    params = {
        'm': get_int("总样本数m (45-54): ", 45, 54),
        'n': get_int("选择样本数n (7-25): ", 7, 25),
        'k': get_int("组合大小k (4-7): ", 4, 7),
        'j': get_int("子集参数j (>=s): ", 3, None),
        's': get_int("覆盖参数s (3-7): ", 3, 7),
        'f': get_int("覆盖次数f (≥1): ", 1, None),
        'run_id': int(time.time())
    }
    
    # 确保j >= s
    params['j'] = max(params['j'], params['s'])
    
    # 选择样本
    samples = select_samples(params)
    
    # 创建优化器并运行
    start_time = time.time()
    optimizer = SimulatedAnnealingOptimizer(samples, params['j'], params['s'], params['k'], params['f'])
    
    solution = optimizer.optimize(
        initial_temp=10.0, 
        cooling_rate=0.995, 
        stopping_temp=0.0001, 
        max_iterations=100000, 
        verbose=True
    )
    
    end_time = time.time()
    
    # 验证解的覆盖情况
    covered, coverage_ratio, total_j_subsets = verify_coverage(
        solution, samples, params['j'], params['s'], params['f']
    )
    
    # 显示结果
    print(f"\n计算耗时: {end_time - start_time:.2f}秒")
    print(f"j子集总数: {total_j_subsets}, 覆盖率: {coverage_ratio:.4f}")
    print(f"覆盖要求满足: {'是' if covered else '否'}")
    
    show_results(solution) 