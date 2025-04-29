import itertools
import random
import math
import time
from collections import defaultdict
from typing import List, Tuple, Set, Dict

class FunnelAlgorithm:
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
        
        # 预计算哪些样本组合可以覆盖哪些j子集
        self.coverage_map = self._precompute_coverage_map()
    
    def set_progress_callback(self, callback):
        """Set progress callback function"""
        self.progress_callback = callback
    
    def _update_progress(self, progress_percent, message=None):
        """Update progress through callback if available"""
        if self.progress_callback:
            return self.progress_callback(progress_percent, message)
        return True  # Continue by default if no callback
    
    def _precompute_coverage_map(self) -> Dict[Tuple[str, ...], List[int]]:
        """预计算覆盖关系，哪些j子集会被哪些k大小组合覆盖"""
        coverage_map = {}
        total_combinations = math.comb(len(self.samples), self.k)
        
        # 如果组合总数太大，先不预计算，等需要时再计算
        if total_combinations > 50000:
            return coverage_map
        
        print(f"Precomputing coverage relationships...")
        all_k_combinations = self.generate_all_k_combinations()
        for k_comb in all_k_combinations:
            k_set = set(k_comb)
            covered_j_indices = []
            for j_idx, j_sub in enumerate(self.j_subsets):
                j_set = set(j_sub)
                if len(j_set.intersection(k_set)) >= self.s:
                    covered_j_indices.append(j_idx)
            
            if covered_j_indices:  # 只保存有覆盖能力的组合
                coverage_map[k_comb] = covered_j_indices
        
        print(f"Precomputed coverage for {len(coverage_map)} combinations")
        return coverage_map
    
    def generate_all_k_combinations(self) -> List[Tuple[str, ...]]:
        """生成所有可能的k大小组合"""
        return list(itertools.combinations(self.samples, self.k))
    
    def evaluate_combination(self, combination: Tuple[str, ...], j_subsets: List[Tuple[str, ...]]) -> int:
        """评估一个组合覆盖了多少个j子集"""
        # 如果已经预计算过覆盖关系，直接使用
        if combination in self.coverage_map:
            return len(self.coverage_map[combination])
        
        covered_count = 0
        combination_set = set(combination)
        
        for j_subset in j_subsets:
            j_set = set(j_subset)
            if len(j_set.intersection(combination_set)) >= self.s:
                covered_count += 1
                
        return covered_count
    
    def generate_intelligent_initial_solution(self, j_indices_to_cover: List[int], max_size=None) -> List[Tuple[str, ...]]:
        """生成一个智能的初始解，基于贪心策略"""
        # 初始化空解和覆盖计数
        solution = []
        coverage_count = defaultdict(int)
        remaining_indices = set(j_indices_to_cover)
        
        # 预计算所有k大小组合的覆盖能力
        if not self.coverage_map:
            # 如果没有预计算，为了效率考虑随机生成一些组合
            all_combinations = []
            total_combinations = math.comb(len(self.samples), self.k)
            if total_combinations > 10000:
                # 随机采样生成组合
                target_size = min(10000, total_combinations)
                print(f"Sampling {target_size} combinations from {total_combinations} possible")
                
                combinations_set = set()
                while len(combinations_set) < target_size:
                    new_comb = tuple(sorted(random.sample(self.samples, self.k)))
                    combinations_set.add(new_comb)
                all_combinations = list(combinations_set)
            else:
                all_combinations = self.generate_all_k_combinations()
                
            # 计算每个组合的覆盖能力
            combination_coverage = {}
            for comb in all_combinations:
                k_set = set(comb)
                covered_indices = []
                for j_idx in j_indices_to_cover:
                    j_set = set(self.j_subsets[j_idx])
                    if len(j_set.intersection(k_set)) >= self.s:
                        covered_indices.append(j_idx)
                
                if covered_indices:  # 只保存有覆盖能力的组合
                    combination_coverage[comb] = covered_indices
        else:
            # 使用预计算的覆盖关系
            combination_coverage = {
                comb: [idx for idx in covered_indices if idx in j_indices_to_cover]
                for comb, covered_indices in self.coverage_map.items()
                if any(idx in j_indices_to_cover for idx in covered_indices)
            }
        
        # 贪心选择组合直到满足覆盖要求或达到最大大小
        while remaining_indices and (max_size is None or len(solution) < max_size):
            # 计算每个候选组合的贡献（能覆盖多少新的j子集）
            best_contribution = 0
            best_combo = None
            best_covered = []
            
            for combo, covered_indices in combination_coverage.items():
                # 计算这个组合能覆盖多少未充分覆盖的j子集
                contribution = sum(1 for idx in covered_indices 
                                  if idx in remaining_indices and coverage_count[idx] < self.f)
                
                # 如果有更好的贡献，更新最佳组合
                if contribution > best_contribution:
                    best_contribution = contribution
                    best_combo = combo
                    best_covered = covered_indices
            
            # 如果没有能改进的组合，随机选择一个
            if best_contribution == 0:
                # 尝试随机选择几次
                for _ in range(10):
                    random_combo = tuple(sorted(random.sample(self.samples, self.k)))
                    if random_combo in combination_coverage and random_combo not in solution:
                        best_combo = random_combo
                        best_covered = combination_coverage[random_combo]
                        break
                
                # 如果仍然没有找到，退出循环
                if best_combo is None:
                    break
            
            # 添加最佳组合到解中
            if best_combo is not None:
                solution.append(best_combo)
                
                # 更新覆盖情况
                for idx in best_covered:
                    coverage_count[idx] += 1
                    # 如果j子集已被充分覆盖，从剩余集合中移除
                    if coverage_count[idx] >= self.f and idx in remaining_indices:
                        remaining_indices.remove(idx)
                
                # 从候选中移除已选组合避免重复选择
                if best_combo in combination_coverage:
                    del combination_coverage[best_combo]
        
        return solution
    
    def optimize(self, max_iterations=1000, verbose=True):
        """改进的漏斗算法主函数"""
        # 初始化进度
        should_continue = self._update_progress(0, "Initializing funnel algorithm...")
        if not should_continue:
            return []
        
        # 第一阶段：智能初始解生成
        if verbose:
            print("Generating intelligent initial solution...")
        
        # 使用智能方法生成初始解
        solution = self.generate_intelligent_initial_solution(list(range(len(self.j_subsets))))
        
        if verbose:
            print(f"Initial solution: {len(solution)} combinations")
            # 验证初始解的覆盖率
            coverage_ratio = self.verify_coverage(solution)
            print(f"Initial coverage ratio: {coverage_ratio:.2%}")
        
        should_continue = self._update_progress(20, f"Generated initial solution with {len(solution)} combinations")
        if not should_continue:
            return solution
        
        # 第二阶段：渐进式改进
        if verbose:
            print("Starting progressive improvement...")
        
        # 跟踪每个j子集被覆盖的次数
        coverage_count = defaultdict(int)
        for j_idx in range(len(self.j_subsets)):
            for group in solution:
                group_set = set(group)
                j_set = set(self.j_subsets[j_idx])
                if len(j_set.intersection(group_set)) >= self.s:
                    coverage_count[j_idx] += 1
        
        # 追踪进度变量
        iterations = 0
        last_improvement = 0
        best_solution = list(solution)
        best_size = len(solution)
        
        # 迭代改进
        while iterations < max_iterations and iterations - last_improvement < 200:
            iterations += 1
            
            # 更新进度
            if iterations % 10 == 0:
                progress = 20 + min(70, (iterations / max_iterations * 70))
                should_continue = self._update_progress(progress, f"Iteration {iterations}: solution size = {len(solution)}")
                if not should_continue:
                    return best_solution
            
            # 尝试各种改进操作
            operation = random.choices(
                ["replace", "add", "remove", "local_optimize"],
                weights=[0.4, 0.1, 0.4, 0.1],
                k=1
            )[0]
            
            if operation == "replace" and solution:
                # 随机替换一个组合
                idx_to_replace = random.randrange(len(solution))
                old_group = solution[idx_to_replace]
                
                # 移除旧组合的覆盖贡献
                for j_idx in range(len(self.j_subsets)):
                    j_set = set(self.j_subsets[j_idx])
                    if len(j_set.intersection(set(old_group))) >= self.s:
                        coverage_count[j_idx] -= 1
                
                # 找出覆盖不足的j子集
                uncovered_indices = [j_idx for j_idx in range(len(self.j_subsets)) if coverage_count[j_idx] < self.f]
                
                if uncovered_indices:
                    # 生成新组合来针对性覆盖
                    new_group = None
                    best_coverage = 0
                    
                    # 尝试多次找到好的替代组合
                    for _ in range(20):
                        # 尝试从未覆盖的j子集中抽取部分样本
                        if uncovered_indices and random.random() < 0.7:
                            target_j = self.j_subsets[random.choice(uncovered_indices)]
                            # 抽取s个样本
                            samples_from_j = random.sample(list(target_j), min(self.s, len(target_j)))
                            # 补齐其余样本
                            remaining = random.sample(
                                list(set(self.samples) - set(samples_from_j)), 
                                self.k - len(samples_from_j)
                            )
                            candidate = tuple(sorted(samples_from_j + remaining))
                        else:
                            # 纯随机生成
                            candidate = tuple(sorted(random.sample(self.samples, self.k)))
                        
                        # 计算覆盖能力
                        coverage = 0
                        for j_idx in uncovered_indices:
                            j_set = set(self.j_subsets[j_idx])
                            if len(j_set.intersection(set(candidate))) >= self.s:
                                coverage += 1
                        
                        if coverage > best_coverage:
                            best_coverage = coverage
                            new_group = candidate
                    
                    # 如果找到了合适的替代组合
                    if new_group and new_group not in solution:
                        # 更新解
                        solution[idx_to_replace] = new_group
                        
                        # 更新覆盖贡献
                        for j_idx in range(len(self.j_subsets)):
                            j_set = set(self.j_subsets[j_idx])
                            if len(j_set.intersection(set(new_group))) >= self.s:
                                coverage_count[j_idx] += 1
                
            elif operation == "add":
                # 找出覆盖不足的j子集
                uncovered_indices = [j_idx for j_idx in range(len(self.j_subsets)) if coverage_count[j_idx] < self.f]
                
                if uncovered_indices:
                    # 智能生成能够覆盖这些j子集的新组合
                    new_solution = self.generate_intelligent_initial_solution(uncovered_indices, max_size=1)
                    
                    if new_solution and new_solution[0] not in solution:
                        new_group = new_solution[0]
                        # 添加到解中
                        solution.append(new_group)
                        
                        # 更新覆盖情况
                        for j_idx in range(len(self.j_subsets)):
                            j_set = set(self.j_subsets[j_idx])
                            if len(j_set.intersection(set(new_group))) >= self.s:
                                coverage_count[j_idx] += 1
            
            elif operation == "remove" and len(solution) > 1:
                # 尝试移除冗余组合
                # 计算每个组合的唯一贡献
                redundant_indices = []
                
                for i, group in enumerate(solution):
                    # 临时移除该组合
                    temp_coverage = defaultdict(int)
                    for j, other_group in enumerate(solution):
                        if i != j:  # 跳过当前组合
                            for j_idx in range(len(self.j_subsets)):
                                j_set = set(self.j_subsets[j_idx])
                                if len(j_set.intersection(set(other_group))) >= self.s:
                                    temp_coverage[j_idx] += 1
                    
                    # 检查移除后是否仍保持充分覆盖
                    is_redundant = all(temp_coverage[j_idx] >= self.f for j_idx in range(len(self.j_subsets)))
                    if is_redundant:
                        redundant_indices.append(i)
                
                # 如果找到了冗余组合，随机移除一个
                if redundant_indices:
                    idx_to_remove = random.choice(redundant_indices)
                    removed_group = solution.pop(idx_to_remove)
                    
                    # 更新覆盖情况
                    for j_idx in range(len(self.j_subsets)):
                        j_set = set(self.j_subsets[j_idx])
                        if len(j_set.intersection(set(removed_group))) >= self.s:
                            coverage_count[j_idx] -= 1
            
            elif operation == "local_optimize" and solution:
                # 局部优化：在保持覆盖率的情况下优化部分解
                # 随机选择解的一部分进行重新优化
                optimize_size = min(len(solution) // 2 + 1, 5)
                indices_to_optimize = random.sample(range(len(solution)), optimize_size)
                
                # 移除这些组合，并记录它们的覆盖贡献
                affected_j_indices = set()
                for idx in sorted(indices_to_optimize, reverse=True):
                    group = solution.pop(idx)
                    for j_idx in range(len(self.j_subsets)):
                        j_set = set(self.j_subsets[j_idx])
                        if len(j_set.intersection(set(group))) >= self.s:
                            coverage_count[j_idx] -= 1
                            if coverage_count[j_idx] < self.f:
                                affected_j_indices.add(j_idx)
                
                # 使用智能方法重新生成覆盖这些j子集的组合
                if affected_j_indices:
                    new_partial_solution = self.generate_intelligent_initial_solution(list(affected_j_indices))
                    
                    # 添加新生成的组合
                    for group in new_partial_solution:
                        if group not in solution:
                            solution.append(group)
                            # 更新覆盖情况
                            for j_idx in range(len(self.j_subsets)):
                                j_set = set(self.j_subsets[j_idx])
                                if len(j_set.intersection(set(group))) >= self.s:
                                    coverage_count[j_idx] += 1
            
            # 验证当前解的覆盖情况
            current_coverage = all(coverage_count[j_idx] >= self.f for j_idx in range(len(self.j_subsets)))
            
            # 如果找到了更好的解
            if current_coverage and len(solution) < best_size:
                best_solution = list(solution)
                best_size = len(solution)
                last_improvement = iterations
                
                if verbose:
                    print(f"Iteration {iterations}: Found better solution with {len(solution)} combinations")
        
        # 最后的优化 - 尝试移除冗余组合
        if verbose:
            print("Performing final optimization...")
        
        should_continue = self._update_progress(90, "Performing final optimization...")
        if not should_continue:
            return best_solution
        
        # 使用更彻底的方法移除冗余组合
        optimized_solution = self.remove_redundant_combinations(best_solution)
        
        # 计算最终覆盖率
        final_coverage = self.verify_coverage(optimized_solution)
        
        # 更新进度
        self._update_progress(100, f"Optimization complete. Final solution has {len(optimized_solution)} groups with {final_coverage:.2%} coverage")
        
        if verbose:
            print(f"Final solution: {len(optimized_solution)} combinations, coverage: {final_coverage:.2%}")
        
        return optimized_solution
    
    def remove_redundant_combinations(self, solution: List[Tuple[str, ...]]) -> List[Tuple[str, ...]]:
        """尝试移除冗余组合，保持完全覆盖"""
        if not solution:
            return []
        
        # 复制解以免修改原始解
        optimized = list(solution)
        
        # 计算每个j子集被覆盖的次数
        coverage_count = defaultdict(int)
        for j_idx in range(len(self.j_subsets)):
            for group in optimized:
                group_set = set(group)
                j_set = set(self.j_subsets[j_idx])
                if len(j_set.intersection(group_set)) >= self.s:
                    coverage_count[j_idx] += 1
        
        # 计算每个组合的贡献
        contribution_scores = []
        for i, group in enumerate(optimized):
            # 计算唯一贡献 - 只有该组合覆盖的j子集数量
            unique_contribution = 0
            critical_contribution = 0  # 恰好覆盖f次的贡献
            
            for j_idx in range(len(self.j_subsets)):
                j_set = set(self.j_subsets[j_idx])
                if len(j_set.intersection(set(group))) >= self.s:
                    if coverage_count[j_idx] == 1:
                        unique_contribution += 1
                    elif coverage_count[j_idx] <= self.f:
                        critical_contribution += 1
            
            contribution_scores.append((i, unique_contribution, critical_contribution))
        
        # 按贡献度排序，优先保留唯一贡献大的组合
        contribution_scores.sort(key=lambda x: (x[1], x[2]), reverse=True)
        
        # 从贡献最小的组合开始尝试移除
        for score_idx in range(len(contribution_scores) - 1, -1, -1):
            idx, unique, critical = contribution_scores[score_idx]
            
            # 如果组合有唯一贡献，不要移除
            if unique > 0:
                continue
            
            # 临时移除并检查覆盖情况
            temp_group = optimized[idx]
            optimized.pop(idx)
            
            # 重新计算覆盖情况
            temp_coverage = defaultdict(int)
            for group in optimized:
                for j_idx in range(len(self.j_subsets)):
                    j_set = set(self.j_subsets[j_idx])
                    if len(j_set.intersection(set(group))) >= self.s:
                        temp_coverage[j_idx] += 1
            
            # 如果移除后覆盖率下降，恢复这个组合
            if not all(temp_coverage[j_idx] >= self.f for j_idx in range(len(self.j_subsets))):
                optimized.insert(idx, temp_group)
            else:
                # 更新其他组合的贡献度评分
                coverage_count = temp_coverage
        
        # 再尝试一轮更随机的移除，可能找到整体更优的解
        improved = True
        while improved:
            improved = False
            
            # 随机打乱顺序，尝试不同的移除顺序
            indices = list(range(len(optimized)))
            random.shuffle(indices)
            
            for idx in indices:
                if len(optimized) <= 1:
                    break
                    
                # 临时移除并检查
                temp_group = optimized[idx]
                optimized.pop(idx)
                
                # 验证覆盖率
                is_valid = self.verify_coverage(optimized) >= 1.0
                
                if not is_valid:
                    # 如果覆盖率降低，恢复组合
                    optimized.insert(idx, temp_group)
                else:
                    # 成功移除一个冗余组合
                    improved = True
        
        return optimized
    
    def verify_coverage(self, solution: List[Tuple[str, ...]]) -> float:
        """验证解的覆盖率"""
        # 计算每个j子集被覆盖的次数
        coverage_count = defaultdict(int)
        
        for j_sub in self.j_subsets:
            j_set = set(j_sub)
            for group in solution:
                group_set = set(group)
                if len(j_set.intersection(group_set)) >= self.s:
                    coverage_count[j_sub] += 1
        
        # 计算覆盖率
        covered_count = sum(1 for cnt in coverage_count.values() if cnt >= self.f)
        return covered_count / len(self.j_subsets) if self.j_subsets else 1.0

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
    print("基于漏斗算法的最小覆盖组合优化")
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
    optimizer = FunnelAlgorithm(samples, params['j'], params['s'], params['k'], params['f'])
    
    solution = optimizer.optimize(
        max_iterations=1000, 
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