import time
import random
import concurrent.futures
import numpy as np
from greedy_algorithm import r2, cov, a2, b2
from itertools import combinations
from collections import defaultdict

class GreedyOptimizer:
    """贪心算法优化器类，与现有的遗传和模拟退火优化器接口一致"""
    
    def __init__(self, samples, j, s, k, f):
        """
        初始化优化器
        samples: 样本列表
        j: 子集参数
        s: 覆盖参数
        k: 组合大小
        f: 覆盖次数
        """
        self.samples = samples    # 样本列表
        self.j = j                # 子集参数
        self.s = s                # 覆盖参数
        self.k = k                # 组合大小
        self.f = f                # 覆盖次数
        self.progress_callback = None
        # 优化参数
        self.max_iterations = 5   # 最大迭代次数
        self.local_search_steps = 3  # 局部搜索步数
        self.random_factor = 0.3  # 随机因子
        self.tabu_tenure = 10     # 禁忌步长
        self.max_parallel_workers = 4  # 最大并行工作线程数
        # 缓存
        self._cache = {}
        
    def set_progress_callback(self, callback):
        """设置进度回调函数"""
        self.progress_callback = callback
        
    def optimize(self):
        """
        执行贪心算法优化
        返回最佳解决方案（最优组合结果）
        """
        if self.progress_callback:
            self.progress_callback(5, "正在准备贪心算法运行环境...")
        
        # 获取样本数量
        n = len(self.samples)
        
        # 构建样本ID映射 (从0开始到n-1)
        sample_to_idx = {sample: i for i, sample in enumerate(sorted(self.samples))}
        idx_to_sample = {i: sample for sample, i in sample_to_idx.items()}
        
        # 确定覆盖次数参数
        y = 'all' if self.f > 1 else 1
        
        if self.progress_callback:
            self.progress_callback(10, f"使用参数 n={n}, k={self.k}, j={self.j}, s={self.s}, y={y}")
        
        # 初始化缓存
        self._initialize_cache(n)
        
        try:
            # 先尝试直接使用原始算法获取基准解
            if self.progress_callback:
                self.progress_callback(15, "尝试使用原始算法获取基准解...")
            
            base_count = None
            if y == 'all':
                base_count = a2(n, self.s, self.k)
            else:
                base_count = b2(n, self.k, self.j, self.s, y if y != 'all' else 1)
                
            if self.progress_callback:
                if base_count:
                    self.progress_callback(20, f"原始算法找到最小组合数量: {base_count}")
                else:
                    self.progress_callback(20, "原始算法未找到可行解，尝试优化方法...")
            
            # 并行运行不同策略
            if self.progress_callback:
                self.progress_callback(25, "启动并行优化计算...")
            
            best_solution = None
            best_solution_size = float('inf')
            
            # 准备并行任务
            tasks = [
                ("标准贪心", lambda: self._run_standard_greedy(n, y)),
                ("约束贪心", lambda: self._run_constrained_greedy(n, y, base_count if base_count else n//2)),
                ("随机贪心1", lambda: self._run_randomized_greedy(n, y, 1)),
                ("随机贪心2", lambda: self._run_randomized_greedy(n, y, 2)),
                ("禁忌搜索", lambda: self._run_tabu_search(n, y))
            ]
            
            # 设置并行执行器
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(tasks), self.max_parallel_workers)) as executor:
                # 提交所有任务
                future_to_task = {executor.submit(task): name for name, task in tasks}
                completed = 0
                
                # 处理完成的任务
                for future in concurrent.futures.as_completed(future_to_task):
                    task_name = future_to_task[future]
                    try:
                        solution = future.result()
                        completed += 1
                        
                        # 更新进度
                        if self.progress_callback:
                            progress = 25 + (completed * 45 // len(tasks))
                            self.progress_callback(progress, f"完成 {task_name} 策略计算 ({completed}/{len(tasks)})")
                        
                        # 如果找到了解并且比当前最佳解更好，则更新
                        if solution and (best_solution is None or len(solution) < best_solution_size):
                            best_solution = solution
                            best_solution_size = len(solution)
                            
                            if self.progress_callback:
                                self.progress_callback(progress + 2, f"找到更优解，组合数量: {best_solution_size} (来自{task_name})")
                    
                    except Exception as e:
                        if self.progress_callback:
                            self.progress_callback(70, f"{task_name}执行出错: {str(e)}")
            
            # 如果找到了解决方案，对其应用局部搜索进行最终优化
            if best_solution:
                if self.progress_callback:
                    self.progress_callback(75, f"对最优解应用局部搜索进行精细优化...")
                
                optimized_solution = self._apply_advanced_local_search(best_solution, n, y)
                
                if optimized_solution and len(optimized_solution) <= len(best_solution):
                    best_solution = optimized_solution
                    best_solution_size = len(best_solution)
                    
                    if self.progress_callback:
                        self.progress_callback(85, f"局部搜索改进后组合数量: {best_solution_size}")
            
            # 如果优化方法没有找到解但原始方法有解，则使用原始方法的近似
            if (best_solution is None or len(best_solution) == 0) and base_count:
                if self.progress_callback:
                    self.progress_callback(90, f"使用近似方法构造解，组合数量: {base_count}")
                
                best_solution = self._construct_approximate_solution(n, base_count)
                
            # 转换结果为样本ID格式
            if best_solution:
                result = []
                for sol in best_solution:
                    group = [idx_to_sample[i] for i in sol]
                    result.append(sorted(group))
                    
                if self.progress_callback:
                    self.progress_callback(100, f"贪心算法优化完成，共找到{len(result)}个组合")
                
                return result
            else:
                if self.progress_callback:
                    self.progress_callback(100, "贪心算法未找到可行解")
                return []
            
        except Exception as e:
            if self.progress_callback:
                self.progress_callback(100, f"贪心算法执行出错: {str(e)}")
            return []
    
    def _initialize_cache(self, n):
        """初始化缓存数据结构，优化集合运算"""
        self._cache = {}
        
        # 预计算j子集与k组合之间的关系
        if self.f == 1:
            self._cache['j_to_k_coverage'] = {}
            # 这里可以预计算j子集与k组合之间的关系
            # 为了避免内存占用过大，这部分实际使用时再计算
    
    def _run_standard_greedy(self, n, y):
        """运行标准贪心算法"""
        if y == 'all':
            return self._run_a2_greedy(n)
        else:
            return self._run_b2_greedy(n, y)
    
    def _run_a2_greedy(self, n):
        """运行a2贪心算法的变种，使用优化的集合操作"""
        if self.s > self.k:
            return []
        
        # 生成所有大小为k的组合
        B = [tuple(b) for b in combinations(range(n), self.k)]
        
        # 计算每个组合的s大小子集 - 使用缓存优化
        if 'C_cache' in self._cache:
            C = self._cache['C_cache']
        else:
            C = [cov(b, self.s) for b in B]
            self._cache['C_cache'] = C
        
        # 所有可能的s大小子集
        all_subs = set(combinations(range(n), self.s))
        
        # 使用快速集合操作的贪心选择
        sol_indices = []
        remaining_subs = all_subs.copy()  # 复制一份，避免修改原始集合
        
        # 创建映射：子集 -> 包含该子集的组合列表，加速查找
        subset_to_combinations = defaultdict(list)
        for i, c_set in enumerate(C):
            for subset in c_set:
                subset_to_combinations[subset].append(i)
        
        while remaining_subs:
            # 找到能覆盖最多未覆盖子集的组合
            max_coverage = 0
            best_idx = -1
            
            # 优化：只考虑那些至少包含一个未覆盖子集的组合
            candidate_indices = set()
            for subset in list(remaining_subs)[:100]:  # 限制检查的子集数量
                candidate_indices.update(subset_to_combinations[subset])
            
            for i in candidate_indices:
                coverage = len(C[i] & remaining_subs)
                if coverage > max_coverage:
                    max_coverage = coverage
                    best_idx = i
            
            # 如果没有找到能增加覆盖的组合，退出
            if max_coverage == 0:
                break
                
            sol_indices.append(best_idx)
            remaining_subs -= C[best_idx]
        
        # 检查是否找到解决方案
        if remaining_subs:
            return []
        
        # 返回选择的组合
        return [set(B[idx]) for idx in sol_indices]
    
    def _run_b2_greedy(self, n, y):
        """运行b2贪心算法的变种，使用优化的集合操作"""
        if self.j < self.s or y < 1:
            return []
        
        # 生成所有大小为k的组合
        B = [set(b) for b in combinations(range(n), self.k)]
        # 生成所有大小为j的组合
        J = [set(t) for t in combinations(range(n), self.j)]
        
        # 预计算B和J之间的关系，优化计算
        B_J_intersect = {}
        for i, b_set in enumerate(B):
            B_J_intersect[i] = {}
            for j, j_set in enumerate(J):
                B_J_intersect[i][j] = len(b_set & j_set) >= self.s
        
        # 贪心选择过程
        sol_indices = []
        
        while True:
            # 找到未满足y次覆盖要求的组合的索引
            uns_indices = []
            for j_idx, j_set in enumerate(J):
                cover_count = sum(1 for b_idx in sol_indices if B_J_intersect[b_idx][j_idx])
                if cover_count < y:
                    uns_indices.append(j_idx)
            
            # 如果所有组合都满足覆盖要求，退出
            if not uns_indices:
                break
            
            # 找到能覆盖最多未满足组合的候选组合
            max_coverage = 0
            best_idx = -1
            
            for i in range(len(B)):
                if i in sol_indices:
                    continue
                
                coverage = sum(1 for j_idx in uns_indices if B_J_intersect[i][j_idx])
                if coverage > max_coverage:
                    max_coverage = coverage
                    best_idx = i
            
            # 如果没有找到能增加覆盖的组合，退出
            if max_coverage == 0:
                return []
                
            sol_indices.append(best_idx)
            
            if len(sol_indices) > n:
                return []
        
        # 返回选择的组合
        return [B[idx] for idx in sol_indices]
    
    def _run_randomized_greedy(self, n, y, seed):
        """运行带随机性的贪心算法"""
        random.seed(seed)  # 使用不同种子保证多样性
        
        if y == 'all':
            if self.s > self.k:
                return []
            
            # 生成所有大小为k的组合
            B = [tuple(b) for b in combinations(range(n), self.k)]
            # 计算每个组合的s大小子集
            C = [cov(b, self.s) for b in B]
            # 所有可能的s大小子集
            subs = set(combinations(range(n), self.s))
            
            # 贪心选择过程，但引入随机性
            sol_indices = []
            while subs:
                # 计算每个组合能覆盖的未覆盖子集数量
                scores = [len(C[i] & subs) for i in range(len(B))]
                if max(scores) == 0:
                    break
                
                # 选择前20%的候选组合
                candidates = sorted(range(len(B)), key=lambda i: scores[i], reverse=True)
                top_k = max(1, int(len(candidates) * 0.2))
                
                # 从前20%中随机选择一个
                idx = random.choice(candidates[:top_k])
                sol_indices.append(idx)
                subs -= C[idx]
            
            # 检查是否找到解决方案
            if subs:
                return []
            
            return [set(B[idx]) for idx in sol_indices]
        else:
            if self.j < self.s or y < 1:
                return []
            
            # 生成所有大小为k的组合
            B = [set(b) for b in combinations(range(n), self.k)]
            # 生成所有大小为j的组合
            J = [set(t) for t in combinations(range(n), self.j)]
            
            # 贪心选择过程，但引入随机性
            sol_indices = []
            while True:
                # 找到未满足y次覆盖要求的组合
                uns = [Jset for Jset in J if sum(1 for b_idx in sol_indices if len(B[b_idx] & Jset) >= self.s) < y]
                if not uns:
                    break
                
                # 计算每个组合能覆盖的未满足组合数量
                scores = [sum(1 for Jset in uns if len(B[i] & Jset) >= self.s) for i in range(len(B))]
                
                # 选择前20%的候选组合
                candidates = sorted(range(len(B)), key=lambda i: scores[i], reverse=True)
                top_k = max(1, int(len(candidates) * 0.2))
                
                # 从前20%中随机选择一个
                idx = random.choice(candidates[:top_k])
                sol_indices.append(idx)
                
                if len(sol_indices) > n:
                    return []
            
            return [B[idx] for idx in sol_indices]
    
    def _run_tabu_search(self, n, y):
        """使用禁忌搜索优化"""
        # 首先使用标准贪心获得初始解
        initial_solution = self._run_standard_greedy(n, y)
        if not initial_solution:
            return []
        
        # 生成所有大小为k的组合
        all_combinations = [set(b) for b in combinations(range(n), self.k)]
        if len(all_combinations) > 1000:  # 如果组合太多，进行采样
            all_combinations = random.sample(all_combinations, 1000)
        
        # 禁忌搜索参数
        tabu_list = {}  # 禁忌列表，记录已经访问过的移动
        best_solution = initial_solution.copy()
        best_size = len(best_solution)
        current_solution = initial_solution.copy()
        
        # 禁忌搜索迭代
        max_iterations = min(50, n * 2)  # 限制迭代次数
        no_improve_count = 0
        
        for iteration in range(max_iterations):
            # 如果连续多次没有改进，增加多样性
            if no_improve_count > 5:
                # 引入扰动
                if len(current_solution) > 2:
                    # 随机移除一个组合
                    random_idx = random.randint(0, len(current_solution) - 1)
                    current_solution.pop(random_idx)
                    
                    # 随机添加一个新组合
                    new_comb = random.choice(all_combinations)
                    while new_comb in current_solution:
                        new_comb = random.choice(all_combinations)
                    current_solution.append(new_comb)
                
                no_improve_count = 0
            
            # 生成邻域：移除一个组合，添加一个新组合
            best_neighbor = None
            best_neighbor_size = float('inf')
            
            # 尝试不同的移除和添加组合
            for i in range(len(current_solution)):
                removed = current_solution[i]
                
                # 检查是否在禁忌列表中
                if (i, "remove") in tabu_list and iteration < tabu_list[(i, "remove")]:
                    continue
                
                # 尝试移除
                temp_solution = current_solution.copy()
                temp_solution.pop(i)
                
                # 如果移除后仍然有效，考虑该解
                if self._is_valid_solution(temp_solution, n, y):
                    if len(temp_solution) < best_neighbor_size:
                        best_neighbor = temp_solution
                        best_neighbor_size = len(temp_solution)
                    continue
                
                # 如果仅移除不可行，尝试添加不同的组合
                for comb in all_combinations:
                    if comb in temp_solution:
                        continue
                    
                    # 检查是否在禁忌列表中
                    if (tuple(sorted(comb)), "add") in tabu_list and iteration < tabu_list[(tuple(sorted(comb)), "add")]:
                        continue
                    
                    # 尝试添加
                    temp_solution_with_add = temp_solution + [comb]
                    
                    # 如果添加后有效且比当前最佳邻居更好
                    if self._is_valid_solution(temp_solution_with_add, n, y) and len(temp_solution_with_add) < best_neighbor_size:
                        best_neighbor = temp_solution_with_add
                        best_neighbor_size = len(temp_solution_with_add)
            
            # 如果找不到更好的邻居，尝试随机扰动
            if best_neighbor is None:
                no_improve_count += 1
                continue
            
            # 更新当前解
            current_solution = best_neighbor
            
            # 更新最佳解
            if best_neighbor_size < best_size:
                best_solution = best_neighbor
                best_size = best_neighbor_size
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            # 更新禁忌列表
            for i, sol in enumerate(current_solution):
                tabu_list[(i, "remove")] = iteration + self.tabu_tenure
                tabu_list[(tuple(sorted(sol)), "add")] = iteration + self.tabu_tenure
        
        return best_solution
    
    def _run_constrained_greedy(self, n, y, target_size):
        """运行带目标大小约束的贪心算法"""
        # 限制最大选择数量为target_size
        if y == 'all':
            if self.s > self.k:
                return []
            
            # 生成所有大小为k的组合
            B = [tuple(b) for b in combinations(range(n), self.k)]
            # 计算每个组合的s大小子集
            C = [cov(b, self.s) for b in B]
            # 所有可能的s大小子集
            subs = set(combinations(range(n), self.s))
            
            # 有约束的贪心选择过程
            sol_indices = []
            count = 0
            while subs and count < target_size:
                # 找到覆盖最多未覆盖子集的组合
                idx = max(range(len(B)), key=lambda i: len(C[i] & subs))
                if not C[idx] & subs:
                    break
                sol_indices.append(idx)
                subs -= C[idx]
                count += 1
            
            # 检查是否找到解决方案
            if subs:
                return []
            
            return [set(B[idx]) for idx in sol_indices]
        else:
            if self.j < self.s or y < 1:
                return []
            
            # 生成所有大小为k的组合
            B = [set(b) for b in combinations(range(n), self.k)]
            # 生成所有大小为j的组合
            J = [set(t) for t in combinations(range(n), self.j)]
            
            # 有约束的贪心选择过程
            sol_indices = []
            count = 0
            while count < target_size:
                # 找到未满足y次覆盖要求的组合
                uns = [Jset for Jset in J if sum(1 for b_idx in sol_indices if len(B[b_idx] & Jset) >= self.s) < y]
                if not uns:
                    break
                
                # 找到能覆盖最多未满足组合的候选组合
                idx = max(range(len(B)), key=lambda i: sum(1 for Jset in uns if len(B[i] & Jset) >= self.s))
                sol_indices.append(idx)
                count += 1
                
                if count > n:
                    return []
            
            return [B[idx] for idx in sol_indices]
    
    def _apply_local_search(self, current_solution, n, y):
        """应用局部搜索优化当前解"""
        if not current_solution:
            return current_solution
        
        # 复制当前解
        best_solution = current_solution.copy()
        best_size = len(best_solution)
        
        # 生成所有大小为k的组合
        all_combinations = list(combinations(range(n), self.k))
        
        # 转换解决方案为集合形式
        solution_sets = [set(sol) for sol in best_solution]
        
        # 局部搜索步骤
        for _ in range(self.local_search_steps):
            # 尝试移除一个组合
            for i in range(len(solution_sets)):
                # 临时移除一个组合
                temp_solution = solution_sets.copy()
                removed = temp_solution.pop(i)
                
                # 检查移除后的解是否仍然可行
                if self._is_valid_solution(temp_solution, n, y):
                    # 如果可行，更新最佳解
                    best_solution = temp_solution
                    best_size = len(best_solution)
                    solution_sets = temp_solution
                    break
                
                # 如果不可行，尝试用另一个组合替换
                for new_comb in all_combinations:
                    if set(new_comb) not in temp_solution:
                        # 添加新组合
                        temp_solution.append(set(new_comb))
                        
                        # 检查新解是否可行且更小
                        if self._is_valid_solution(temp_solution, n, y) and len(temp_solution) < best_size:
                            best_solution = temp_solution
                            best_size = len(best_solution)
                            solution_sets = temp_solution
                            break
                        
                        # 移除尝试的新组合
                        temp_solution.pop()
        
        return best_solution
    
    def _apply_advanced_local_search(self, current_solution, n, y):
        """应用高级局部搜索优化当前解，包括变邻域搜索"""
        if not current_solution:
            return current_solution
        
        # 复制当前解
        best_solution = current_solution.copy()
        best_size = len(best_solution)
        
        # 生成所有大小为k的组合
        all_combinations = [set(comb) for comb in combinations(range(n), self.k)]
        if len(all_combinations) > 1000:  # 如果组合太多，进行采样
            all_combinations = random.sample(all_combinations, 1000)
        
        # 几种不同的邻域操作
        neighborhood_ops = [
            self._neighborhood_remove_one,  # 尝试移除一个组合
            self._neighborhood_replace_one,  # 尝试替换一个组合
            self._neighborhood_remove_add_two  # 尝试移除一个，添加两个
        ]
        
        # 变邻域搜索
        improved = True
        max_iterations = 10  # 最大迭代次数
        
        for iteration in range(max_iterations):
            if not improved:
                break
                
            improved = False
            
            # 尝试每一种邻域操作
            for op in neighborhood_ops:
                new_solution = op(best_solution, all_combinations, n, y)
                
                if new_solution and len(new_solution) < best_size:
                    best_solution = new_solution
                    best_size = len(new_solution)
                    improved = True
                    break  # 如果找到更好的解，重新从第一个邻域开始
        
        return best_solution
    
    def _neighborhood_remove_one(self, solution, all_combinations, n, y):
        """邻域操作：尝试移除一个组合"""
        # 复制当前解
        best_solution = None
        
        # 尝试移除解中的每一个组合
        for i in range(len(solution)):
            temp_solution = solution.copy()
            temp_solution.pop(i)
            
            # 检查移除后的解是否仍然可行
            if self._is_valid_solution(temp_solution, n, y):
                # 找到可行解，返回
                return temp_solution
        
        return None
    
    def _neighborhood_replace_one(self, solution, all_combinations, n, y):
        """邻域操作：尝试替换一个组合"""
        # 复制当前解
        best_solution = None
        best_size = len(solution)
        
        # 尝试替换解中的每一个组合
        for i in range(len(solution)):
            for new_comb in all_combinations:
                if new_comb in solution:
                    continue
                
                # 替换
                temp_solution = solution.copy()
                temp_solution[i] = new_comb
                
                # 检查替换后的解是否可行
                if self._is_valid_solution(temp_solution, n, y) and len(temp_solution) <= best_size:
                    best_solution = temp_solution
                    best_size = len(temp_solution)
                    return best_solution
        
        return None
    
    def _neighborhood_remove_add_two(self, solution, all_combinations, n, y):
        """邻域操作：移除一个组合，添加最多两个新组合，但总数不增加"""
        if len(solution) <= 1:
            return None
            
        # 复制当前解
        best_solution = None
        best_size = len(solution)
        
        # 尝试移除解中的每一个组合
        for i in range(len(solution)):
            temp_solution = solution.copy()
            removed = temp_solution.pop(i)
            
            # 如果移除后就可行，直接返回
            if self._is_valid_solution(temp_solution, n, y):
                return temp_solution
            
            # 尝试添加一个组合
            for new_comb in all_combinations:
                if new_comb in temp_solution:
                    continue
                
                temp_solution_with_one = temp_solution.copy()
                temp_solution_with_one.append(new_comb)
                
                # 检查添加一个后是否可行
                if self._is_valid_solution(temp_solution_with_one, n, y) and len(temp_solution_with_one) < best_size:
                    best_solution = temp_solution_with_one
                    best_size = len(temp_solution_with_one)
                    return best_solution
        
        return None
    
    def _is_valid_solution(self, solution, n, y):
        """检查解决方案是否有效"""
        if len(solution) == 0:
            return False
            
        solution_key = tuple(sorted([tuple(sorted(s)) for s in solution]))
        if solution_key in self._cache.get('validity_cache', {}):
            return self._cache['validity_cache'][solution_key]
            
        result = False
        if y == 'all':
            # 检查所有s大小子集是否被覆盖
            all_subs = set(combinations(range(n), self.s))
            covered_subs = set()
            
            for sol in solution:
                sol_tuple = tuple(sorted(sol))
                covered_subs.update(cov(sol_tuple, self.s))
            
            result = covered_subs == all_subs
        else:
            # 检查每个j大小子集是否至少有y个k大小组合与其交集大小至少为s
            for j_subset in combinations(range(n), self.j):
                j_set = set(j_subset)
                cover_count = sum(1 for sol in solution if len(j_set & sol) >= self.s)
                if cover_count < y:
                    result = False
                    break
            else:
                result = True
        
        # 缓存结果
        if 'validity_cache' not in self._cache:
            self._cache['validity_cache'] = {}
        self._cache['validity_cache'][solution_key] = result
        
        return result
    
    def _construct_approximate_solution(self, n, base_count):
        """构造近似解决方案"""
        # 生成所有大小为k的组合
        all_combinations = list(combinations(range(n), self.k))
        
        # 如果组合数量太多，随机选择一部分
        if len(all_combinations) > 1000:
            all_combinations = random.sample(all_combinations, 1000)
        
        # 尝试不同的随机种子
        best_solution = None
        min_size = float('inf')
        
        for seed in range(5):  # 尝试5个不同的随机种子
            random.seed(seed)
            # 随机选择base_count个组合
            solution = [set(random.choice(all_combinations)) for _ in range(base_count)]
            
            # 应用局部搜索优化
            solution = self._apply_local_search(solution, n, 'all' if self.f > 1 else 1)
            
            # 更新最佳解
            if solution and len(solution) < min_size:
                best_solution = solution
                min_size = len(solution)
        
        return best_solution
        
    def evaluate_solution_quality(self, solution, n, y):
        """评估解决方案质量
        返回值越小表示质量越高
        """
        if not solution:
            return float('inf')
            
        # 基础质量 = 组合数量
        quality = len(solution)
        
        # 冗余度评估
        if y == 'all':
            # 计算s大小子集的覆盖次数分布
            all_subs = list(combinations(range(n), self.s))
            coverage_counts = [0] * len(all_subs)
            
            for sol in solution:
                sol_tuple = tuple(sorted(sol))
                sol_cov = cov(sol_tuple, self.s)
                for i, sub in enumerate(all_subs):
                    if sub in sol_cov:
                        coverage_counts[i] += 1
            
            # 计算覆盖均匀性，标准差越小越好
            if coverage_counts:
                mean_coverage = sum(coverage_counts) / len(coverage_counts)
                std_dev = (sum((c - mean_coverage) ** 2 for c in coverage_counts) / len(coverage_counts)) ** 0.5
                
                # 添加均匀性惩罚
                quality += std_dev * 0.1
        else:
            # 计算j大小子集的覆盖次数分布
            j_subsets = list(combinations(range(n), self.j))
            coverage_counts = []
            
            for j_subset in j_subsets:
                j_set = set(j_subset)
                cover_count = sum(1 for sol in solution if len(j_set & sol) >= self.s)
                if cover_count >= y:  # 只考虑满足要求的
                    coverage_counts.append(cover_count - y)  # 超出y的覆盖次数
            
            # 计算过度覆盖的程度
            if coverage_counts:
                excess_coverage = sum(coverage_counts) / len(coverage_counts)
                
                # 添加过度覆盖惩罚
                quality += excess_coverage * 0.1
        
        return quality 