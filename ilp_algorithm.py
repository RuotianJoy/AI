import itertools
import random
import time
import math
import os
from collections import defaultdict
from typing import List, Tuple, Set, Dict, Optional
try:
    import pulp
except ImportError:
    print("PuLP库未安装，请运行: pip install pulp")
    raise

class ILPOptimizer:
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

        # 计算预期的k大小组合数量
        self.expected_k_combinations = math.comb(len(self.samples), self.k)
        
        # 如果问题规模合适，直接生成所有k组合；否则先不生成
        self.k_combinations = self._generate_feasible_k_combinations() if self.expected_k_combinations <= 50000 else []
        
        # 预计算j子集的覆盖关系（如果问题规模允许）
        self.coverage_matrix = {}
        if self.expected_k_combinations <= 50000:
            print(f"Precomputing coverage matrix...")
            self.coverage_matrix = self._construct_coverage_matrix()
    
    def set_progress_callback(self, callback):
        """Set progress callback function"""
        self.progress_callback = callback
    
    def _update_progress(self, progress_percent, message=None):
        """Update progress through callback if available"""
        if self.progress_callback:
            return self.progress_callback(progress_percent, message)
        return True  # Continue by default if no callback

    def _generate_feasible_k_combinations(self) -> List[Tuple[str, ...]]:
        """生成所有可行的k大小组合，根据问题规模可能采样"""
        total_combinations = math.comb(len(self.samples), self.k)
        
        # 如果k组合太多，使用随机采样
        if total_combinations > 10000:
            print(f"Total k-combinations ({total_combinations}) exceed limit, sampling...")
            k_combinations = []
            target_size = min(10000, total_combinations)
            
            # 使用集合去重
            combinations_set = set()
            while len(combinations_set) < target_size:
                new_comb = tuple(sorted(random.sample(self.samples, self.k)))
                combinations_set.add(new_comb)
            
            k_combinations = list(combinations_set)
        else:
            # 生成所有k大小组合
            k_combinations = list(itertools.combinations(self.samples, self.k))
        
        print(f"Generated {len(k_combinations)} k-combinations")
        return k_combinations

    def _construct_coverage_matrix(self) -> Dict[int, List[int]]:
        """构建覆盖矩阵 - 哪些k组合覆盖了哪些j子集
        
        返回字典：{k_组合索引: [覆盖的j子集索引]}
        """
        # 初始化进度
        self._update_progress(5, "Constructing coverage matrix...")
        
        # 创建覆盖矩阵
        coverage_matrix = {}
        
        # 如果没有预先生成k组合，先生成
        if not self.k_combinations:
            self.k_combinations = self._generate_feasible_k_combinations()
        
        # 计算每个k组合覆盖了哪些j子集
        for i, k_comb in enumerate(self.k_combinations):
            if i % 100 == 0:
                progress = 5 + (i / len(self.k_combinations) * 15)
                self._update_progress(progress, f"Processing combination {i}/{len(self.k_combinations)}")
            
            k_set = set(k_comb)
            covered_j_subsets = []
            
            # 检查这个k组合能覆盖哪些j子集
            for j_idx, j_sub in enumerate(self.j_subsets):
                j_set = set(j_sub)
                if len(j_set.intersection(k_set)) >= self.s:
                    covered_j_subsets.append(j_idx)
            
            # 只保留有覆盖能力的组合
            if covered_j_subsets:
                coverage_matrix[i] = covered_j_subsets
        
        self._update_progress(20, f"Coverage matrix constructed with {len(coverage_matrix)} combinations")
        return coverage_matrix

    def _find_initial_solution(self) -> List[int]:
        """使用贪心算法找到一个初始解，用于加速ILP求解
        
        返回选择的k组合的索引列表
        """
        # 如果需要重新构建覆盖矩阵
        if not self.coverage_matrix:
            self.coverage_matrix = self._construct_coverage_matrix()
        
        # 构建反向索引：每个j子集被哪些k组合覆盖
        j_covered_by = defaultdict(list)
        for k_idx, covered_j_idxs in self.coverage_matrix.items():
            for j_idx in covered_j_idxs:
                j_covered_by[j_idx].append(k_idx)
        
        # 开始贪心选择
        solution = []
        remaining_coverage = {j_idx: self.f for j_idx in range(len(self.j_subsets))}
        
        # 当还有未充分覆盖的j子集时，继续选择
        while any(remaining > 0 for remaining in remaining_coverage.values()):
            # 找出能够覆盖最多未充分覆盖j子集的k组合
            best_coverage = 0
            best_k_idx = None
            
            for k_idx, covered_j_idxs in self.coverage_matrix.items():
                if k_idx in solution:
                    continue  # 跳过已选的组合
                
                # 计算这个组合能覆盖多少未充分覆盖的j子集
                coverage = sum(1 for j_idx in covered_j_idxs 
                               if j_idx in remaining_coverage and remaining_coverage[j_idx] > 0)
                
                if coverage > best_coverage:
                    best_coverage = coverage
                    best_k_idx = k_idx
            
            # 如果找不到能改善的组合，则跳出循环
            if best_k_idx is None or best_coverage == 0:
                break
            
            # 添加最佳组合到解中
            solution.append(best_k_idx)
            
            # 更新覆盖情况
            for j_idx in self.coverage_matrix[best_k_idx]:
                if j_idx in remaining_coverage and remaining_coverage[j_idx] > 0:
                    remaining_coverage[j_idx] -= 1
        
        return solution

    def _test_solution_coverage(self, selected_k_indices: List[int]) -> float:
        """测试选择的k组合的覆盖率
        
        返回覆盖率（0.0-1.0）
        """
        # 计算每个j子集被覆盖的次数
        coverage_count = defaultdict(int)
        
        for k_idx in selected_k_indices:
            for j_idx in self.coverage_matrix.get(k_idx, []):
                coverage_count[j_idx] += 1
        
        # 计算达到覆盖要求的j子集比例
        covered_count = sum(1 for j_idx in range(len(self.j_subsets)) 
                           if coverage_count[j_idx] >= self.f)
        
        return covered_count / len(self.j_subsets) if self.j_subsets else 1.0

    def optimize(self, time_limit=300, verbose=True, use_initial=True, solver_threads=None, solver_gap=0.001):
        """使用整数线性规划解决覆盖问题"""
        if verbose:
            print("正在构建整数线性规划模型...")
        
        # 如果需要重新构建覆盖矩阵
        if not self.coverage_matrix:
            self.coverage_matrix = self._construct_coverage_matrix()
        
        # 创建ILP问题
        self._update_progress(25, "Creating ILP model...")
        problem = pulp.LpProblem("Minimum_Covering_Combinations", pulp.LpMinimize)
        
        # 定义决策变量（每个k组合是否被选中）
        x = {}
        for i in self.coverage_matrix.keys():
            x[i] = pulp.LpVariable(f"x_{i}", cat='Binary')
        
        # 设置目标函数：最小化选中的k组合数量
        problem += pulp.lpSum(x[i] for i in self.coverage_matrix.keys())
        
        # 构建反向索引：每个j子集被哪些k组合覆盖
        j_covered_by = defaultdict(list)
        for k_idx, covered_j_idxs in self.coverage_matrix.items():
            for j_idx in covered_j_idxs:
                j_covered_by[j_idx].append(k_idx)
        
        # 添加约束：每个j子集必须被覆盖至少f次
        self._update_progress(35, "Adding constraints...")
        constraints_added = 0
        for j_idx in range(len(self.j_subsets)):
            # 找出所有能覆盖这个j子集的k组合
            covering_combinations = j_covered_by[j_idx]
            
            # 添加覆盖约束
            if covering_combinations:
                problem += pulp.lpSum(x[i] for i in covering_combinations) >= self.f, f"Cover_j_subset_{j_idx}"
                constraints_added += 1
            else:
                if verbose:
                    print(f"Warning: j-subset {j_idx} cannot be covered by any k-combination!")
        
        if verbose:
            print(f"Added {constraints_added} coverage constraints")
        
        # 使用初始解加速求解 - 通过贪心算法获取一个初始解
        initial_solution = None
        if use_initial:
            self._update_progress(40, "Finding initial solution using greedy algorithm...")
            initial_solution = self._find_initial_solution()
            
            if verbose and initial_solution:
                initial_coverage = self._test_solution_coverage(initial_solution)
                print(f"Found initial solution with {len(initial_solution)} combinations, coverage: {initial_coverage:.2%}")
                
                # 添加上界约束，可能大幅减少搜索空间
                problem += pulp.lpSum(x[i] for i in self.coverage_matrix.keys()) <= len(initial_solution), "Upper_bound"
        
        # 设置求解器参数
        solver_options = []
        if time_limit:
            solver_options.append(f"timeLimit={time_limit}")
        if solver_threads:
            solver_options.append(f"threads={solver_threads}")
        if solver_gap:
            solver_options.append(f"gapRel={solver_gap}")
        
        # 创建求解器
        solver = pulp.PULP_CBC_CMD(options=solver_options)
        
        # 设置warm start，如果有初始解
        if initial_solution:
            for k_idx in self.coverage_matrix.keys():
                x[k_idx].setInitialValue(1 if k_idx in initial_solution else 0)
        
        # 求解问题
        self._update_progress(45, "Solving ILP model...")
        start_time = time.time()
        problem.solve(solver)
        solve_time = time.time() - start_time
        
        # 检查是否找到解
        if verbose:
            print(f"Solver status: {pulp.LpStatus[problem.status]}")
            print(f"Solution time: {solve_time:.2f} seconds")
        
        # 如果没有找到最优解，但有初始解，则使用初始解作为备选
        if problem.status != pulp.LpStatusOptimal:
            if verbose:
                print(f"No optimal solution found. Status: {pulp.LpStatus[problem.status]}")
            
            # 如果有初始解并且ILP未找到更好的解，则返回初始解
            if initial_solution:
                if verbose:
                    print(f"Using initial solution with {len(initial_solution)} combinations")
                solution = [self.k_combinations[i] for i in initial_solution]
                
                # 验证覆盖率
                coverage_ratio = self.verify_coverage(solution)
                self._update_progress(100, f"Using initial solution with {len(solution)} groups and {coverage_ratio:.2%} coverage")
                return solution
            
            self._update_progress(100, "No optimal solution found")
            return []
        
        # 提取结果
        solution = []
        selected_indices = []
        for i in self.coverage_matrix.keys():
            if pulp.value(x[i]) == 1:
                solution.append(self.k_combinations[i])
                selected_indices.append(i)
        
        if verbose:
            print(f"Found optimal solution with {len(solution)} combinations.")
            
            # 如果有初始解，比较改进
            if initial_solution:
                improvement = len(initial_solution) - len(solution)
                if improvement > 0:
                    print(f"Improved by {improvement} combinations compared to initial solution")
                else:
                    print("No improvement over initial solution")
        
        # 验证解的覆盖率
        coverage_ratio = self.verify_coverage(solution)
        self._update_progress(100, f"Optimization complete. Final solution has {len(solution)} groups with {coverage_ratio:.2%} coverage")
        
        # 尝试进一步优化解
        if verbose and coverage_ratio == 1.0:
            print("Attempting to further optimize solution...")
            optimized_solution = self._remove_redundant_combinations(solution)
            if len(optimized_solution) < len(solution):
                print(f"Further reduced to {len(optimized_solution)} combinations")
                solution = optimized_solution
        
        return solution
    
    def _remove_redundant_combinations(self, solution: List[Tuple[str, ...]]) -> List[Tuple[str, ...]]:
        """尝试进一步移除冗余组合"""
        if not solution:
            return []
        
        # 复制解以免修改原始解
        optimized = list(solution)
        
        # 计算每个j子集被覆盖的次数
        coverage_count = defaultdict(int)
        
        for j_sub in self.j_subsets:
            j_set = set(j_sub)
            for group in optimized:
                group_set = set(group)
                if len(j_set.intersection(group_set)) >= self.s:
                    coverage_count[j_sub] += 1
        
        # 随机打乱顺序，尝试从不同顺序移除
        indices = list(range(len(optimized)))
        random.shuffle(indices)
        
        for idx in indices:
            if len(optimized) <= 1:
                break
                
            # 临时移除
            group = optimized.pop(idx)
            
            # 检查移除后的覆盖情况
            temp_coverage = defaultdict(int)
            for j_sub in self.j_subsets:
                j_set = set(j_sub)
                for remaining_group in optimized:
                    group_set = set(remaining_group)
                    if len(j_set.intersection(group_set)) >= self.s:
                        temp_coverage[j_sub] += 1
            
            # 如果移除导致覆盖率下降，则恢复
            if not all(temp_coverage[j_sub] >= self.f for j_sub in self.j_subsets):
                optimized.insert(idx, group)
        
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

    def solve_from_examples(self, examples=None, named_samples=None, verbose=True):
        """从示例中解决问题，针对字母标记的样本"""
        if named_samples is None:
            # 默认使用A,B,C...标记样本
            alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            named_samples = {sample: alphabet[i] if i < len(alphabet) else f"Sample{i+1}" 
                            for i, sample in enumerate(self.samples)}
        
        # 如果用户提供了示例解
        if examples:
            if verbose:
                print("Validating provided examples...")
            
            # 将字母样本名转换为实际样本ID
            converted_examples = []
            sample_to_id = {v: k for k, v in named_samples.items()}
            
            for example in examples:
                converted_group = tuple(sorted(sample_to_id[name] for name in example))
                converted_examples.append(converted_group)
            
            # 验证这个解是否满足覆盖要求
            coverage_ratio = self.verify_coverage(converted_examples)
            if verbose:
                print(f"Example solution has {len(converted_examples)} groups with {coverage_ratio:.2%} coverage")
            
            if coverage_ratio == 1.0:
                return converted_examples
        
        # 使用ILP求解
        return self.optimize(verbose=verbose)

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

def show_results(solution, named_samples=None):
    """显示结果，可选择使用命名的样本(如A,B,C...)"""
    if not solution:
        print("未找到有效解")
        return
    
    print(f"\n最优解包含 {len(solution)} 个组合:")
    for i, group in enumerate(solution, 1):
        if named_samples:
            # 转换为命名样本
            named_group = [named_samples.get(sample, sample) for sample in group]
            print(f"组合{i}: {', '.join(named_group)}")
        else:
            print(f"组合{i}: {', '.join(group)}")

def verify_coverage(solution, samples, j, s, f=1, named_samples=None):
    """验证解是否满足覆盖要求"""
    # 如果使用命名样本，需要先转换
    if named_samples:
        sample_to_id = {v: k for k, v in named_samples.items()}
        converted_solution = []
        for group in solution:
            converted_group = tuple(sorted(sample_to_id.get(sample, sample) for sample in group))
            converted_solution.append(converted_group)
        solution = converted_solution
    
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

def example_from_text(example_text, letter_samples=None):
    """从文本中提取示例组合"""
    if letter_samples is None:
        letter_samples = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    
    examples = []
    lines = example_text.strip().split('\n')
    
    for line in lines:
        if not line.strip():
            continue
            
        # 查找字母组合，如 A,B,C,D,E,G 或 A,B,C,E,F,G
        groups = []
        parts = line.split('.')
        for part in parts:
            # 提取所有字母
            letters = [c for c in part if c.upper() in letter_samples]
            if letters:
                groups.append(letters)
        
        if groups:
            examples.extend(groups)
    
    return examples

if __name__ == "__main__":
    print("基于整数线性规划的最小覆盖组合优化")
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
    
    # 设置求解器参数
    use_initial = input("\n是否使用贪心算法生成初始解加速求解？(y/n): ").lower() == 'y'
    time_limit = int(input("\n设置ILP求解时间限制（秒）(建议300-600): "))
    
    # 使用多线程
    threads = None
    if input("\n使用多线程求解？(y/n): ").lower() == 'y':
        import multiprocessing
        recommended = max(1, multiprocessing.cpu_count() - 1)
        threads = get_int(f"线程数（推荐{recommended}）: ", 1, None)
    
    optimizer = ILPOptimizer(samples, params['j'], params['s'], params['k'], params['f'])
    
    # 创建字母标记样本映射
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    named_samples = {}
    for i, sample in enumerate(samples):
        if i < len(alphabet):
            named_samples[sample] = alphabet[i]
        else:
            named_samples[sample] = f"S{i+1}"
    
    # 询问是否提供示例解
    examples = None
    if input("\n是否提供示例解作为参考？(y/n): ").lower() == 'y':
        example_text = input("请输入示例解（每行一个组合，形如 A,B,C,D,E,G 格式）:\n")
        if example_text.strip():
            examples = example_from_text(example_text, list(named_samples.values()))
    
    # 求解问题
    solution = optimizer.optimize(
        time_limit=time_limit, 
        verbose=True,
        use_initial=use_initial,
        solver_threads=threads
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
    
    show_results(solution, named_samples) 