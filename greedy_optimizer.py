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
        
    # 位掩码算法辅助函数
    def _generate_bitmasks(self, size, choice):
        """
        生成位掩码版本的组合矩阵
        每个组合表示为一个整数，其二进制表示中的1表示元素被选中
        """
        position = list(combinations(range(size), choice))
        bitmasks = []
        for pos in position:
            mask = 0
            for p in pos:
                mask |= (1 << p)  # 将p位置的位设为1
            bitmasks.append(mask)
        return bitmasks

    def _count_bits(self, n):
        """计算整数的二进制表示中1的个数"""
        count = 0
        while n:
            n &= (n - 1)  # 清除最低位的1
            count += 1
        return count

    def _count_bigger_than_s_bitmask(self, k_masks, j_masks, s):
        """
        计算每个k_mask与所有j_mask的相似度大于等于s的数量
        相似度通过位与操作和计算共同的位数实现
        """
        counts = []
        for k in k_masks:
            count = 0
            for j in j_masks:
                # 计算两个掩码之间共同的1的个数（即交集大小）
                intersection = k & j
                if self._count_bits(intersection) >= s:
                    count += 1
            counts.append(count)
        return counts

    def _find_max_index(self, lst):
        """找出列表中最大值的索引"""
        return lst.index(max(lst))

    def _extract_mask(self, masks, index):
        """提取指定索引的掩码并从列表中删除"""
        mask = masks[index]
        del masks[index]
        return mask, masks

    def _filter_j_masks(self, k_mask, j_masks, s):
        """
        过滤出与k_mask相似度大于等于s的j_masks
        返回剩余的j_masks和被移除的数量
        """
        remaining = []
        removed = []
        for j in j_masks:
            intersection = k_mask & j
            if self._count_bits(intersection) >= s:
                removed.append(j)
            else:
                remaining.append(j)
        return remaining, len(removed)

    def _when_have_ls_bitmask(self, select_j_masks, selected_k_masks, s, ls, size):
        """
        使用位掩码实现when_have_ls函数
        """
        fail_j = []
        for j_mask in select_j_masks:
            # 获取j_mask中设置为1的位置
            j_positions = [i for i in range(size) if (j_mask & (1 << i))]

            # 生成所有s大小的子集的位掩码
            s_masks = []
            for s_pos in combinations(j_positions, s):
                s_mask = 0
                for p in s_pos:
                    s_mask |= (1 << p)
                s_masks.append(s_mask)

            # 检查是否满足ls条件
            match_count = 0
            for k_mask in selected_k_masks:
                for s_mask in s_masks:
                    intersection = k_mask & s_mask
                    if self._count_bits(intersection) == s:
                        match_count += 1
                        break  # 找到一个满足条件的s_mask就可以了

            if match_count < ls:
                fail_j.append(j_mask)

        return fail_j

    def _mask_to_indices(self, mask, size):
        """将位掩码转换为索引列表"""
        return [i for i in range(size) if (mask & (1 << i))]

    def _greedy_search_bitmask(self, n, y):
        """使用位掩码的贪心搜索算法"""
        start_time = time.time()

        # 生成k和j的位掩码
        k_masks = self._generate_bitmasks(n, self.k)
        j_masks = self._generate_bitmasks(n, self.j)

        # 预计算k和j掩码之间的关系
        if self.progress_callback:
            self.progress_callback(30, f"预计算匹配关系...")
        
        # 预计算k_mask与j_mask的匹配关系
        match_matrix = []
        for k_mask in k_masks:
            matches = []
            for j_idx, j_mask in enumerate(j_masks):
                intersection = k_mask & j_mask
                if self._count_bits(intersection) >= self.s:
                    matches.append(j_idx)
            match_matrix.append(matches)

        # 开始贪心选择过程
        selected_k_masks = []
        selected_k_indices = []
        remaining_j_indices = set(range(len(j_masks)))
        total_j = len(j_masks)
        processed = 0
        
        if self.progress_callback:
            self.progress_callback(35, f"开始位掩码贪心搜索，处理 {total_j} 个组合...")

        while remaining_j_indices:
            # 计算每个k_mask能覆盖的剩余j_mask数量
            coverage_counts = []
            for k_idx, matches in enumerate(match_matrix):
                if k_idx in selected_k_indices:
                    coverage_counts.append(0)  # 已选中的k不再考虑
                else:
                    count = sum(1 for j_idx in matches if j_idx in remaining_j_indices)
                    coverage_counts.append(count)
            
            if not coverage_counts or max(coverage_counts) == 0:
                break

            # 找到覆盖最多剩余j_mask的k_mask
            best_index = self._find_max_index(coverage_counts)
            
            # 添加到已选择集合
            selected_k_indices.append(best_index)
            selected_k_masks.append(k_masks[best_index])
            
            # 记录被覆盖的j索引
            covered_j_indices = [j_idx for j_idx in match_matrix[best_index] if j_idx in remaining_j_indices]
            removed_count = len(covered_j_indices)
            
            # 从剩余集合中移除被覆盖的j
            for j_idx in covered_j_indices:
                remaining_j_indices.remove(j_idx)
            
            # 如果有ls条件，执行额外的处理
            if y != 'all' and int(y) > 1 and removed_count > 0:
                # 对被选中的j_masks进行LS条件检查
                fail_j_indices = self._check_ls_condition(
                    covered_j_indices, 
                    selected_k_masks, 
                    j_masks, 
                    self.s, 
                    int(y), 
                    n
                )
                
                # 将不满足条件的j添加回剩余集合
                remaining_j_indices.update(fail_j_indices)
            
            processed += removed_count
            
            # 更新进度
            if self.progress_callback and total_j > 0:
                progress = 35 + int(50 * processed / total_j)
                self.progress_callback(min(85, progress), f"已处理 {processed}/{total_j} 个组合...")

        end_time = time.time()
        if self.progress_callback:
            self.progress_callback(90, f"位掩码贪心搜索完成，用时: {end_time - start_time:.2f}秒")

        # 将掩码结果转换为集合形式
        result = [set(self._mask_to_indices(mask, n)) for mask in selected_k_masks]
        return result
    
    def _check_ls_condition(self, j_indices, selected_k_masks, j_masks, s, ls, size):
        """检查被选中的j是否满足ls条件"""
        fail_j_indices = []
        
        for j_idx in j_indices:
            j_mask = j_masks[j_idx]
            # 获取j_mask中设置为1的位置
            j_positions = [i for i in range(size) if (j_mask & (1 << i))]

            # 生成所有s大小的子集的位掩码
            s_masks = []
            for s_pos in combinations(j_positions, s):
                s_mask = 0
                for p in s_pos:
                    s_mask |= (1 << p)
                s_masks.append(s_mask)

            # 检查是否满足ls条件
            match_count = 0
            for k_mask in selected_k_masks:
                for s_mask in s_masks:
                    intersection = k_mask & s_mask
                    if self._count_bits(intersection) == s:
                        match_count += 1
                        break  # 找到一个满足条件的s_mask就可以了

            if match_count < ls:
                fail_j_indices.append(j_idx)

        return fail_j_indices
        
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
        
        try:
            # 使用位掩码算法获取解
            if self.progress_callback:
                self.progress_callback(20, "使用位掩码贪心算法寻找最优解...")
            
            solution = self._greedy_search_bitmask(n, y)
            
            if solution:
                if self.progress_callback:
                    self.progress_callback(90, f"位掩码贪心算法找到解决方案，组合数量: {len(solution)}")
                
                # 转换结果为样本ID格式
                result = []
                for sol in solution:
                    group = [idx_to_sample[i] for i in sol]
                    result.append(sorted(group))
                    
                if self.progress_callback:
                    self.progress_callback(100, f"贪心算法优化完成，共找到{len(result)}个组合")
                
                return result
            else:
                if self.progress_callback:
                    self.progress_callback(95, "位掩码贪心算法未找到可行解，尝试构造近似解...")
                
                # 构造一个基本解决方案
                approx_solution = self._construct_approximate_solution(n, n//2)
                
                if approx_solution:
                    # 转换结果为样本ID格式
                    result = []
                    for sol in approx_solution:
                        group = [idx_to_sample[i] for i in sol]
                        result.append(sorted(group))
                        
                    if self.progress_callback:
                        self.progress_callback(100, f"构造了近似解决方案，共{len(result)}个组合")
                    
                    return result
                else:
                    if self.progress_callback:
                        self.progress_callback(100, "贪心算法未找到可行解")
                    return []
            
        except Exception as e:
            if self.progress_callback:
                self.progress_callback(100, f"贪心算法执行出错: {str(e)}")
            return []
    
    def _construct_approximate_solution(self, n, base_count):
        """构造近似解决方案"""
        # 使用位掩码版本生成组合
        all_masks = self._generate_bitmasks(n, self.k)
        
        if len(all_masks) > 1000:
            all_masks = random.sample(all_masks, 1000)
        
        # 随机选择base_count个组合
        solution = []
        for _ in range(min(base_count, len(all_masks))):
            idx = random.randint(0, len(all_masks) - 1)
            solution.append(set(self._mask_to_indices(all_masks[idx], n)))
            all_masks.pop(idx)
        
        return solution
        
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