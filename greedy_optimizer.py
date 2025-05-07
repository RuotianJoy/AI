import time
import random
import concurrent.futures
import numpy as np
from greedy_algorithm import r2, cov, a2, b2
from itertools import combinations
from collections import defaultdict

class GreedyOptimizer:
    """Greedy algorithm optimizer class, with interface consistent with existing genetic and simulated annealing optimizers"""
    
    def __init__(self, samples, j, s, k, f):
        """
        Initialize optimizer
        samples: Sample list
        j: Subset parameter
        s: Coverage parameter
        k: Combination size
        f: Coverage frequency
        """
        self.samples = samples    # Sample list
        self.j = j                # Subset parameter
        self.s = s                # Coverage parameter
        self.k = k                # Combination size
        self.f = f                # Coverage frequency
        self.progress_callback = None
        # Optimization parameters
        self.max_iterations = 5   # Maximum iteration count
        self.local_search_steps = 3  # Local search steps
        self.random_factor = 0.3  # Random factor
        self.tabu_tenure = 10     # Tabu tenure
        self.max_parallel_workers = 4  # Maximum parallel worker threads
        # Cache
        self._cache = {}
        
    def set_progress_callback(self, callback):
        """Set progress callback function"""
        self.progress_callback = callback
        
    # Bitmask algorithm helper functions
    def _generate_bitmasks(self, size, choice):
        """
        Generate bitmask version of combination matrix
        Each combination is represented as an integer, where 1s in binary representation indicate selected elements
        """
        position = list(combinations(range(size), choice))
        bitmasks = []
        for pos in position:
            mask = 0
            for p in pos:
                mask |= (1 << p)  # Set bit at position p to 1
            bitmasks.append(mask)
        return bitmasks

    def _count_bits(self, n):
        """Count number of 1s in binary representation of integer"""
        count = 0
        while n:
            n &= (n - 1)  # Clear the lowest 1
            count += 1
        return count

    def _count_bigger_than_s_bitmask(self, k_masks, j_masks, s):
        """
        Count the number of j_masks with similarity greater than or equal to s for each k_mask
        Similarity is implemented through bitwise AND operation and counting common bits
        """
        counts = []
        for k in k_masks:
            count = 0
            for j in j_masks:
                # Calculate number of common 1s between two masks (intersection size)
                intersection = k & j
                if self._count_bits(intersection) >= s:
                    count += 1
            counts.append(count)
        return counts

    def _find_max_index(self, lst):
        """Find index of maximum value in list"""
        return lst.index(max(lst))

    def _extract_mask(self, masks, index):
        """Extract mask at specified index and remove from list"""
        mask = masks[index]
        del masks[index]
        return mask, masks

    def _filter_j_masks(self, k_mask, j_masks, s):
        """
        Filter j_masks with similarity greater than or equal to s with k_mask
        Return remaining j_masks and number of removed masks
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
        Implement when_have_ls function using bitmasks
        """
        fail_j = []
        for j_mask in select_j_masks:
            # Get positions set to 1 in j_mask
            j_positions = [i for i in range(size) if (j_mask & (1 << i))]

            # Generate bitmasks for all subsets of size s
            s_masks = []
            for s_pos in combinations(j_positions, s):
                s_mask = 0
                for p in s_pos:
                    s_mask |= (1 << p)
                s_masks.append(s_mask)

            # Check if ls condition is satisfied
            match_count = 0
            for k_mask in selected_k_masks:
                for s_mask in s_masks:
                    intersection = k_mask & s_mask
                    if self._count_bits(intersection) == s:
                        match_count += 1
                        break  # Found one s_mask that satisfies the condition

            if match_count < ls:
                fail_j.append(j_mask)

        return fail_j

    def _mask_to_indices(self, mask, size):
        """Convert bitmask to index list"""
        return [i for i in range(size) if (mask & (1 << i))]

    def _greedy_search_bitmask(self, n, coverage_frequency):
        """Greedy search algorithm using bitmasks"""
        start_time = time.time()

        # Generate bitmasks for k and j
        k_masks = self._generate_bitmasks(n, self.k)
        j_masks = self._generate_bitmasks(n, self.j)

        # Precompute relationships between k and j masks
        if self.progress_callback:
            self.progress_callback(30, f"Precomputing matching relationships...")
        
        # Precompute matching relationships between k_mask and j_mask
        match_matrix = []
        for k_mask in k_masks:
            matches = []
            for j_idx, j_mask in enumerate(j_masks):
                intersection = k_mask & j_mask
                if self._count_bits(intersection) >= self.s:
                    matches.append(j_idx)
            match_matrix.append(matches)

        # Start greedy selection process
        selected_k_masks = []
        selected_k_indices = []
        
        # Track coverage count for each j-subset
        j_coverage_counts = [0] * len(j_masks)
        
        # Remaining j indices are those that haven't been covered f times
        remaining_j_indices = set(range(len(j_masks)))
        
        total_j = len(j_masks)
        processed = 0
        
        if self.progress_callback:
            self.progress_callback(35, f"Starting bitmask greedy search, processing {total_j} combinations...")

        while remaining_j_indices:
            # Calculate number of uncovered j_masks for each k_mask
            coverage_counts = []
            for k_idx, matches in enumerate(match_matrix):
                if k_idx in selected_k_indices:
                    coverage_counts.append(0)  # Already selected k are not considered
                else:
                    # Count j-subsets that would benefit from this k-subset
                    # (those that haven't been covered f times yet)
                    count = sum(1 for j_idx in matches if j_idx in remaining_j_indices and j_coverage_counts[j_idx] < coverage_frequency)
                    coverage_counts.append(count)
            
            if not coverage_counts or max(coverage_counts) == 0:
                break

            # Find k_mask that covers the most remaining j_masks
            best_index = self._find_max_index(coverage_counts)
            
            # Add to selected set
            selected_k_indices.append(best_index)
            selected_k_masks.append(k_masks[best_index])
            
            # Update coverage counts for j-subsets
            newly_covered = []
            for j_idx in match_matrix[best_index]:
                if j_idx in remaining_j_indices:
                    j_coverage_counts[j_idx] += 1
                    
                    # If this j-subset has now been covered f times, remove it from remaining
                    if j_coverage_counts[j_idx] >= coverage_frequency:
                        newly_covered.append(j_idx)
            
            # Remove fully covered j-subsets from remaining set
            for j_idx in newly_covered:
                remaining_j_indices.remove(j_idx)
                processed += 1
            
            # Update progress
            if self.progress_callback and total_j > 0:
                progress = 35 + int(50 * processed / total_j)
                self.progress_callback(min(85, progress), f"Processed {processed}/{total_j} combinations...")

        end_time = time.time()
        if self.progress_callback:
            self.progress_callback(90, f"Bitmask greedy search completed, time used: {end_time - start_time:.2f} seconds")

        # Convert mask results to set form
        result = [set(self._mask_to_indices(mask, n)) for mask in selected_k_masks]
        return result
    
    def _check_ls_condition(self, j_indices, selected_k_masks, j_masks, s, ls, size):
        """Check if selected j satisfies ls condition"""
        fail_j_indices = []
        
        for j_idx in j_indices:
            j_mask = j_masks[j_idx]
            # Get positions set to 1 in j_mask
            j_positions = [i for i in range(size) if (j_mask & (1 << i))]

            # Generate bitmasks for all subsets of size s
            s_masks = []
            for s_pos in combinations(j_positions, s):
                s_mask = 0
                for p in s_pos:
                    s_mask |= (1 << p)
                s_masks.append(s_mask)

            # Check if ls condition is satisfied
            match_count = 0
            for k_mask in selected_k_masks:
                for s_mask in s_masks:
                    intersection = k_mask & s_mask
                    if self._count_bits(intersection) == s:
                        match_count += 1
                        break  # Found one s_mask that satisfies the condition

            if match_count < ls:
                fail_j_indices.append(j_idx)

        return fail_j_indices
    
    def _verify_solution(self, solution, n, coverage_frequency):
        """
        验证解决方案是否满足覆盖要求
        返回：验证通过标志，覆盖率，详细统计数据
        """
        if not solution:
            return False, 0, {}
        
        # 生成所有j子集
        j_subsets = list(combinations(range(n), self.j))
        coverage_counts = {}
        
        # 统计每个j子集被覆盖的次数
        for i, j_subset in enumerate(j_subsets):
            j_set = set(j_subset)
            cover_count = sum(1 for sol in solution if len(j_set & sol) >= self.s)
            coverage_counts[i] = cover_count
        
        # 计算覆盖率
        properly_covered = sum(1 for count in coverage_counts.values() if count >= coverage_frequency)
        total_subsets = len(j_subsets)
        coverage_ratio = properly_covered / total_subsets if total_subsets else 0
        
        # 统计覆盖分布
        coverage_distribution = defaultdict(int)
        for count in coverage_counts.values():
            coverage_distribution[count] += 1
        
        # 计算平均覆盖次数
        avg_coverage = sum(coverage_counts.values()) / total_subsets if total_subsets else 0
        
        # 检查是否所有子集都被覆盖f次
        all_covered = all(count >= coverage_frequency for count in coverage_counts.values())
        
        stats = {
            "total_j_subsets": total_subsets,
            "properly_covered": properly_covered,
            "coverage_ratio": coverage_ratio,
            "average_coverage": avg_coverage,
            "coverage_distribution": dict(coverage_distribution),
            "under_covered": total_subsets - properly_covered,
            "solution_size": len(solution)
        }
        
        return all_covered, coverage_ratio, stats
    
    def _remove_redundant_combinations(self, solution, n, coverage_frequency):
        """
        移除冗余组合，保持覆盖要求不变
        采用保守策略，避免删除过多组合
        """
        if not solution or len(solution) <= 1:
            return solution
        
        if self.progress_callback:
            self.progress_callback(92, f"优化解决方案，移除冗余组合...")
        
        # 复制解决方案，避免修改原始数据
        solution = solution.copy()
        optimized = False
        
        # 计算每个j子集的覆盖信息
        j_subsets = list(combinations(range(n), self.j))
        
        # 建立每个k组合对j子集的覆盖映射
        coverage_map = []
        for sol in solution:
            covered_j = []
            for i, j_subset in enumerate(j_subsets):
                j_set = set(j_subset)
                if len(j_set & sol) >= self.s:
                    covered_j.append(i)
            coverage_map.append(covered_j)
        
        # 计算当前j子集的覆盖次数
        j_coverage_counts = [0] * len(j_subsets)
        for covered_j in coverage_map:
            for j_idx in covered_j:
                j_coverage_counts[j_idx] += 1
        
        # 识别关键组合 - 对某些j子集的覆盖是不可或缺的
        critical_combinations = set()
        
        # 找出覆盖次数恰好等于所需频率f的j子集
        critical_j_indices = [i for i, count in enumerate(j_coverage_counts) 
                             if count == coverage_frequency]
        
        # 对于这些临界j子集，将覆盖它们的所有组合标记为关键组合
        for j_idx in critical_j_indices:
            for k_idx, covered_j in enumerate(coverage_map):
                if j_idx in covered_j:
                    critical_combinations.add(k_idx)
        
        # 找出覆盖次数为f+1的j子集，这些也需要额外保护
        near_critical_j_indices = [i for i, count in enumerate(j_coverage_counts) 
                                  if count == coverage_frequency + 1]
        
        # 计算每个组合的"冗余分数"
        # 分数越高表示组合越冗余，越可能被删除
        redundancy_scores = []
        for k_idx, covered_j in enumerate(coverage_map):
            if k_idx in critical_combinations:
                # 关键组合有最低分数，不应被删除
                redundancy_scores.append(-1000)
                continue
                
            # 计算该组合覆盖的j子集的平均覆盖次数
            avg_coverage = 0
            critical_coverage = 0
            near_critical_coverage = 0
            
            if covered_j:
                avg_coverage = sum(j_coverage_counts[j] for j in covered_j) / len(covered_j)
                # 计算覆盖的临界j子集数量
                critical_coverage = sum(1 for j in covered_j if j in critical_j_indices)
                # 计算覆盖的接近临界j子集数量
                near_critical_coverage = sum(1 for j in covered_j if j in near_critical_j_indices)
            
            # 分数计算:
            # 1. 覆盖的j子集越多，分数越低(越不冗余)
            # 2. 覆盖的j子集平均覆盖次数越高，分数越高(越冗余)
            # 3. 覆盖临界和接近临界的j子集，大幅降低分数(保护这些组合)
            score = avg_coverage - len(covered_j) * 0.5 - critical_coverage * 10 - near_critical_coverage * 5
            redundancy_scores.append(score)
        
        # 按冗余分数排序索引(从高到低)
        sorted_indices = sorted(range(len(solution)), 
                               key=lambda i: -redundancy_scores[i])
        
        # 设置最大允许删除的组合数量(保守策略)
        max_removal = min(len(solution) // 5, 10)  # 最多删除20%或10个组合
        
        # 尝试移除组合，如果移除后仍然满足覆盖要求，则永久移除
        removed_count = 0
        for idx in sorted_indices:
            # 如果达到最大删除数量，停止删除
            if removed_count >= max_removal:
                break
                
            # 调整索引考虑已删除的组合
            adjusted_idx = idx
            for prev_idx in sorted_indices[:sorted_indices.index(idx)]:
                if prev_idx < idx and redundancy_scores[prev_idx] > 0:  # 之前已删除的组合
                    adjusted_idx -= 1
            
            if adjusted_idx >= len(solution) or adjusted_idx < 0:
                continue
                
            # 如果是关键组合，跳过
            if redundancy_scores[idx] < 0:
                continue
                
            # 检查移除此组合后是否仍满足覆盖要求
            can_remove = True
            # 创建临时覆盖计数副本
            temp_coverage_counts = j_coverage_counts.copy()
            
            # 更新临时覆盖计数
            for j_idx in coverage_map[adjusted_idx]:
                temp_coverage_counts[j_idx] -= 1
                if temp_coverage_counts[j_idx] < coverage_frequency:
                    can_remove = False
                    break
            
            # 增加额外保守条件：不要过度减少覆盖
            if can_remove:
                # 计算移除前后的平均冗余覆盖率
                before_avg = sum(max(0, count - coverage_frequency) for count in j_coverage_counts) / len(j_subsets)
                after_avg = sum(max(0, count - coverage_frequency) for count in temp_coverage_counts) / len(j_subsets)
                
                # 如果移除会导致平均冗余降低过多，则不删除
                if before_avg > 0 and (before_avg - after_avg) / before_avg > 0.3:  # 超过30%的冗余减少
                    can_remove = False
            
            if can_remove:
                # 更新实际覆盖计数
                for j_idx in coverage_map[adjusted_idx]:
                    j_coverage_counts[j_idx] -= 1
                
                # 移除组合及其覆盖映射
                del solution[adjusted_idx]
                del coverage_map[adjusted_idx]
                removed_count += 1
                optimized = True
                
                if self.progress_callback and removed_count % 2 == 0:
                    self.progress_callback(92 + min(7, removed_count//2), 
                                         f"已移除 {removed_count} 个冗余组合...")
        
        if self.progress_callback and optimized:
            self.progress_callback(99, f"优化完成，共移除 {removed_count} 个冗余组合")
        elif self.progress_callback:
            self.progress_callback(99, "无冗余组合可移除")
        
        return solution
        
    def _remove_similar_combinations(self, solution, n):

        if not solution or len(solution) <= 1:
            return solution
            
        if self.progress_callback:
            self.progress_callback(95, f"start...")
            
        # 复制解决方案，避免修改原始数据
        solution = solution.copy()
        
        # 创建要移除的组合集合
        to_remove = set()
        
        # 根据j/2的值确定处理策略
        j_half_value = self.j // 2  # 整数除法，自动向下取整
        
        # 根据j_half_value决定采用两两比较还是三元组比较
        if j_half_value <= 2:  # 小于等于2时采用两两比较
            if self.progress_callback:
                self.progress_callback(95, f"j={self.j}")
                
            # 创建所有组合对的索引
            pairs = list(combinations(range(len(solution)), 2))
            
            # 随机打乱对的顺序，以确保随机性
            random.shuffle(pairs)
            
            # 多线程处理函数
            def process_batch(batch_pairs):
                local_to_remove = set()
                for i, j in batch_pairs:
                    # 如果其中一个已经在移除列表中，跳过
                    if i in local_to_remove or j in local_to_remove:
                        continue
                        
                    # 计算两个组合的交集大小
                    intersection_size = len(solution[i] & solution[j])
                    
                    # 如果交集大小恰好为k-1，表示有k-1个元素相同
                    if intersection_size == self.k - 1:
                        # 随机选择要移除的组合
                        to_remove_idx = random.choice([i, j])
                        local_to_remove.add(to_remove_idx)
                return local_to_remove
                
            # 计算每个线程处理的组合数
            num_workers = min(self.max_parallel_workers, 8)  # 最多使用8个线程
            batch_size = max(1, len(pairs) // num_workers)
            
            # 创建批次
            batches = [pairs[i:i + batch_size] for i in range(0, len(pairs), batch_size)]
            
            # 使用线程池处理批次
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                # 启动所有任务
                future_to_batch = {executor.submit(process_batch, batch): batch for batch in batches}
                
                # 收集结果
                completed = 0
                for future in concurrent.futures.as_completed(future_to_batch):
                    batch = future_to_batch[future]
                    try:
                        batch_result = future.result()
                        to_remove.update(batch_result)
                        
                        # 更新进度
                        completed += 1
                        if self.progress_callback:
                            progress = 95 + int(3 * completed / len(batches))
                            self.progress_callback(min(98, progress), 
                                                 f"已检查 {completed}/{len(batches)} 批次，找到 {len(to_remove)} 个相似组合...")
                    except Exception as e:
                        print(f"处理批次时出错: {e}")
        
        else:  # j_half_value > 2，采用三元组比较
            if len(solution) <= 2:  # 至少需要3个组合才能形成三元组
                return solution
                
            if self.progress_callback:
                self.progress_callback(95, f"j={self.j}")
                
            # 创建所有三元组的索引
            triplets = list(combinations(range(len(solution)), 3))
            
            # 随机打乱三元组的顺序，以确保随机性
            random.shuffle(triplets)
            
            # 多线程处理函数
            def process_batch(batch_triplets):
                local_to_remove = set()
                for i, j, k in batch_triplets:
                    # 如果其中有已经在移除列表中的组合，跳过
                    if i in local_to_remove or j in local_to_remove or k in local_to_remove:
                        continue
                    
                    # 计算两两之间的交集大小
                    intersection_size_ij = len(solution[i] & solution[j])
                    intersection_size_ik = len(solution[i] & solution[k])
                    intersection_size_jk = len(solution[j] & solution[k])
                    
                    # 计算高度相似对的数量（交集大小为k-1的对数）
                    high_similarity_count = 0
                    if intersection_size_ij == self.k - 1:
                        high_similarity_count += 1
                    if intersection_size_ik == self.k - 1:
                        high_similarity_count += 1
                    if intersection_size_jk == self.k - 1:
                        high_similarity_count += 1
                    
                    # 如果三元组中有至少2对高度相似的组合，则保留一个，删除其他两个
                    if high_similarity_count >= 2:
                        # 随机选择要保留的组合索引
                        to_keep = random.choice([i, j, k])
                        # 将其他两个组合添加到移除列表
                        to_remove_indices = [idx for idx in [i, j, k] if idx != to_keep]
                        local_to_remove.update(to_remove_indices)
                return local_to_remove
                
            # 计算每个线程处理的组合数
            num_workers = min(self.max_parallel_workers, 8)  # 最多使用8个线程
            batch_size = max(1, len(triplets) // num_workers)
            
            # 创建批次
            batches = [triplets[i:i + batch_size] for i in range(0, len(triplets), batch_size)]
            
            # 使用线程池处理批次
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                # 启动所有任务
                future_to_batch = {executor.submit(process_batch, batch): batch for batch in batches}
                
                # 收集结果
                completed = 0
                for future in concurrent.futures.as_completed(future_to_batch):
                    batch = future_to_batch[future]
                    try:
                        batch_result = future.result()
                        to_remove.update(batch_result)
                        
                        # 更新进度
                        completed += 1
                        if self.progress_callback:
                            progress = 95 + int(3 * completed / len(batches))
                            self.progress_callback(min(98, progress), 
                                                 f"已检查 {completed}/{len(batches)} 批次，找到 {len(to_remove)} 个相似组合...")
                    except Exception as e:
                        print(f"处理批次时出错: {e}")
        
        # 从解决方案中移除标记的组合（从大到小的索引，避免索引改变）
        to_remove_list = sorted(to_remove, reverse=True)
        
        # 判断要删除的组合数是否大于保留的组合数
        if len(to_remove_list) > len(solution) - len(to_remove_list):
            if self.progress_callback:
                self.progress_callback(98, f"要删除的组合数({len(to_remove_list)})大于保留的组合数({len(solution) - len(to_remove_list)})，反转删除逻辑...")
            
            # 反转逻辑：保留原本要删除的组合，删除原本要保留的组合
            to_keep_indices = list(to_remove)
            new_solution = []
            for idx in to_keep_indices:
                if idx < len(solution):  # 确保索引有效
                    new_solution.append(solution[idx])
            
            if self.progress_callback:
                self.progress_callback(99, f"完成相似组合检查，通过反转逻辑保留了 {len(new_solution)} 个组合")
            
            return new_solution
        else:
            # 原有逻辑：删除标记的组合
            for idx in to_remove_list:
                if idx < len(solution):  # 确保索引仍然有效
                    del solution[idx]
                
            if self.progress_callback:
                self.progress_callback(99, f"完成相似组合检查，共移除 {len(to_remove_list)} 个相似组合")
            
            return solution
        
    def optimize(self):
        """
        Execute greedy algorithm optimization
        Return best solution (optimal combination result)
        """
        if self.progress_callback:
            self.progress_callback(5, "Preparing greedy algorithm runtime environment...")
        
        # Get sample count
        n = len(self.samples)
        
        # Build sample ID mapping (from 0 to n-1)
        sample_to_idx = {sample: i for i, sample in enumerate(sorted(self.samples))}
        idx_to_sample = {i: sample for sample, i in sample_to_idx.items()}
        
        # Use the actual coverage frequency parameter
        coverage_frequency = self.f
        
        if self.progress_callback:
            self.progress_callback(10, f"Using parameters n={n}, k={self.k}, j={self.j}, s={self.s}, f={coverage_frequency}")
        
        try:
            # Get solution using bitmask algorithm
            if self.progress_callback:
                self.progress_callback(20, "Using bitmask greedy algorithm to find optimal solution...")
            
            solution = self._greedy_search_bitmask(n, coverage_frequency)
            
            if solution:
                # 验证解决方案
                if self.progress_callback:
                    self.progress_callback(91, f"验证算法解决方案...")
                
                is_valid, coverage_ratio, stats = self._verify_solution(solution, n, coverage_frequency)
                
                # 移除冗余组合
                solution = self._remove_redundant_combinations(solution, n, coverage_frequency)
                if n >= 13 and len(solution) >= 100:
                    solution = self._remove_similar_combinations(solution, n)
                
                # 再次验证优化后的解决方案
                final_valid, final_coverage, final_stats = self._verify_solution(solution, n, coverage_frequency)
                
                if self.progress_callback:
                    if final_valid:
                        self.progress_callback(100, f"解决方案验证通过！组合数量: {len(solution)}，j子集覆盖率: {final_coverage*100:.2f}%")
                    else:
                        self.progress_callback(100, f"警告：解决方案验证未通过，j子集覆盖率: {final_coverage*100:.2f}%")
                
                # Convert result to sample ID format
                result = []
                for sol in solution:
                    group = [idx_to_sample[i] for i in sol]
                    result.append(sorted(group))
                    
                return result
            else:
                if self.progress_callback:
                    self.progress_callback(95, "Bitmask greedy algorithm didn't find feasible solution, trying to construct approximate solution...")
                
                # Construct a basic solution
                approx_solution = self._construct_approximate_solution(n, n//2)
                
                if approx_solution:
                    # 验证和优化近似解决方案
                    is_valid, coverage_ratio, stats = self._verify_solution(approx_solution, n, coverage_frequency)
                    
                    if coverage_ratio > 0.5:  # 如果覆盖率超过50%，尝试优化
                        approx_solution = self._remove_redundant_combinations(approx_solution, n, coverage_frequency)
                        
                        # 对近似解决方案也进行相似组合检查
                        approx_solution = self._remove_similar_combinations(approx_solution, n)
                    
                    # Convert result to sample ID format
                    result = []
                    for sol in approx_solution:
                        group = [idx_to_sample[i] for i in sol]
                        result.append(sorted(group))
                        
                    if self.progress_callback:
                        self.progress_callback(100, f"构建近似解决方案，组合数量: {len(result)}，覆盖率: {coverage_ratio*100:.2f}%")
                    
                    return result
                else:
                    if self.progress_callback:
                        self.progress_callback(100, "Greedy algorithm didn't find feasible solution")
                    return []
            
        except Exception as e:
            if self.progress_callback:
                self.progress_callback(100, f"Greedy algorithm execution error: {str(e)}")
            return []
    
    def _construct_approximate_solution(self, n, base_count):
        """Construct approximate solution"""
        # Generate combinations using bitmask version
        all_masks = self._generate_bitmasks(n, self.k)
        
        if len(all_masks) > 1000:
            all_masks = random.sample(all_masks, 1000)
        
        # Randomly select base_count combinations
        solution = []
        for _ in range(min(base_count, len(all_masks))):
            idx = random.randint(0, len(all_masks) - 1)
            solution.append(set(self._mask_to_indices(all_masks[idx], n)))
            all_masks.pop(idx)
        
        return solution
        
    def evaluate_solution_quality(self, solution, n, coverage_frequency):
        """Evaluate solution quality
        Lower return value indicates higher quality
        """
        if not solution:
            return float('inf')
            
        # Base quality = number of combinations
        quality = len(solution)
        
        # Calculate coverage frequency distribution for subsets of size j
        j_subsets = list(combinations(range(n), self.j))
        coverage_counts = []
        
        for j_subset in j_subsets:
            j_set = set(j_subset)
            cover_count = sum(1 for sol in solution if len(j_set & sol) >= self.s)
            coverage_counts.append(cover_count)
        
        # Evaluate quality based on how many j-subsets are covered at least f times
        properly_covered = sum(1 for count in coverage_counts if count >= coverage_frequency)
        coverage_ratio = properly_covered / len(j_subsets) if j_subsets else 0
        
        # Add coverage ratio penalty (lower ratio = higher penalty)
        coverage_penalty = (1 - coverage_ratio) * 100
        
        # Calculate over-coverage (excess beyond required frequency)
        excess_coverage = sum(max(0, count - coverage_frequency) for count in coverage_counts)
        if j_subsets:
            avg_excess = excess_coverage / len(j_subsets)
            # Add over-coverage penalty
            quality += avg_excess * 0.1
        
        # Total quality score combines combination count, coverage ratio, and over-coverage
        return quality + coverage_penalty 