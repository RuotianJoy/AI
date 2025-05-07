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
                if self.progress_callback:
                    self.progress_callback(90, f"Bitmask greedy algorithm found solution, combination count: {len(solution)}")
                
                # Convert result to sample ID format
                result = []
                for sol in solution:
                    group = [idx_to_sample[i] for i in sol]
                    result.append(sorted(group))
                    
                if self.progress_callback:
                    self.progress_callback(100, f"Greedy algorithm optimization completed, found {len(result)} combinations")
                
                return result
            else:
                if self.progress_callback:
                    self.progress_callback(95, "Bitmask greedy algorithm didn't find feasible solution, trying to construct approximate solution...")
                
                # Construct a basic solution
                approx_solution = self._construct_approximate_solution(n, n//2)
                
                if approx_solution:
                    # Convert result to sample ID format
                    result = []
                    for sol in approx_solution:
                        group = [idx_to_sample[i] for i in sol]
                        result.append(sorted(group))
                        
                    if self.progress_callback:
                        self.progress_callback(100, f"Constructed approximate solution with {len(result)} combinations")
                    
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