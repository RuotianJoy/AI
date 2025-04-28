import itertools
import random
import time
from collections import defaultdict

from deap import base, creator, tools, algorithms
from typing import List

if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMax)

class GeneticOptimizer:
    def __init__(self, samples: List[str], j: int, s: int, k: int, f: int):
        if not (3 <= s <= j <= k <= len(samples)):
            raise ValueError(f"Invalid parameters: 3≤s≤j≤k≤样本数, got s={s}, j={j}, k={k}, samples={len(samples)}")
        if f < 1:
            raise ValueError("覆盖次数f必须≥1")
        self.samples = samples
        self.j = j
        self.s = s
        self.k = k
        self.f = f
        self.population_size = 100
        self.max_generations = 100
        self.mutation_rate = 0.1
        self.elitism_rate = 0.2
        self.progress_callback = None
        self.j_subsets = list(itertools.combinations(samples, j))

        self.toolbox = base.Toolbox()
        self._register_operations()

    def set_progress_callback(self, callback):
        """Set a callback function for progress updates"""
        self.progress_callback = callback
    
    def update_progress(self, percent, message=None):
        """Update progress if callback is set"""
        if self.progress_callback:
            self.progress_callback(percent, message)

    def _register_operations(self):
        """注册遗传算法操作"""
        self.toolbox.register("individual", self.init_individual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("evaluate", self.evaluate)
        self.toolbox.register("mate", self.crossover)
        self.toolbox.register("mutate", self.mutate)
        self.toolbox.register("select", tools.selTournament, tournsize=3)

    def init_individual(self):
        """基于覆盖优先的初始化策略"""
        candidate_size = max(50, 2 * len(self.samples))
        candidate_groups = [tuple(sorted(random.sample(self.samples, self.k))) for _ in range(candidate_size)]
        candidate_groups.sort(
            key=lambda g: sum(len(set(g) & set(j_sub)) >= self.s for j_sub in self.j_subsets),
            reverse=True
        )
        return creator.Individual(candidate_groups[:random.randint(3, 5)])

    def evaluate(self, individual):
        """优化后的适应度评估，考虑覆盖次数"""
        if len(self.j_subsets) > 10000:
            sampled_j = random.sample(self.j_subsets, 10000)
        else:
            sampled_j = self.j_subsets

        # 统计每个j_sub的被覆盖次数
        coverage_count = {j_sub: 0 for j_sub in sampled_j}
        for j_sub in sampled_j:
            j_set = set(j_sub)
            for group in individual:
                if len(j_set & set(group)) >= self.s:
                    coverage_count[j_sub] += 1

        # 计算满足条件的比例
        satisfied = sum(1 for cnt in coverage_count.values() if cnt >= self.f)
        coverage_score = satisfied / len(sampled_j)

        # 计算覆盖次数缺口
        coverage_gap = sum(max(self.f - cnt, 0) for cnt in coverage_count.values())
        gap_penalty = coverage_gap / (len(sampled_j) * self.f)  # 标准化惩罚项

        # 动态权重调整
        if coverage_score >= 0.99:
            fitness = 5 * coverage_score - 0.5 * len(individual) / len(self.samples) - 0.2 * gap_penalty
        elif coverage_score >= 0.95:
            fitness = 3 * coverage_score - 0.3 * len(individual) / len(self.samples) - 0.3 * gap_penalty
        else:
            fitness = coverage_score - 0.5 * gap_penalty

        return (fitness,)

    def crossover(self, ind1, ind2):
        combined = list(set(ind1 + ind2))
        combined.sort(
            key=lambda g: sum(len(set(g) & set(j_sub)) >= self.s for j_sub in self.j_subsets),
            reverse=True
        )
        split_point = random.randint(1, len(combined)-1)
        return (
            creator.Individual(combined[:split_point]),
            creator.Individual(combined[split_point:])
        )

    def mutate(self, individual):
        """变异操作"""
        mutated_ind = list(individual)
        rand_val = random.random()

        # 第一阶段：覆盖次数不足f次的j元组 (70%概率)
        if rand_val < 0.7:
            # 统计所有j_sub的当前覆盖次数
            coverage_count = defaultdict(int)
            for j_sub in self.j_subsets:
                j_set = set(j_sub)
                for group in mutated_ind:
                    if len(j_set & set(group)) >= self.s:
                        coverage_count[j_sub] += 1

            # 获取覆盖不足的j元组（覆盖次数 < f）
            under_covered = [j_sub for j_sub in self.j_subsets
                             if coverage_count[j_sub] < self.f]

            if under_covered:
                # 随机选择需要补充覆盖的j元组
                target_j = random.choice(under_covered)

                # Step 1: 包含目标j元组中的s个样本
                required = random.sample(target_j, self.s)

                # Step 2: 补充剩余样本（允许重复采样）
                remaining_pool = [s for s in self.samples if s not in target_j]
                try:
                    # 优先无重复采样
                    if len(remaining_pool) >= self.k - self.s:
                        remaining = random.sample(remaining_pool, self.k - self.s)
                    else:
                        # 当剩余样本不足时允许重复（需k-s <= len(remaining_pool)）
                        remaining = random.choices(remaining_pool, k=self.k - self.s)
                except ValueError:
                    remaining = []

                # 生成新组合并去重
                new_group = tuple(sorted(required + remaining))
                if new_group not in mutated_ind:
                    mutated_ind.append(new_group)

        # 第二阶段：定向删除冗余组 (40%概率)
        if random.random() < 0.9 and len(mutated_ind) > 2:
            group_contributions = [
                (g, sum(1 for j_sub in self.j_subsets if len(set(g) & set(j_sub)) >= self.s))
                for g in mutated_ind
            ]
            min_contribution = min(c for _, c in group_contributions)
            candidates = [g for g, c in group_contributions if c == min_contribution]
            if candidates:
                try:
                    mutated_ind.remove(random.choice(candidates))
                except ValueError:
                    pass

        # 第三阶段：随机变异保底 (10%概率)
        if random.random() < 0.1:
            # 随机添加/删除
            if random.choice([True, False]):
                # 添加新组
                new_group = tuple(sorted(random.sample(self.samples, self.k)))
                if new_group not in mutated_ind:
                    mutated_ind.append(new_group)
            else:
                # 删除旧组（保证最小数量）
                if len(mutated_ind) > 2:
                    mutated_ind.pop(random.randrange(len(mutated_ind)))

        if len(mutated_ind) < 2:
            mutated_ind.extend(self.toolbox.individual()[:2])

        return (creator.Individual(mutated_ind),)

    def optimize(self, population_size=100, generations=150, cxpb=0.5, mutpb=0.2):
        """优化执行，支持进度更新"""
        # Initialize parameters
        self.population_size = population_size
        self.max_generations = generations


        # Create initial population
        pop = self.toolbox.population(n=population_size)
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", lambda x: sum(v[0] for v in x)/len(x))
        
        # Start evolution
        best_fitness = 0
        
        # Initial progress update
        self.update_progress(0, "Starting optimization...")
        
        # Loop through generations
        for gen in range(generations):
            # Calculate progress percentage
            progress = int((gen / generations) * 100)
            
            # Update progress
            status_message = f"Generation {gen+1}/{generations}"
            self.update_progress(progress, status_message)
            
            # Select the next generation individuals
            offspring = self.toolbox.select(pop, len(pop))
            
            # Vary the pool of individuals
            offspring = algorithms.varAnd(offspring, self.toolbox, cxpb, mutpb)
            
            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            
            # Update the hall of fame with the generated individuals
            if hof is not None:
                hof.update(offspring)
                
                # Check if we have a new best individual
                if hof[0].fitness.values[0] > best_fitness:
                    best_fitness = hof[0].fitness.values[0]
                    self.update_progress(progress, f"New best solution found (fitness: {best_fitness:.2f})")
            
            # Replace the current population by the offspring
            pop[:] = offspring
            
            # Add a small delay to allow UI to update
            time.sleep(0.02)
            
        # Final progress update
        self.update_progress(100, "Optimization complete")
        
        # Return the best solution
        return list(hof[0])
    
    def initialize_population(self):
        """Create initial population of random combinations"""
        population = []
        
        # Create valid combinations - each with k samples
        for _ in range(self.population_size):
            individual = []
            available_samples = self.samples.copy()
            
            # Build individual combinations
            for _ in range(self.f):
                # Ensure we have enough samples
                if len(available_samples) >= self.k:
                    combination = random.sample(available_samples, self.k)
                    individual.append(combination)
                    
                    # For diversity, we might remove some selected samples
                    for sample in combination:
                        if random.random() < 0.3 and len(available_samples) > self.k:
                            available_samples.remove(sample)
            
            # Ensure minimum size
            if len(individual) < 1:
                individual = [random.sample(self.samples, self.k)]
                
            population.append(individual)
        
        return population
    
    def evaluate_fitness(self, individual):
        """Calculate fitness of an individual solution"""
        # Implement fitness evaluation based on coverage requirements
        fitness = 0
        
        # Check if the solution has any combinations
        if not individual:
            return 0
        
        # Count how many j-sized combinations are covered by the solution
        total_combinations = list(itertools.combinations(self.samples, self.j))
        covered_combinations = set()
        
        for combination in individual:
            # Get all j-sized subsets from this combination
            for j_subset in itertools.combinations(combination, self.j):
                # Convert to tuple for set operations
                covered_combinations.add(j_subset)
        
        # Calculate coverage rate
        coverage_rate = len(covered_combinations) / len(total_combinations)
        
        # Calculate s-wise coverage
        s_coverage = self.calculate_s_coverage(individual)
        
        # Total fitness is weighted combination of coverage rate and solution size
        fitness = (coverage_rate * 0.7) + (s_coverage * 0.3) - (len(individual) * 0.01)
        
        return fitness
    
    def calculate_s_coverage(self, individual):
        """Calculate s-wise coverage of the solution"""
        # Count how many s-sized combinations are covered
        total_s_combinations = list(itertools.combinations(self.samples, self.s))
        covered_s_combinations = set()
        
        for combination in individual:
            # Get all s-sized subsets from this combination
            for s_subset in itertools.combinations(combination, self.s):
                # Convert to tuple for set operations
                covered_s_combinations.add(s_subset)
        
        # Calculate s-coverage rate
        if total_s_combinations:
            return len(covered_s_combinations) / len(total_s_combinations)
        return 0
    
    def selection(self, population, fitness_scores):
        """Select individuals for reproduction based on fitness"""
        # Tournament selection
        parents = []
        
        for _ in range(self.population_size):
            # Select random individuals for tournament
            tournament_size = 3
            tournament_indices = random.sample(range(len(population)), tournament_size)
            
            # Get the best individual from tournament
            best_index = max(tournament_indices, key=lambda i: fitness_scores[i])
            parents.append(population[best_index])
        
        return parents
    
    def is_optimal_solution(self, solution, fitness):
        """Check if solution meets all requirements"""
        # This is a simplified check
        # For real applications, you would implement more detailed checks
        
        # Example: Consider solution optimal if fitness is above a threshold
        fitness_threshold = 0.95
        return fitness > fitness_threshold
    
    def format_solution(self, solution):
        """Format the solution for output"""
        # Convert internal representation to final output format
        formatted = []
        
        for combination in solution:
            # Sort the combination for consistent display
            sorted_combination = sorted(combination)
            formatted.append(sorted_combination)
            
        # Sort the entire solution for consistent display
        formatted.sort()
        
        return formatted 