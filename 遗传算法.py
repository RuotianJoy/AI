import itertools
import random

from collections import defaultdict

from deap import base, creator, tools, algorithms
import time
from typing import List

if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMax)
class GeneticOptimizer:
    def __init__(self, samples: List[str], j: int, s: int, k: int, f: int):  # 增加f参数
        if not (3 <= s <= j <= k <= len(samples)):
            raise ValueError(f"Invalid parameters: 3≤s≤j≤k≤样本数, got s={s}, j={j}, k={k}, samples={len(samples)}")
        if f < 1:
            raise ValueError("覆盖次数f必须≥1")
        self.samples = samples
        self.j = j
        self.s = s
        self.k = k
        self.f = f  # 新增属性
        self.j_subsets = list(itertools.combinations(samples, j))

        self.toolbox = base.Toolbox()
        self._register_operations()

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

    def optimize(self, population_size=100, generations=150, cxpb=0.65, mutpb=0.35):
        """参数匹配的优化执行"""
        pop = self.toolbox.population(n=population_size)
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", lambda x: sum(v[0] for v in x)/len(x))

        l= int((2 * population_size * 1.15) // 1)
        print(l)

        algorithms.eaMuPlusLambda(
            pop, self.toolbox, mu=population_size, lambda_=l,
            cxpb=cxpb, mutpb=mutpb, ngen=generations,
            stats=stats, halloffame=hof, verbose=True
        )
        return hof[0]

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

    print(f"\n最优解包含 {len(solution)} 个组合:")
    for i, group in enumerate(solution, 1):
        print(f"组合{i}: {', '.join(group)}")



if __name__ == "__main__":
    print("start")
    """获取用户参数输入"""
    params = {
        'm': get_int("总样本数m (45-54): ", 45, 54),
        'n': get_int("选择样本数n (7-25): ", 7, 25),
        'k': get_int("组合大小k (4-7): ", 4, 7),
        'j': get_int("子集参数j (>=s): ", 3, None),
        's': get_int("覆盖参数s (3-7): ", 3, 7),
        'f': get_int("覆盖次数f (≥1): ", 1, None),  # 新增参数
        'run_id': int(time.time())
    }
    params['j'] = max(params['j'], params['s'])
    samples = select_samples(params)
    optimizer = GeneticOptimizer(samples, params['j'],
                                 params['s'], params['k'], params['f'])  # 传递f参数
    print("\n正在优化组合...")
    best_solution = optimizer.optimize()
    show_results(best_solution)



