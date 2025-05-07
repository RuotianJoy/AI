import itertools
import pulp
import threading
import sys
from time import time

best_solution = None
best_length = float('inf')
termination_event = threading.Event()  # 线程间同步事件
best_solution_found_by = None  # 记录找到最优解的算法名称
iddfs_done = threading.Event()  # 标记IDDFS完成的全局事件

def main():
    global best_solution, best_length, termination_event, best_solution_found_by, iddfs_done

    # 输入参数验证
    while True:
        try:
            m = int(input("输入总样本数 m (45 ≤ m ≤ 54): "))
            n = int(input("输入样本数 n (7 ≤ n ≤ 25): "))
            k = int(input("输入每组样本的大小 k (4 ≤ k ≤ 7): "))
            j = int(input("输入子集大小 j (s ≤ j ≤ k): "))
            s = int(input("输入子集的最小元素数量 s (3 ≤ s ≤ 7): "))

            assert 45 <= m <= 54 and 7 <= n <= 25 and 4 <= k <= 7
            assert s <= j <= k and 3 <= s <= 7
            break
        except:
            print("输入参数不符合条件，请重新输入！")

    samples = list(range(1, n + 1))

    if j > k:
        print("无解，因为子集大小j不能超过组大小k")
        return

    # 预处理子集和它们的索引
    subsets = list(itertools.combinations(samples, j))
    subset_indices = {subset: idx for idx, subset in enumerate(subsets)}
    num_subsets = len(subsets)

    # 计算每个子集的最大可能覆盖次数（即最多能被多少个组覆盖）
    k_combinations = list(itertools.combinations(samples, k))
    max_cover_counts = []
    # for subset in subsets:
    #     subset_set = set(subset)
    #     count = 0
    #     for comb in k_combinations:
    #         if len(set(comb) & subset_set) >= s:
    #             count += 1
    #     max_cover_counts.append(count)

    # max_t_possible = min(max_cover_counts) if max_cover_counts else 0

    max_t_possible = 10

    if max_t_possible == 0:
        print("无解，因为无法满足最小覆盖次数要求")
        return

    # 输入t参数
    while True:
        try:
            t = int(input(f"输入每个子集需要被覆盖的次数 t (1 ≤ t ≤ {max_t_possible}): "))
            assert 1 <= t <= max_t_possible
            break
        except:
            print(f"输入参数不符合条件，请重新输入！（有效范围：1 ≤ t ≤ {max_t_possible})")

    # 预处理组合的覆盖信息
    comb_masks = []
    for comb in k_combinations:
        elements = set(comb)
        covered_subsets = []
        for subset in subsets:
            if len(elements & set(subset)) >= s:
                covered_subsets.append(subset_indices[subset])
        comb_masks.append(covered_subsets)

    # 计算理论下界
    max_cover = max(len(mask) for mask in comb_masks) if comb_masks else 0
    theoretical_lower = (num_subsets * t + max_cover - 1) // max_cover

    # 判断是否是大规模问题（n >= 14）
    is_large_problem = n >= 14

    # 启动两种算法并行运行
    threads = []
    try:
        # 线程1：迭代加深DFS（所有情况都运行）
        t1 = threading.Thread(target=iterative_deepening_dfs,
                              args=(comb_masks, k_combinations, num_subsets, t, theoretical_lower))
        threads.append(t1)

        # 线程2：ILP求解（无论n的大小都运行）
        t2 = threading.Thread(target=ilp_solver,
                              args=(comb_masks, k_combinations, num_subsets, t))
        threads.append(t2)

        for t in threads:
            t.start()

        # 根据问题规模选择不同的超时逻辑
        if is_large_problem:
            # 大规模问题：谁先完成就用谁的结果
            while not termination_event.is_set():
                time.sleep(0.1)
        else:
            # 小规模问题：等待15秒后比较结果
            for t in threads:
                t.join(15)  # 等待15秒

        # 强制终止所有线程
        for t in threads:
            t.join(0.1)

    except KeyboardInterrupt:
        termination_event.set()
        for t in threads:
            t.join()
        sys.exit(1)

    # 输出结果
    if best_solution:
        print(f"找到解，组合数量：{len(best_solution)}（理论下界：{theoretical_lower}）")
        print("组合列表：", [tuple(sorted(k_combinations[i])) for i in best_solution])
        print(f"最优解由算法 {best_solution_found_by} 找到")
    else:
        print("未找到解或超时！")

def ilp_solver(comb_masks, k_combinations, num_subsets, t):
    global best_solution, best_length, termination_event, best_solution_found_by

    # 创建ILP模型（隐藏输出信息）
    model = pulp.LpProblem("Subset_Cover", pulp.LpMinimize)

    # 决策变量：每个组合是否被选中
    x = pulp.LpVariable.dicts("x", range(len(k_combinations)), cat=pulp.LpBinary)

    # 目标函数：最小化选中的组合数量
    model += pulp.lpSum(x[i] for i in range(len(k_combinations)))

    # 约束条件：每个子集至少被覆盖t次
    for subset_idx in range(num_subsets):
        covering_combs = [i for i, mask in enumerate(comb_masks) if subset_idx in mask]
        model += pulp.lpSum(x[i] for i in covering_combs) >= t

    # 使用COIN_CMD求解器并设置超时（根据问题规模动态调整）
    try:
        # 尝试使用maxSeconds参数（新版本PuLP）
        solver = pulp.PULP_CBC_CMD(
            msg=0,          # 关闭求解器输出
            maxSeconds=60   # 默认超时60秒（主线程会额外处理）
        )
    except TypeError:
        # 回退使用timeLimit参数（旧版本PuLP）
        solver = pulp.PULP_CBC_CMD(
            msg=0,
            timeLimit=60
        )

    model.solve(solver)  # 调用求解器

    # 解析结果
    if pulp.LpStatus[model.status] == "Optimal":
        selected = [i for i in range(len(k_combinations)) if x[i].value() >= 0.9]
        if len(selected) < best_length:
            best_solution = selected.copy()
            best_length = len(selected)
            best_solution_found_by = "ILP"
            termination_event.set()

def iterative_deepening_dfs(comb_masks, k_combinations, num_subsets, t, theoretical_lower):
    global best_solution, best_length, termination_event, best_solution_found_by, iddfs_done

    max_depth = theoretical_lower

    def dfs(current_depth, remaining_counts, selected_indices, path, max_depth):
        global best_solution, best_length, termination_event, best_solution_found_by

        if termination_event.is_set() or current_depth >= best_length:
            return
        if all(count <= 0 for count in remaining_counts):
            if current_depth < best_length:
                best_solution = path.copy()
                best_length = current_depth
                best_solution_found_by = "IDDFS"
                termination_event.set()
            return

        # 动态排序候选组合（优先选择覆盖最多未满足子集的组合）
        available = sorted(
            [i for i in range(len(comb_masks)) if i not in selected_indices],
            key=lambda x: -sum(1 for idx in comb_masks[x] if remaining_counts[idx] > 0)
        )

        for comb_idx in available:
            new_remaining = remaining_counts.copy()
            for subset_idx in comb_masks[comb_idx]:
                if new_remaining[subset_idx] > 0:
                    new_remaining[subset_idx] -= 1
            dfs(current_depth + 1, new_remaining, selected_indices | {comb_idx}, path + [comb_idx], max_depth)

    initial_remaining = [t] * num_subsets
    while not termination_event.is_set():
        dfs(0, initial_remaining, set(), [], max_depth)
        if best_solution is not None:
            break
        max_depth += 1
        if max_depth > len(comb_masks):
            break

    # 标记IDDFS完成
    iddfs_done.set()

if __name__ == "__main__":
    main()