import itertools
import threading
import sys
from time import time

best_solution = None
best_length = float('inf')
termination_event = threading.Event()  # 线程间同步事件
best_solution_found_by = None  # 用于记录找到最优解的算法名称

def main():
    global best_solution, best_length, termination_event, best_solution_found_by

    # 输入参数验证
    while True:
        try:
            m = int(input("输入总样本数 m (45 ≤ m ≤ 54): "))
            n = int(input("输入样本数 n (7 ≤ n ≤ 25): "))
            k = int(input("输入每组样本的大小 k (4 ≤ k ≤ 7): "))
            j = int(input("输入子集大小 j (s ≤ j ≤ k): "))
            s = int(input("输入子集的最小元素数量 s (3 ≤ s ≤ 7): "))

            assert 45 <= m <= 54 and 7 <= n <= 25 and 4 <= k <= 7
            assert s <= j <= k and 3 <= s <= 7 and j <= k
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

    # 预处理组合的覆盖信息
    k_combinations = list(itertools.combinations(samples, k))
    comb_masks = []
    for comb in k_combinations:
        elements = set(comb)
        mask = 0
        for subset in subsets:
            if len(elements & set(subset)) >= s:
                mask |= 1 << subset_indices[subset]
        comb_masks.append(mask)

    # 启动IDDFS算法
    try:
        # 算法1：迭代加深DFS
        t1 = threading.Thread(target=iterative_deepening_dfs,
                              args=(comb_masks, k_combinations, num_subsets, "Iterative Deepening DFS"))
        t1.start()

        start_time = time()
        while time() - start_time < 600:  # 600秒超时
            if termination_event.is_set():
                break


        t1.join(0.1)  # 等待线程结束

    except KeyboardInterrupt:
        termination_event.set()
        t1.join()
        sys.exit(1)

    # 输出结果
    if best_solution:
        print(f"找到解，组合数量：{len(best_solution)}")
        print("组合列表：", [tuple(sorted(k_combinations[i])) for i in best_solution])
    else:
        print("未找到解或超时！")

    termination_event.set()


def iterative_deepening_dfs(comb_masks, k_combinations, num_subsets, algorithm_name):
    global best_solution, best_length, termination_event, best_solution_found_by

    max_depth = 1

    def dfs(current_depth, current_mask, selected_indices, path, max_depth):
        global best_solution, best_length, termination_event, best_solution_found_by

        if termination_event.is_set() or current_depth >= best_length:
            return
        if current_mask == 0:
            if current_depth < best_length:
                best_solution = path.copy()
                best_length = current_depth
                best_solution_found_by = algorithm_name  # 记录找到最优解的算法名称
                termination_event.set()
            return

        available = sorted(
            [i for i in range(len(comb_masks)) if i not in selected_indices],
            key=lambda x: -((comb_masks[x] & current_mask).bit_count())
        )
        for comb_idx in available:
            if (comb_masks[comb_idx] & current_mask) == 0:
                continue
            new_mask = current_mask & (~comb_masks[comb_idx])
            new_selected = selected_indices | {comb_idx}
            dfs(current_depth + 1, new_mask, new_selected, path + [comb_idx], max_depth)

    while not termination_event.is_set():
        dfs(0, (1 << num_subsets) - 1, set(), [], max_depth)
        if best_solution is not None:
            return
        max_depth += 1
        if max_depth > len(comb_masks):
            break


if __name__ == "__main__":
    main()



