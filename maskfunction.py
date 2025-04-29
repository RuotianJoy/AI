import itertools
import numpy as np
from tqdm import tqdm
import time

'''第一步生成k_group{0,1}^n和j_group{0,1}^n的矩阵图，被选择中为1 不被选中则为0 ：check
    第二步k_group点乘j_group 的转置矩阵求出相似的个数，记录成矩阵                check
    第三步将相似度小于等于s的位置提取出来
    第四步从j_group中移除已被选择的样本
    第五步如果ls 大于1 则另外由j_group生成s_group
        1.记录下j_group样本位置信息然后通过combinenation 去生成s_group
        2.通过s_group的组合信息拓展成全新的{0,1}矩阵
        3.然后让s_group与k做点乘进行比较 如果=s的数量在ls之下就讲j退回j_group
        4.释放s_group然后生成下一个s_group
        5.遍历完全部j_group后解放select_j_group
        6.重复1-5
    重复步骤直到j_group归0
    ps:速度优化可以使用位掩码的方法来加速运算'''

"""需要的函数有：  生成矩阵的函数 generate_matrix()
                转置矩阵 matrix.T
                点乘函数 np.dot()
                提取函数 extract()
                总生成函数 greedy_selection()
                解码函数 
                随机生成用例 random_samples()   
                """


def random_samples(m, n):
    samples = np.random.choice(range(1, m + 1), n, replace=False)
    samples.sort()
    return samples


def generate_bitmasks(size, choice):
    """
    生成位掩码版本的组合矩阵
    每个组合表示为一个整数，其二进制表示中的1表示元素被选中
    """
    position = list(itertools.combinations(range(size), choice))
    bitmasks = []
    for pos in position:
        mask = 0
        for p in pos:
            mask |= (1 << p)  # 将p位置的位设为1
        bitmasks.append(mask)
    return bitmasks


def count_bits(n):
    """计算整数的二进制表示中1的个数"""
    count = 0
    while n:
        n &= (n - 1)  # 清除最低位的1
        count += 1
    return count


def count_bigger_than_s_bitmask(k_masks, j_masks, s):
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
            if count_bits(intersection) >= s:
                count += 1
        counts.append(count)
    return counts


def find_max_index(lst):
    """找出列表中最大值的索引"""
    return lst.index(max(lst))


def extract_mask(masks, index):
    """提取指定索引的掩码并从列表中删除"""
    mask = masks[index]
    del masks[index]
    return mask, masks


def filter_j_masks(k_mask, j_masks, s):
    """
    过滤出与k_mask相似度大于等于s的j_masks
    返回剩余的j_masks和被移除的数量
    """
    remaining = []
    removed = []
    for j in j_masks:
        intersection = k_mask & j
        if count_bits(intersection) >= s:
            removed.append(j)
        else:
            remaining.append(j)
    return remaining, len(removed)


def when_have_ls_bitmask(select_j_masks, selected_k_masks, s, ls, size):
    """
    使用位掩码实现when_have_ls函数
    """
    fail_j = []
    for j_mask in select_j_masks:
        # 获取j_mask中设置为1的位置
        j_positions = [i for i in range(size) if (j_mask & (1 << i))]

        # 生成所有s大小的子集的位掩码
        s_masks = []
        for s_pos in itertools.combinations(j_positions, s):
            s_mask = 0
            for p in s_pos:
                s_mask |= (1 << p)
            s_masks.append(s_mask)

        # 检查是否满足ls条件
        match_count = 0
        for k_mask in selected_k_masks:
            for s_mask in s_masks:
                intersection = k_mask & s_mask
                if count_bits(intersection) > s:
                    match_count += 1
                    break  # 找到一个满足条件的s_mask就可以了

        if match_count < ls:
            fail_j.append(j_mask)

    return fail_j


def mask_to_indices(mask, size):
    """将位掩码转换为索引列表"""
    return [i for i in range(size) if (mask & (1 << i))]


def decode_bitmask(sample, select_k_masks, size):
    """解码位掩码版本的结果"""
    results = []
    for k_mask in select_k_masks:
        # 获取k_mask中为1的位置
        selected_indices = mask_to_indices(k_mask, size)
        # 根据选定的索引获取样本
        selected_sample = [sample[i] for i in selected_indices]
        results.append(selected_sample)
    return results


def greedy_search_bitmask(n, k, j, s, ls=0):
    """使用位掩码的贪心搜索算法"""
    start_time = time.time()

    selected_k_masks = []
    k_masks = generate_bitmasks(n, k)
    j_masks = generate_bitmasks(n, j)

    total_j = len(j_masks)
    pbar = tqdm(total=total_j)

    while j_masks:
        # 计算每个k_mask覆盖的j_mask数量
        coverage_counts = count_bigger_than_s_bitmask(k_masks, j_masks, s)

        # 找到覆盖最多j_mask的k_mask
        best_index = find_max_index(coverage_counts)

        # 提取并保存最优的k_mask
        best_k_mask, k_masks = extract_mask(k_masks, best_index)
        selected_k_masks.append(best_k_mask)

        # 过滤掉被覆盖的j_masks
        j_masks, removed_count = filter_j_masks(best_k_mask, j_masks, s)

        # 如果有ls条件，执行额外的处理
        if ls > 0 and removed_count > 0:
            # 这里需要实现ls相关的逻辑
            pass

        pbar.update(removed_count)

    pbar.close()
    end_time = time.time()
    print(f"处理完成，用时: {end_time - start_time:.2f}秒")

    return selected_k_masks


if __name__ == "__main__":
    n = 16
    k = 6
    j = 6
    s = 5
    ls = 1

    # 使用位掩码版本的算法
    selected_k_masks = greedy_search_bitmask(n, k, j, s, ls)

    # 随机生成样本用于解码
    sample = random_samples(45, n)

    # 解码结果
    last_result = decode_bitmask(sample, selected_k_masks, n)

    # 打印结果
    print(f"选择的组合数量: {len(selected_k_masks)}")
    print("结果:")
    for result in last_result:
        print(result)
