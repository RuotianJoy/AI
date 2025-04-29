import itertools
import numpy as np
from tqdm import tqdm
from scipy.sparse import csr_matrix

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


def generate_matrix(size, choice):
    # 生成所有组合的位置引索
    position = list(itertools.combinations(range(size), choice))
    # 初始化矩阵
    matrix = np.zeros((len(position), size), dtype=np.uint8)
    # 将对应位置全部转换成1
    for i, pos in enumerate(position):
        matrix[i, list(pos)] = 1
    return matrix

def generate_s_group(s_position,n):
    result = []
    for group in s_position:
        arr = [0]*n
        for idx in group:
            arr[idx] = 1
        result.append(arr)
    return result


def count_bigger_than_s(matrix, s):
    # 对比每一个的覆盖率
    np_matrix = np.array(matrix)
    return np.sum(np_matrix >= s, axis=1)

def count_less_than_s(matrix, s):
    # 对比每一个的覆盖率
    np_matrix = np.array(matrix)
    return np.sum(np_matrix < s, axis=1)

def extract_vector(matrix, selected, index):
    # 提取出当前最优解
    selected.append(matrix[index].tolist())
    matrix = np.delete(matrix, index, axis=0)
    return matrix


def find_max(matrix):
    # 返回当前最大值的位置
    return np.argmax(matrix)


def filter_rows(value, j_group, index, s):
    selected = []
    # 根据最大值的位置找出在value中对应的行
    selected_row = value[index]
    mask = (selected_row >= s)
    # 生成布尔掩码来找出被覆盖的j的位置
    column_indices = np.where(mask)[0]
    if j_group.shape[0] != value.shape[1]:
        raise ValueError('Matrix_b do not have the same number of rows')
    selected.extend(j_group[i].tolist() for i in column_indices)
    nj_group = np.delete(j_group, column_indices, axis=0)
    return nj_group, len(column_indices)


def Pruning (value,k_group):
    zero_row = np.where(np.all(value == 0,axis=1))[0]
    if zero_row.size != 0:
        k_group = np.delete(k_group, zero_row, axis=0)
    return k_group

def bitwise_dot(matrix_a, matrix_b_T):
    """使用按位与运算代替点乘操作
    matrix_a: 第一个矩阵
    matrix_b_T: 第二个矩阵的转置
    """
    rows_a, cols_a = matrix_a.shape
    cols_b, rows_b = matrix_b_T.shape  # 注意这里是转置矩阵的形状
    
    result = np.zeros((rows_a, rows_b), dtype=np.int32)
    
    for i in range(rows_a):
        for j in range(rows_b):
            # 对每列进行与运算，然后求和
            # 这里要确保形状一致：matrix_a[i] 是 (cols_a,)，matrix_b_T[:,j] 也是 (cols_a,)
            result[i, j] = np.sum(np.bitwise_and(matrix_a[i], matrix_b_T[:,j]))
    
    return result

def when_have_ls(select_j, selected_k, s, ls,n):
    fail_j = []
    for j in select_j:
        j_position = [i for i, x in enumerate(j) if x == 1]
        #获取j 的位置信息用这些信息重新组合
        s_position = list(itertools.combinations(j_position, s))
        #拓展s——group的大小使得能与k做点乘
        s_group = np.array(generate_s_group(s_position,n))
        #使用与运算代替点乘
        compare_num = bitwise_dot(np.array([selected_k]), s_group.T)[0]
        if np.sum(compare_num > s) < ls:
            fail_j.append(j)
    return fail_j


def decode (sample,select_k):
    last_answer = np.array(sample)
    mask_np = np.array(select_k, dtype=bool)
    return [last_answer[select_k].tolist() for select_k in mask_np]

def greedy_search(n,k,j,s):
    selected_k = []
    selected_j = []
    k_matrix = generate_matrix(n, k)
    j_matrix = generate_matrix(n, j)
    while(len(j_matrix) != 0):
        # 使用与运算代替点乘
        value = bitwise_dot(k_matrix, j_matrix.T)
        bts = count_bigger_than_s(value,s)
        index_k = find_max(bts)
        k_matrix = extract_vector(k_matrix,selected_k,index_k)
        j_matrix,added_count = filter_rows(value,j_matrix,index_k,selected_j,s)

    return selected_k


if __name__ == "__main__":
    n=7
    k = 6
    j = 5
    s = 5
    ls = 1
    last_result = []
    selected_k = []
    success_k = []
    selected_j = []
    k_matrix = generate_matrix(n, k)
    j_matrix = generate_matrix(n, j)
    pbar = tqdm(total=len(j_matrix))
    while (len(j_matrix) != 0):
        # 使用与运算代替点乘
        value = bitwise_dot(k_matrix, j_matrix.T)
        bts = count_bigger_than_s(value, s)
        lts = count_less_than_s(value,s)
        print(lts)
        print(bts)
        index_k = find_max(bts)
        k_matrix = extract_vector(k_matrix, selected_k, index_k)
        j_matrix, added_count = filter_rows(value, j_matrix, index_k, s)

        # success_k.append(selected_k[0])
        # when_ls = np.array(when_have_ls(selected_j, selected_k,s,ls,n))
        # selected_k = []
        # selected_j = []
        # j_matrix = np.vstack([j_matrix, when_ls])
        pbar.update(added_count)
    pbar.close()
    # selected_k = greedy_search(n,k,j,s)
    sample = random_samples(45,n)
    last_result = decode(sample,selected_k)
    print(len(selected_j))
    print(last_result)
