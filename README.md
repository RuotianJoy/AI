# Optimal Samples Selection System

一个智能样本组合优化系统，使用遗传算法、模拟退火算法和贪心算法进行最佳样本选择。

## 系统概述

该系统帮助用户寻找满足特定覆盖要求的最佳样本组合。它提供了三种优化算法：
- 遗传算法（Genetic Algorithm）
- 模拟退火算法（Simulated Annealing）
- 贪心算法（Greedy Algorithm）

系统具有用户友好的GUI界面，并维护优化结果数据库以供参考。

## 功能特点

- **多算法支持**
  - 遗传算法优化
  - 模拟退火算法优化
  - 贪心算法优化
  - 实时进度跟踪
  - 详细结果可视化
  - 算法对比分析

- **参数配置**
  - 总样本数量 (m)：45-54
  - 选择样本数量 (n)：7-25
  - 组合大小 (k)：4-7
  - 子集参数 (j)：≥3
  - 覆盖参数 (s)：3-7
  - 覆盖次数 (f)：≥1

- **样本选择方法**
  - 随机选择
  - 手动输入

- **结果管理**
  - 将优化结果保存到数据库
  - 查看历史优化记录
  - 导出结果到文本文件或PDF
  - 删除不需要的记录

## 系统要求

- Python 3.6+
- PyQt5
- SQLite3
- DEAP（用于遗传算法）
- NumPy
- SciPy
- tqdm
- reportlab（用于PDF导出，可选）

## 安装

1. 克隆仓库：
```bash
git clone [repository-url]
cd [repository-name]
```

2. 安装所需包：
```bash
pip install -r requirements.txt
```

## 使用方法

1. 运行应用程序：
```bash
python main_window.py
```

2. 配置参数：
   - 设置总样本数量 (m)
   - 设置选择样本数量 (n)
   - 设置组合大小 (k)
   - 设置子集参数 (j)
   - 设置覆盖参数 (s)
   - 设置覆盖次数 (f)
   - 选择优化算法

3. 选择样本：
   - 选择随机生成或手动输入
   - 若手动输入，请输入以逗号分隔的样本编号

4. 执行优化：
   - 点击"执行"按钮
   - 实时监控进度
   - 在显示区域查看结果

5. 管理结果：
   - 将结果保存到数据库
   - 查看历史记录
   - 导出结果到文本文件或PDF
   - 删除不需要的记录

6. 算法对比：
   - 在对比选项卡中可以同时运行多种算法
   - 比较不同算法的性能和结果质量
   - 验证结果是否满足覆盖要求

## 数据库结构

系统使用SQLite3数据库，包含以下表：

- **runs**：存储优化运行信息
  - 参数 (m, n, k, j, s, f)
  - 时间戳
  - 执行时间
  - 使用的算法
  - 运行计数
  - 格式化ID

- **samples**：存储每次运行的选定样本
  - 运行ID
  - 样本编号

- **results**：存储优化结果
  - 运行ID
  - 组ID
  - 样本编号

## 文件结构

```
.
├── main_window.py          # 主应用程序入口和GUI实现
├── genetic_algorithm.py    # 遗传算法实现
├── simulated_annealing.py  # 模拟退火算法实现
├── greedy_optimizer.py     # 贪心算法实现
├── solution_validator.py   # 解决方案验证器
├── results.db              # SQLite数据库
├── requirements.txt        # Python依赖项
└── 算法流程.png             # 算法流程图
```

## 算法说明

- **遗传算法**：通过模拟自然选择过程，利用交叉、变异等操作进行样本组合的优化。
- **模拟退火算法**：通过模拟物理退火过程，以一定概率接受差解，避免陷入局部最优解。
- **贪心算法**：基于覆盖贡献度选择样本组合，每次选择能够增加最大覆盖率的组合。

## 贡献

欢迎贡献！请随时提交Pull Request。

## 联系方式

如有任何问题或建议，请联系项目维护者。

# GPU加速优化器

这个项目提供了三种常用组合优化算法的GPU加速实现：贪心算法、模拟退火算法和遗传算法。通过利用GPU的并行计算能力，可以显著提高这些算法在大规模组合问题上的性能。

## 功能特点

- **三种算法的GPU加速版本**：
  - 贪心算法 (Greedy)
  - 模拟退火算法 (Simulated Annealing)
  - 遗传算法 (Genetic Algorithm)

- **性能优势**：
  - 位掩码操作的批量并行处理
  - 矩阵计算的GPU加速
  - 多个解的并行评估

- **易于使用的接口**：
  - 与现有CPU版本保持一致的API
  - 工厂模式创建不同类型的优化器
  - 进度回调机制

## 环境要求

- Python 3.6+
- PyTorch 1.7+
- CUDA支持的GPU (对于GPU加速)
- NumPy

## 安装

1. 确保已安装CUDA和兼容的GPU驱动
2. 安装PyTorch (带CUDA支持)：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

3. 安装其他依赖：

```bash
pip install numpy
```

## 使用方法

### 基本用法

```python
from gpu_optimizer import GPUOptimizerFactory

# 准备数据
samples = [f"sample_{i}" for i in range(100)]
j = 3  # 子集参数
s = 2  # 覆盖参数
k = 4  # 组合大小
f = 1  # 覆盖次数

# 创建优化器 (可选: 'greedy', 'sa', 'ga')
optimizer = GPUOptimizerFactory.create_optimizer(
    'greedy',
    samples, j, s, k, f
)

# 设置进度回调
def progress_callback(percentage, message):
    print(f"进度: {percentage}%, {message}")

optimizer.set_progress_callback(progress_callback)

# 执行优化
result = optimizer.optimize()

# 打印结果
print(f"找到{len(result)}个组合:")
for i, group in enumerate(result):
    print(f"组合 {i+1}: {group}")
```

### 与CPU版本比较

使用提供的比较工具来对比CPU和GPU版本的性能：

```python
from gpu_optimizer_usage import compare_performance

# 生成测试数据
samples = [f"sample_{i}" for i in range(50)]

# 比较性能
compare_performance(samples, j=3, s=2, k=4, f=1)
```

### 比较不同算法

比较三种优化算法的性能和结果质量：

```python
from gpu_optimizer_usage import run_all_algorithms

# 生成测试数据
samples = [f"sample_{i}" for i in range(50)]

# 运行所有算法并比较
run_all_algorithms(samples, j=3, s=2, k=4, f=1)
```

## 算法说明

### 贪心算法 (Greedy)

贪心算法在每一步选择覆盖最多未覆盖j子集的k组合。GPU加速版本主要优化了位掩码操作和匹配矩阵的计算。

### 模拟退火算法 (Simulated Annealing)

模拟退火算法通过随机搜索找到近似最优解，避免陷入局部最优。GPU加速版本支持多个并行马尔可夫链，加快收敛速度。

### 遗传算法 (Genetic Algorithm)

遗传算法模拟生物进化过程，通过选择、交叉和变异操作搜索最优解。GPU加速版本支持种群并行评估和适应度计算。

## 性能提示

- 对于小规模问题 (n < 30)，CPU和GPU版本性能差异不明显
- 对于中等规模问题 (30 <= n < 100)，GPU版本可能提供2-10倍的加速
- 对于大规模问题 (n >= 100)，GPU版本可提供10-100倍的加速
- 遗传算法相比其他算法通常从GPU加速中获益最多

## 许可证

此项目使用MIT许可证 - 详情请参见LICENSE文件 

