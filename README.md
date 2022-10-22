# 前言

> 本仓库来自[RUC-AIDB/DeepO](https://github.com/RUC-AIDB/DeepO), 在保留作者原本意图基础上做了一些小修改, 以符合自己的开发习惯
> 全是私货, 想学习原滋原味的DeepO, 请到作者仓库[RUC-AIDB/DeepO](https://github.com/RUC-AIDB/DeepO)
> 1. 改用PIP安装各种第三方库
> 2. 符合PEP8代码标准
> 3. 减少重复代码量
> 以下为施工日志

## 2022/10/18 准备工作

- [x] 拜读作者论文
- [x] 下载源码，建立自己仓库

## 2022/10/22 阅读源码

>> 按照作者的思路, 应当按照一下思路训练自己的代价学习器

```shell
# Generate scan embedding intermediate result
# 生成 AST的嵌入的中间表示结果
python scan_embedding.py
# embedding plan into sequence
# 查询计划嵌入序列, 还分了两个文件
python plan_to_seq.py
# train the Cost Learner
# 训练和评估
python cost_learning.py
# evaluate the learning performance
python cost_evaluation.py
```