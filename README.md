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

> > 按照作者的思路, 应当按照一下思路训练自己的代价学习器

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

## 2022/10/23 阅读源码

> > 已完成以下文件的研究, 应该是为了获取优化后的查询计划, 这三个就是一体的

1. `tmp/get_plan.py`
2. `tmp/tmp_for_example.py`
3. `tmp/get_plan_for_limit.py`

## 2022/11/1 获取所有的查询计划

1. `example/SQL/0`保存的是优化之前的, 没有加 `hint`的查询计划
2. `example/SQL_with_hint/0`保存到是各种不同的优化后的查询计划, 用作测试

### 不同类型的查询计划的生成

> > 论文中提到, 根据三种不同的执行策略生成了`hint`, 包括 `scan`(叶子节点), `join`(连接, 在非叶子节点)和`join order`(
> > 连接顺序)三个方面

1. 相关代码应该是`hint_generation.py`, 好像用到了`learnedcardinalities`这个库, 已经下载到`data`目录了
2. 和代价估计是一块的， 应该是论文中所阐述的第二阶段或者在线阶段, 目前主要研究离线阶段,
   就是查询计划嵌入...相关代码在`tmp/embedding`文件夹内

### SCAN_EMBEDDING.py CR

1. 事实上, 前两个函数明显可以用RE去做啊.
2. 把scan_embedding.py plan_to_seq.py整理了一下, 这两个文件将查询计划表示为一种树的形式

## 2022/11/16

> > 作者的实现有很多BUG, 也能看出对Python并不熟练
> 1. `Seq Scan`的嵌入的第一阶段貌似是获取查询计划中叶子节点的关键信息, 但是对模式串匹配的并不好(本人对Embedding了解不多,
     但感觉这样会导致很多问题), 做字典(`vocab_dict`)的时候 很多不算词汇的都被识别为词汇