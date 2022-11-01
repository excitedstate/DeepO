"""
    该文件对get_plan.py中的程序多了一个限制, 当优化后的查询计划出现 canceling时, 去掉
"""

# %%
import os
from get_plan import get_query_plan

# %%
with open("../example/SQL_with_hint/0", "r") as f:
    queries_with_hint = f.readlines()

# %%
optimized_plan_files = os.listdir("../example/optimized plans")
for file in optimized_plan_files:
    print(file)
    with open(os.path.join("../example/optimized plans", file), "r") as f:
        operators = []
        optimized_plan = f.readlines()
        if ("canceling" in optimized_plan[0]):
            query = queries_with_hint[int(file)].replace("explain analyse", "explain")
            plan = get_query_plan(query, save_path="../example/optimized plans/{}".format(file))

        else:
            continue
    # break
# %%
