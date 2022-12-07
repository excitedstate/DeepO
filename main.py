import logging
import os
import queue
import threading

from src.basic import PostgresDB, GeneralLogger
from src.config import DATA_PATH_PLANS_FOR_TRAIN, LC_SQL_CSV_FOR_TEST, DATA_PATH, DB_LAB_VM_CONFIG

MainLogger = GeneralLogger(name="test", stdout_flag=True, stdout_level=logging.INFO,
                           file_mode="a")


def get_train_data():
    os.makedirs(DATA_PATH_PLANS_FOR_TRAIN, exist_ok=True)

    filename = "scale"
    sql_path = os.path.join(LC_SQL_CSV_FOR_TEST, f"workloads\\{filename}.sql")
    output_dir = os.path.join(DATA_PATH, f"{filename}_plan")
    os.makedirs(output_dir, 0o777, True)
    q = queue.Queue()

    def __inner_thread(tid):
        db = PostgresDB(_config=DB_LAB_VM_CONFIG)
        while not q.empty():
            try:
                idx, sql = q.get_nowait()
                MainLogger.info(f"{idx}: {sql.strip()}")
                _plan = db.get_query_plan(sql)
                if isinstance(_plan, tuple):
                    # # 获取成功 写入文件
                    with open(os.path.join(output_dir, f'{filename}-{idx}'), "w", encoding="utf-8") as _out_f:
                        _out_f.write("\n".join(tuple(zip(*_plan))[0]))
            except queue.Empty:
                MainLogger.info(f"thread {tid} executed over!")
                break
            except Exception as e:
                MainLogger.warning(f"thread {tid} e: {e}, ignore")
                continue

    with open(sql_path, "r", encoding="utf-8") as f:
        for i, s in enumerate(f):
            q.put((i, s))
        threads = [threading.Thread(target=__inner_thread, args=(tid,)) for tid in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()


def embedding_and_train_cost_net():
    from src.embedding import Embedding
    from src.cost_learner import NetTrainer
    # # 获取数据
    data_x, data_y = Embedding().flow()

    nt = NetTrainer(
        data_x,
        data_y,
        epochs=10
    )
    nt.flow(plot=True)


def optimize_sql_query(idx=28):
    """

    """
    from src.hint_generate import CostEstimation
    from src.query_plan import QueryPlanNode

    ce = CostEstimation.load_from(idx, "data/output_2")
    res = ce.predict_costs()
    for plan in ce.query_plans:
        plan.post_order()
        print(plan)
    sorted_res = sorted(range(len(res)), key=lambda i: (res[i]['mean'], -res[i]['std']), reverse=True)

    def callback(cur_node: QueryPlanNode):
        # _layer = '\t' * cur_node.layer
        print(
            f"id: {cur_node.id:02d}, parent: {cur_node.parent.id if cur_node.parent is not None else 0:02d}, layer: {cur_node.layer}, vector: {list(cur_node.to_vector())}")

    print("sorted data: ")
    for i in sorted_res:
        print(
            f"std: {res[i]['std']:06f}, mean: {res[i]['mean']:06f}, real cost: {res[i]['inverse_function'](res[i]['mean']):06f}")
    print("sorted index: ", sorted_res)
    for i, idx in enumerate(sorted_res[:3]):
        print(f"top {i}: {idx}")
        print(f"sql with hint: {ce.sql_with_hints[idx].strip()}")
        print(f"query plan:")
        ce.query_plans[idx].post_order(callback)
    # for plan in ce.query_plans:
    #     plan.post_order()
    #     print()
    # print(ce.query_plans)


def get_width_of_confidence_intervals():
    # # p是第一个概率超过0.9的ci_multiplier
    from src.cost_learner import CostLearner
    import rich

    *_, res_ratio, p = CostLearner.get_width_of_confidence_intervals()
    print(p)
    for key, value in res_ratio:
        print(key, value, sep=",")
    rich.print(res_ratio)
    return p


def query_optimization_demo():
    # # 4.1 获取SQL语句的全部查询计划
    from src.hint_generate import SQLWithHintsGenerator, CostEstimation
    from src.query_plan import QueryPlanNode

    SQLWithHintsGenerator.test("data/output_2")

    # # 4.2 执行第28号SQL的查询优化
    ce = CostEstimation.load_from(28, "data/output_2")
    res = ce.predict_costs()
    for plan in ce.query_plans:
        plan.post_order()
        print(plan)
    sorted_res = sorted(range(len(res)), key=lambda i: (res[i]['mean'], -res[i]['std']), reverse=True)

    def callback(cur_node: QueryPlanNode):
        # _layer = '\t' * cur_node.layer
        print(
            f"id: {cur_node.id:02d}, parent: {cur_node.parent.id if cur_node.parent is not None else 0:02d}, layer: {cur_node.layer}, vector: {list(cur_node.to_vector())}")

    print("sorted data: ")
    for i in sorted_res:
        print(
            f"std: {res[i]['std']:06f}, mean: {res[i]['mean']:06f}, real cost: {res[i]['inverse_function'](res[i]['mean']):06f}")
    print("sorted index: ", sorted_res)
    for i, idx in enumerate(sorted_res[:3]):
        print(f"top {i}: {idx}")
        print(f"sql with hint: {ce.sql_with_hints[idx].strip()}")
        print(f"query plan:")
        ce.query_plans[idx].post_order(callback)
    # for plan in ce.query_plans:
    #     plan.post_order()
    #     print()
    # print(ce.query_plans)


if __name__ == '__main__':
    # # part 1: 获取训练数据
    get_train_data()
    # # part 2: embedding
    embedding_and_train_cost_net()
    # # part3: 获取最佳置信区间
    p = get_width_of_confidence_intervals()
    # # part4: 在线查询优化 demo
    query_optimization_demo()


