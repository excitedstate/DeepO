"""
    测试
"""
import functools
import json
import logging
import os
import queue
import threading

import rich
import tqdm

from src.basic import PostgresDB, GeneralLogger
from src.config import DB_LAB_VM_CONFIG, DATA_PATH_PLANS_FOR_TRAIN, DATA_PATH_NPY, LC_SQL_CSV_FOR_TEST, DATA_PATH, \
    DATA_PATH_LC_SQL_TRAIN_CSV, DATA_PATH_OUTPUT
from src.cost_learner import CostLearner

TestLogger = GeneralLogger(name="test", stdout_flag=True, stdout_level=logging.INFO,
                           file_mode="a")


def test_1():
    db = PostgresDB(_config=DB_LAB_VM_CONFIG)
    # # test sql
    test_sql = """SELECT * FROM aka_name WHERE aka_name."id" < 1000;"""
    res = db.get_query_plan(test_sql)
    # print(res)
    for _item in res:
        print(_item[0])
    # print("\n".join())


def test_2():
    """

    Returns:

    """
    db = PostgresDB(_config=DB_LAB_VM_CONFIG)
    test_sql = """/*+ SeqScan(t) SeqScan(mi) SeqScan(mi_idx) NestLoop(t mi)  """ + \
               """NestLoop(t mi mi_idx) Leading (( mi ( t mi_idx ) )) */  """ + \
               """explain analyse select count(*) from title t, movie_info mi, movie_info_idx mi_idx """ + \
               """where t.id=mi.movie_id and t.id=mi_idx.movie_id and mi.info_type_id > 16 and """ + \
               """mi_idx.info_type_id = 100;"""
    res = db.execute(test_sql)
    print("\n".join(map(lambda x: str(x[0]), res)))


def test_3():
    """
        把所有Seq Scan拿出来
    Returns:

    """
    import re
    from src.config import DATA_PATH_PLANS_FOR_TRAIN, DATA_PATH
    _out_f = open(os.path.join(DATA_PATH, "all_seq_scan.txt"), "w")
    _out_f_1 = open(os.path.join(DATA_PATH, "all_seq_scan_formatted.txt"), "w")
    _sentences = open(os.path.join(DATA_PATH, "sentences.txt"), "w")
    _seq_scan_set = list()
    _single_scan_rows_set = list()
    _ptn = re.compile("\s.*?Filter: \((.*)\)")
    _ptn_2 = re.compile("\) | \(")
    _ptn_3 = re.compile("'(.*?)'::text")
    _ptn_4 = re.compile(".*?Seq Scan on (.*?) (.*?)\s\s\(.*?\) \(actual time=.*?rows=(\d+).*?\)")
    for _file in sorted(os.listdir(DATA_PATH_PLANS_FOR_TRAIN)):
        _base_name = os.path.basename(_file)
        _last_line_is_seq_scan = False
        _single_scan_set = list()
        with open(os.path.join(DATA_PATH_PLANS_FOR_TRAIN, _base_name), "r", encoding="utf-8") as _in_f:
            for _line in _in_f:
                if _last_line_is_seq_scan:
                    _out_f.write(f"\t" + _line.strip() + "\n")
                    if "Filter" in _line:
                        _out_f_1.write(f"\t" + _line.strip() + "\n")
                        # # 将选择条件加入, 模式串
                        _temp_res_1 = _ptn.match(_line).groups()[0]
                        _temp_res_2 = _ptn_2.split(_temp_res_1)
                        # # _temp_res_3 应比 _temp_res_2 少1
                        # _temp_res_3 = _ptn_2.match(_line).groups()
                        # 用AND和OR分开, 还有一件事
                        _single_scan_set = functools.reduce(lambda _l1, _l2: _l1 + _l2,
                                                            map(lambda _item: _ptn_3.sub("", _item).split(),
                                                                _temp_res_2),
                                                            _single_scan_set)
                        # # 去掉两边的()
                        _single_scan_set = list(map(lambda x: x.rstrip(")").lstrip("("), _single_scan_set))
                    else:
                        # # 加一个空字符, 不知道有什么用, 复制一份, 不知道有什么用
                        _single_scan_set.append("")
                        _single_scan_set.extend(_single_scan_set[:2])
                    _seq_scan_set.append(_single_scan_set)
                    _single_scan_set = list()
                if "Seq Scan" in _line and 'never executed' not in _line:
                    _out_f.write(f"{_base_name} " + _line.strip() + "\n")
                    _out_f_1.write(f"{_base_name} " + _line.strip().replace("->  ", "") + "\n")
                    _ptn_4_match_res = _ptn_4.match(_line).groups()
                    _single_scan_set.extend(_ptn_4_match_res[:-1])
                    _single_scan_rows_set.append(int(_ptn_4_match_res[-1]))
                    _last_line_is_seq_scan = True
                else:
                    _last_line_is_seq_scan = False
    print(_seq_scan_set)
    print(_single_scan_rows_set)


def test_4():
    """
        把所有Seq Scan拿出来
    Returns:

    """
    import re

    import collections
    from src.config import DATA_PATH_PLANS_FOR_TRAIN, DATA_PATH
    _out_f = open(os.path.join(DATA_PATH, "all_operators_map.json"), "w")
    _out_f2 = open(os.path.join(DATA_PATH, "all_operators.json"), "w")
    _out_f3 = open(os.path.join(DATA_PATH, "all_operators_without_table.json"), "w")
    _operators = collections.defaultdict(list)
    for _file in sorted(os.listdir(DATA_PATH_PLANS_FOR_TRAIN)):
        _base_name = os.path.basename(_file)
        _last_line_is_seq_scan = False
        _single_scan_set = list()
        with open(os.path.join(DATA_PATH_PLANS_FOR_TRAIN, _base_name), "r", encoding="utf-8") as _in_f:
            for _line in _in_f:
                if "->" in _line:
                    _operators[_base_name].append(_line.replace("->", "").strip().split("  ")[0])
    _out_f.write(json.dumps(_operators))

    _out_f2.write(json.dumps(sorted(set(
        functools.reduce(lambda x, y: x + y, _operators.values(), list())
    ))))

    ptn3 = re.compile(" using .*| on .*")

    _out_f3.write(json.dumps(sorted(set(
        functools.reduce(
            lambda x, y: x + y,
            map(
                lambda l: list(map(
                    lambda s: ptn3.sub("", s).replace("Parallel ", ""),
                    l)
                ), _operators.values(),
            ),
            list()
        )
    ))))


def test_5():
    # with open(os.path.join(DATA_PATH_PLANS_FOR_TRAIN, "8b")) as query_plan_file:
    #     qp = QueryPlan(query_plan_file.read())
    #     qp.pre_order()
    #     qp.make_digraph(draw=True)
    from src.query_plan import QueryPlan

    for file_name in os.listdir(DATA_PATH_PLANS_FOR_TRAIN):
        with open(os.path.join(DATA_PATH_PLANS_FOR_TRAIN, file_name)) as query_plan_file:
            qp = QueryPlan(query_plan_file.read())
            qp.post_order()
            qp.make_digraph(draw=False)
    # all_words = list()
    # all_tables_1 = list()
    # all_tables_2 = list()
    # for item in qp.COND_EXP:
    #     print(item[0], ",".join(item[1:]), sep=', ')
    #     all_words.extend(item[1:])
    #     all_tables_1.append(item[1])
    #     all_tables_2.append((item[0], item[3]))
    # for item in set(all_tables_1):
    #     print(item)
    # print()
    # subset_a = set(filter(lambda x: "'" not in x[1] and '.' in x[1], all_tables_2))
    # for item in subset_a:
    #     print(item)
    # print()
    # for item in set(all_tables_2) - subset_a:
    #     print(item)
    # print(len(subset_a))
    # print(len(set(all_words)))


def test_6():
    from src.embedding import Embedding
    Embedding().flow()


def test_7():
    from src.cost_learner import NetTrainer
    nt = NetTrainer(
        os.path.join(DATA_PATH_NPY, "cost_learner_x.npy"),
        os.path.join(DATA_PATH_NPY, "cost_learner_y.npy"),
        epochs=10
    )
    nt.flow(plot=True)


def get_train_data():
    os.makedirs(DATA_PATH_PLANS_FOR_TRAIN, exist_ok=True)

    filename = "scale"
    scale_sql_path = os.path.join(LC_SQL_CSV_FOR_TEST, f"workloads\\{filename}.sql")
    output_dir = os.path.join(DATA_PATH, f"{filename}_plan")
    os.makedirs(output_dir, 0o777, True)
    q = queue.Queue()

    def __inner_thread(tid):
        db = PostgresDB(_config=DB_LAB_VM_CONFIG)
        while not q.empty():
            try:
                idx, sql = q.get_nowait()
                TestLogger.info(f"{idx}: {sql.strip()}")
                _plan = db.get_query_plan(sql)
                if isinstance(_plan, tuple):
                    # # 获取成功 写入文件
                    with open(os.path.join(output_dir, f'{filename}-{idx}'), "w", encoding="utf-8") as _out_f:
                        _out_f.write("\n".join(tuple(zip(*_plan))[0]))
            except queue.Empty:
                TestLogger.info(f"thread {tid} executed over!")
                break
            except Exception as e:
                TestLogger.warning(f"thread {tid} e: {e}, ignore")
                continue

    with open(scale_sql_path, "r", encoding="utf-8") as f:
        for i, s in enumerate(f):
            q.put((i, s))
        threads = [threading.Thread(target=__inner_thread, args=(tid,)) for tid in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()


def test_8():
    from src.embedding import Embedding
    from src.cost_learner import NetTrainer
    data_x, data_y = Embedding().flow()

    nt = NetTrainer(
        data_x,
        data_y,
        epochs=10
    )
    nt.flow(plot=True)


def test_9():
    """

    """
    from src.hint_generate import CostEstimation
    from src.query_plan import QueryPlan, QueryPlanNode
    # for i in range(100):
    #     ce = CostEstimation.load_from(i)
    #     print(ce.query_plans)
    # for i in range(100):
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


def test_10():
    *_, res_ratio, p = CostLearner.get_width_of_confidence_intervals()
    print(p)
    for key, value in res_ratio:
        print(key, value, sep=",")
    rich.print(res_ratio)


def test_11():
    from src.hint_generate import SQLWithHintsGenerator

    SQLWithHintsGenerator.test("data/output_2")


if __name__ == '__main__':
    # test_11()
    # test_10()
    test_9()
