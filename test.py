"""
    测试
"""
import functools
import json
import os

from src.basic import PostgresDB
from src.config import DB_LAB_VM_CONFIG


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


if __name__ == '__main__':
    test_4()
