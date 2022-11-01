"""
    测试
"""

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
    print(res)


if __name__ == '__main__':
    test_2()
