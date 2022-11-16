"""
    basic functions and class implements
"""
import logging
import os
import sys
import time
import typing
import rich
import tqdm

from src.config import PG_STATEMENT_TIMEOUT, DATA_PATH_PLANS_FOR_TRAIN, TEST_DB_CONFIG, DATA_PATH_LOGS
import rich.logging

import psycopg2


# -*-*-*-*-*-*-*-*-*-*-*-*-*-日 志 定 义-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*- #
class GeneralLogger:
    """
        解决多进程不同同时写入的问题
    """

    def __init__(self, level=logging.DEBUG, name='default',
                 log_format="""%(asctime)s [P-%(process)d] [%(threadName)s] [%(name)s] [%(levelname)s] """ +
                            f"""%(filename)s[line:%(lineno)d] %(message)s""",
                 date_format='%Y-%m-%d %H:%M:%S',
                 stdout_flag=False, stdout_level=logging.WARNING, file_mode="a", *,
                 use_rich=False, rich_console=None):
        """
            支持多线程, 不支持多进程
        """
        self._log_format = log_format
        self._date_format = date_format
        self._log_level = level
        self._log_name = name
        self._log_dir = os.path.join(DATA_PATH_LOGS, f"{name}-{time.strftime('%Y%m%d')}")
        self._log_path = os.path.join(self._log_dir, f"{name}-p-{os.getpid()}.log")
        self._error_log_path = os.path.join(self._log_dir, f"{name}-error-p-{os.getpid()}.log")
        if not os.path.exists(self._log_dir):
            # # 这里必须确保该目录下没有和文件夹同名的文件名称
            os.mkdir(self._log_dir)
        # if not os.path.exists(self._log_path):
        # # log_path detected to avoid the logger chaos
        # warnings.warn(f"Warning: Path to record logging doesn't exist: {self._log_path}")
        # if not os.path.exists(self._error_log_path):
        # # log_path detected to avoid the logger chaos
        # warnings.warn(f"Warning: Path to record logging doesn't exist: {self._error_log_path}")
        self._log = logging.getLogger(self._log_name)
        self._log.setLevel(self._log_level)
        # # create file handler
        self._file_handler = logging.FileHandler(self._log_path, mode=file_mode, encoding="utf-8")
        self._file_handler.setLevel(self._log_level)
        self._file_handler.setFormatter(logging.Formatter(self._log_format, self._date_format))
        self._log.addHandler(self._file_handler)
        # # 错误日志 单独记录
        self._error_file_handler = logging.FileHandler(self._error_log_path, mode=file_mode, encoding="utf-8")
        self._error_file_handler.setLevel(logging.ERROR)
        self._error_file_handler.setFormatter(logging.Formatter(self._log_format, self._date_format))
        self._log.addHandler(self._error_file_handler)
        # # 控制台日志
        if stdout_flag:
            if use_rich:
                self._std_handler = rich.logging.RichHandler(stdout_level)
                self._rich_console = rich_console if rich_console else rich.console.Console()
                self._std_handler.setFormatter(
                    logging.Formatter("[P-%(process)d] [%(threadName)s] [%(name)s] %(message)s"))
            else:
                self._std_handler = logging.StreamHandler(sys.stdout)
                self._std_handler.setLevel(stdout_level)
                self._std_handler.setFormatter(logging.Formatter(self._log_format, self._date_format))
            self._log.addHandler(self._std_handler)
        self.info = self._log.info
        self.debug = self._log.debug
        self.error = self._log.error
        self.warning = self._log.warning
        self.critical = self._log.critical
        self.exception = self._log.exception

    def clear(self):
        with open(self._log_path, "w") as f:
            f.seek(0)
            f.truncate()

    def clear_error_log(self):
        with open(self._error_log_path, "w") as f:
            f.seek(0)
            f.truncate()

    def __call__(self, *args, **kwargs):
        return self.info(*args, **kwargs)


BasicLogger = GeneralLogger(name="basic", stdout_flag=True, stdout_level=logging.WARNING,
                            file_mode="a")


class PostgresDB:
    def __init__(self, _host="127.0.0.1",
                 _user="postgres",
                 _db_name="postgres",
                 _passwd="",
                 _config: dict = None,
                 _try_connect=True,
                 _load_hint_plan=True):
        self._host = _host
        self._user = _user
        self._passwd = _passwd
        self._db = _db_name
        self._conn = None
        if _config is not None:
            assert isinstance(_config, dict), "config error"
            self._host = _config.get("host", _host)
            self._user = _config.get("user", _user)
            self._passwd = _config.get("passwd", _passwd)
            self._db = _config.get("db", _db_name)
        if _try_connect:
            self.connect(_load_hint_plan)

    def connect(self, _load_hint_plan=True, _timeout=PG_STATEMENT_TIMEOUT):
        self._conn = psycopg2.connect(dbname=self._db, user=self._user, host=self._host)
        if _load_hint_plan:
            # # 判断是否连接成功
            assert self._conn is not None, "database not connect"

            _cur = self._conn.cursor()
            # # ... 设置本次Session上SQL语句的执行超时时间
            _cur.execute(f"SET statement_timeout = {_timeout};")
            # # ... 加载 pg_hint_plan, 否则之后的过程无法生效
            _cur.execute("LOAD 'pg_hint_plan';")

    def execute(self, _sql="") -> typing.Tuple[typing.Tuple[typing.Any]]:
        assert self._conn is not None, "database not connect"
        try:
            _cur = self._conn.cursor()

            _cur.execute(_sql)
            _res = _cur.fetchall()

            return tuple(_res)
        except Exception as e:
            BasicLogger.exception(f"failed to execute sql, exception {e.__doc__}, sql: {_sql}")

    def get_query_plan(self, _query_sql: str) -> typing.Tuple[typing.Tuple[str]]:
        """
            获取执行计划
        Args:
            _query_sql: 应该是一个SQL QUERY
        Returns: Plan
        """
        assert not (_query_sql.startswith("EXPLAIN") or _query_sql.startswith("explain")), "sql starts with 'analyse'"
        return self.execute("EXPLAIN ANALYSE {}".format(_query_sql))

    @staticmethod
    def test_get_query_plan(_sql_with_hints_path=DATA_PATH_PLANS_FOR_TRAIN,
                            _when_canceling: typing.Callable[
                                [int, str, typing.Iterable, "PostgresDB"], typing.Iterable] = None):
        """
            测试: 获取查询计划
        :param _sql_with_hints_path:
        :return:
        """
        _db = PostgresDB(_config=TEST_DB_CONFIG, _try_connect=True)
        with open(_sql_with_hints_path, "r") as _f:
            for _idx, _query_sql_with_hints in tqdm.tqdm(enumerate(_f.readlines())):
                _plan = _db.get_query_plan(_query_sql_with_hints)
                BasicLogger.debug("\n".join(map(lambda _items: _items[0], _plan)))
                # # 直接记录到日志中
                if "canceling" in _plan[0]:
                    _new_plan = _when_canceling(_idx, _query_sql_with_hints, _plan, _db)
                    BasicLogger.debug("canceling, new plan as following: ")
                    BasicLogger.debug("\n".join(map(lambda _items: _items[0], _plan)))

    @staticmethod
    def limit_callback(_idx: int, _query_sql_with_hints: str, _plan: typing.Iterable,
                       _db: "PostgresDB") -> typing.Iterable:
        return _db.get_query_plan(_query_sql_with_hints.replace("explain analyse", "explain"))
