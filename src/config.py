"""
    stores some configs
"""
import os.path

PROJECTS_ROOT = r""
SRC_PATH = os.path.join(PROJECTS_ROOT, "src")
TEMP_PATH = os.path.join(PROJECTS_ROOT, "tmp")
THIRD_PARTY_PATH = os.path.join(PROJECTS_ROOT, "third_party")
# # DATA
DATA_PATH = os.path.join(PROJECTS_ROOT, "data")
DATA_PATH_LOGS = os.path.join(DATA_PATH, "log")
DATA_PATH_TXT = os.path.join(DATA_PATH, "txt")
DATA_PATH_PKL = os.path.join(DATA_PATH, "pkl")
DATA_PATH_NPY = os.path.join(DATA_PATH, "npy")
DATA_PATH_PIC = os.path.join(DATA_PATH, "pic")
DATA_PATH_PLANS_FOR_TRAIN = os.path.join(DATA_PATH, "plan")  # # 用于训练的数据
DATA_PATH_MODEL = os.path.join(DATA_PATH, "model")  # # 模型
DATA_PATH_OUTPUT = os.path.join(DATA_PATH, "output")  # # 得到的最好的查询计划以及估计
# # 第三方
JOB_SQL_FOR_PLANS = os.path.join(THIRD_PARTY_PATH, "join-order-benchmark")  # # 用这里的SQL生成查询计划
# # 用于测试的数据
LC_SQL_CSV_FOR_TEST = os.path.join(THIRD_PARTY_PATH, "learnedcardinalities")
DATA_PATH_LC_SQL_TRAIN_CSV = os.path.join(LC_SQL_CSV_FOR_TEST, "data", "train.csv")

DB_CONFIGS = {
    "test": {
        "host": "127.0.0.1",
        "user": "postgres",
        "db": "postgres"
    },
    "db_lab_vm_bing": {
        "host": "192.168.72.128",
        "user": "postgres",
        "db": "imdbload"
    }
}
TEST_DB_CONFIG = DB_CONFIGS["test"]
DB_LAB_VM_CONFIG = DB_CONFIGS["db_lab_vm_bing"]

# # STATEMENT TIMEOUT, PG执行的语句超时时间, 单位是ms
PG_STATEMENT_TIMEOUT = 60000
