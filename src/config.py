"""
    stores some configs
"""
import os.path

PROJECTS_ROOT = r""
SRC_PATH = os.path.join(PROJECTS_ROOT, "src")
TEMP_PATH = os.path.join(PROJECTS_ROOT, "tmp")
# # DATA相关
DATA_PATH = os.path.join(PROJECTS_ROOT, "data")
DATA_PATH_LOGS = os.path.join(DATA_PATH, "logs")
# # SQL和查询计划
DATA_PATH_GENE_SQL = os.path.join(DATA_PATH, "generated_sql_queries")
DATA_PATH_GOT_PLANS = os.path.join(DATA_PATH, "generated_plans")
DATA_PATH_GENE_SQL_WITH_HINTS = os.path.join(DATA_PATH, "generated_sql_queries_with_hint")
# # 这个路径是用来做测试的
DATA_PATH_SQL_FOR_PLANS = os.path.join(DATA_PATH, "join-order-benchmark")
DATA_PATH_PLANS_FOR_TRAIN = os.path.join(DATA_PATH, "plans_for_train")
# # learnedcardinalities
DATA_PATH_LC = os.path.join(DATA_PATH, "learnedcardinalities", "data")
DATA_PATH_LC_COLUMN_MIN_MAX_VALS_CSV = os.path.join(DATA_PATH_LC, "column_min_max_vals.csv")
DATA_PATH_LC_SQL_TRAIN_CSV = os.path.join(DATA_PATH_LC, "train.csv")
# # --- MODEL MATH
MODEL_PATH = os.path.join(PROJECTS_ROOT, "model")

MODEL_PATH_COST_MODEL_PREFIX = os.path.join(MODEL_PATH, "cost_model_new")
MODEL_PATH_EMBEDDING_MODEL = os.path.join(MODEL_PATH, "embedding_model.h5")

# # 最大值最小值预处理的结果是需要保存的, 这个过程应该是十分漫长的
MODEL_PATH_STD_SCALER = os.path.join(MODEL_PATH, "std_scaler.bin")

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
