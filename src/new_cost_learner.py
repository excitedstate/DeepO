import logging

from src.basic import GeneralLogger

CostLearnerLogger = GeneralLogger(name="embedding", stdout_flag=True, stdout_level=logging.INFO,
                                  file_mode="a")
