import sys
sys.path.append('D:/WORK/Personnel/Python projects/Projects/used-car-price')
from src.logger import logging

def error_message_detail(error):
    _,_,exc_tb = sys.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = f"Error occured in python script name [{file_name}] line number [{exc_tb.tb_lineno}] error message [{str(error)}]"

    return error_message

class CustomException(Exception):
    def __init__(self, error, error_detail):
        super().__init__(error_message_detail(error))
        self.error_message = error_message_detail(error)

    def __str__(self):
        return self.error_message
        
