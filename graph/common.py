# 自定义异常类
class InvalidInputError(Exception):
    """
    自定义异常类，用于表示无效的输入格式
    """

    def __init__(self, message: str, error_code=400):
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)
