# coding=utf-8
'''
import abc
 @abc.abstractmethod #定义抽象方法，无需实现功能
    def read(self):
        '子类必须定义读功能'
        pass
'''
class Preprocessor:

    def __init__(self, path, file_suffix):
        self.path = path
        self.train_file = path + "train" + "." + file_suffix
        self.test_file = path + "test" + "." + file_suffix
        self.dev_file = path + "dev" + "." + file_suffix

    # 兼容接口
    def split_file(self, file_name, file_suffix):
        pass
