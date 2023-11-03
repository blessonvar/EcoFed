# Data_Generator ABC
from abc import ABC, abstractmethod

class Data_Generator(ABC):
    
    @abstractmethod
    def get_local_dataloader(self, *args, **kwargs):
        pass
