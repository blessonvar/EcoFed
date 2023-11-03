# Client_Sampler ABC
from abc import ABC, abstractmethod

class Client_Sampler(ABC):
    
    @abstractmethod
    def get_samplers(self, *args, **kwargs):
        pass
