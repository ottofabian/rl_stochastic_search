from abc import abstractmethod, ABC

from models.policy.abstract_policy import AbstractPolicy


class AbstractPolicyUpdate(ABC):

    def __init__(self, policy: AbstractPolicy):
        self.policy = policy

    @abstractmethod
    def sample(self, mean, std, n, context=None, **kwargs):
        """
        Generate some samples

        Args:
            mean:
            std:
            n:
            context:
            **kwargs:

        Returns:

        """
        pass

    @abstractmethod
    def learn(self, num_iter: int, **kwargs):
        """
        Execute full training run for a given amount of iterations
        Args:
            num_iter: number of iterations
            **kwargs:

        Returns:

        """
        pass

    @abstractmethod
    def step(self, policy_new: AbstractPolicy, sample_dict: dict = {}):
        """
        Excecute a single step of the algorithm
        Args:
            policy_new:
            sample_dict:

        Returns:

        """
        pass
