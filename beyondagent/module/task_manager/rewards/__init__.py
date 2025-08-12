import functools
from typing import Tuple, Type, Dict, Callable, Any

import functools
from dataclasses import dataclass
from typing import Type, Dict, Callable, Any, Optional


@dataclass
class _RegEntry:
    """
    注册条目：
    - cls: 注册的计算器类
    - singleton: 是否按全局单例返回
    - instance: 当 singleton=True 时缓存的单例实例
    """
    cls: Type
    singleton: bool = False
    instance: Optional[Any] = None

class RewardCalculatorManager:
    """
    一个单例类，用于管理和实例化不同的奖励计算器。

    该类维护一个从名称到计算器类的注册表，并提供一个装饰器
    用于自动注册，以及一个工厂方法用于根据名称获取实例。
    支持在注册时声明某个计算器为“全局单例”。
    """
    _instance = None
    _registry: Dict[str, _RegEntry] = {}

    def __new__(cls, *args, **kwargs):
        """
        实现单例模式。如果实例不存在，则创建一个新实例；否则，返回现有实例。
        """
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    def reg(self, name: str) -> Callable:
        """
        注册装饰器。将一个类与一个给定的名称关联起来。
        可通过 singleton=True 将该计算器以全局单例方式提供。

        用法:
            @calculator_manager.reg("my_calc")  # 普通（每次新建实例）
            class MyCalc: ...

            @calculator_manager.reg("my_singleton_calc", singleton=True)  # 全局单例
            class MySingletonCalc: ...

        参数:
            name: 注册名
            singleton: 是否作为全局单例提供实例
        """
        def decorator(calculator_cls: Type) -> Type:
            if name in self._registry:
                print(f"警告：名称 '{name}' 已被注册，将被新的类 '{calculator_cls.__name__}' 覆盖。")
            # 覆盖注册时，若此前有单例实例，会被丢弃并以新类为准
            self._registry[name] = _RegEntry(cls=calculator_cls, singleton=False, instance=None)
            return calculator_cls
        return decorator

    def get_calculator(self, name: str, *args, **kwargs) -> Any:
        """
        工厂方法。根据注册的名称获取一个计算器类的实例。

        - 对于普通注册（singleton=False），每次返回一个新的实例。
        - 对于单例注册（singleton=True），首次调用时用提供的参数创建实例并缓存，之后始终返回同一个实例，
          再次调用传入的参数将被忽略。

        :param name: 注册时使用的字符串名称。
        :param args: 传递给计算器类构造函数的定位参数。
        :param kwargs: 传递给计算器类构造函数的关键字参数。
        :return: 对应计算器类的实例（可能是新建，也可能是全局单例）。
        :raises ValueError: 如果提供的名称没有被注册。
        """
        entry = self._registry.get(name)
        if not entry:
            raise ValueError(f"错误：没有找到名为 '{name}' 的奖励计算器。可用名称: {list(self._registry.keys())}")

        if entry.singleton:
            if entry.instance is None:
                # 首次创建单例实例并缓存
                entry.instance = entry.cls(*args, **kwargs)
            return entry.instance

        # 非单例：每次返回新实例
        return entry.cls(*args, **kwargs)
# ----------------------------------------------------------------------------
# 示例用法
# ----------------------------------------------------------------------------

# 1. 创建管理器单例。在整个应用程序中，您都应该使用这同一个实例。
grader_manager = RewardCalculatorManager()


from .judge_with_gt import LlmAsJudgeRewardCalculatorWithGT
from .reward import LlmAsJudgeRewardCalculator
from .binary_judge import LlmAsJudgeBinaryRewardCalculator
from .binary_judge_gt import LlmAsJudgeBinaryRewardCalculatorWithGT
from .env_grader import EnvGrader

__all__=[
    "LlmAsJudgeRewardCalculatorWithGT",
    "LlmAsJudgeRewardCalculator",
    "LlmAsJudgeBinaryRewardCalculator",
    "LlmAsJudgeBinaryRewardCalculatorWithGT",
    "EnvGrader",
    "grader_manager"
]