# coding=utf-8
from typing import Optional, Callable, Generator

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


def build_gradient_clipper(config: dict) -> Optional[Callable]:
    """
    梯度裁剪
    Define the function for gradient clipping as specified in configuration
    If not specified, return None
    Current optional:
        - clip_grad_val: clip the gradient if they exceed this value
        see `torch.nn.utils.clip_grad_value_`
        - 'clip_grad_norm': clip thr gradient if their norm exceeds this value
        see `torch.nn.utils.clip_grad_norm_`

    :param config: dictionary with training configurations
    :return: clipping function (in-place) or None if no gradient clipping
    """

    clip_grad_fun = None

    # 位于绝对值之间
    if 'clip_grad_val' in config.keys():
        clip_value = config['clip_grad_val']
        clip_grad_fun = lambda params: nn.utils.clip_grad_value_(
            parameters=params, clip_value=clip_value
        )
    elif 'clip_grad_norm' in config.keys():
        max_norm = config['clip_grad_norm']
        clip_grad_fun = lambda params: nn.utils.clip_grad_norm_(
            parameters=params, max_norm=max_norm
        )

    if "clip_grad_val" in config.keys() and "clip_grad_norm" in config.keys():
        raise ConfigurationError(
            "You can only specify either clip_grad_val or clip_grad_norm.")

    return clip_grad_fun


def build_optimizer(config: dict, parameters: Generator) -> Optimizer:
    """
    对计算图的每个参数进行梯度下降，（梯度已经计算好）
    lr：初始学习率
    weight: 权重衰减，1-weight表示原始变量有多少留在新的变量中
    name： 使用哪个计算梯度的公式

    create an optimizer for the given parameters as specified in configuration

    Except for the weight decay and initial learning rate
    default optimizer setting are used

    Current supported configuration setting for 'optimizer':
      - 'sgd' (default): see 'torch.optim.SGD'
      - 'adam': see 'torch.optim.adam'
      - 'adagrad': see 'torch.optim.adagrad'
      - 'adadelta': see 'torch.optim.adadelta'
      - 'rmsprop': see 'torch.optim.RMSprop'

    The initial learning rate is set according to 'learning_rate' in the config.
    the weight decay is set according to 'weight_decay' in the config
    If they are not specified,the initial learning rate is set to 3.0e-4,
    the weight decay to 0

    Note that the scheduler state is saved in the checkpoint, so if you
    load a model for further training you have to use the same type of scheduler

    :param config: configuration dictionary
    :param parameters:
    :return:
    """

    optimizer_name = config.get('optimizer', "sgd").lower()
    learning_rate = config.get('learning_rate', 3.0e-4)
    weight_decay = config.get('weight_decay', 0)

    if optimizer_name == "adam":
        adam_betas = config.get("adam_betas", (0.9, 0.999))
        optimizer = torch.optim.Adam(parameters, weight_decay=weight_decay,
                                     lr=learning_rate, betas=adam_betas)

    elif optimizer_name == 'adagrad':
        optimizer = torch.optim.Adagrad(parameters, weight_decay=weight_decay,
                                        lr=learning_rate)

    elif optimizer_name == 'adadelta':
        optimizer = torch.optim.AdaDelta(parameters, weight_decay=weight_decay,
                                         lr=learning_rate)
    elif optimizer_name == 'rmsprop':
        optimizer = torch.optim.RMSprop(parameters, weight_decay=weight_decay,
                                        lr=learning_rate)
    elif optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(parameters, weight_decay=weight_decay,
                                    lr=learning_rate)
    else:
        raise ConfigurationError("Invalid optimizer. Valid options: 'adam', "
                                 "'adagrad', 'adadelta', 'rmsprop', 'sgd'.")

    return optimizer


def build_scheduler(config: dict, optimizer: Optimizer, scheduler_mode: str, hidden_size: int = 0) -> (
        Optional[_LRScheduler], Optional[str]):
    """
    处理学习率用的，一般学习率都是可变的，随着训练轮数的不同而不同
    create a learning rate scheduler if specified in config and determine when
    a scheduler step should be executed

    Current options:
        - "plateau": see `torch.optim.lr_scheduler.ReduceLROnPlateau`
        - "decaying": see `torch.optim.lr_scheduler.StepLR`
        - "exponential": see `torch.optim.lr_scheduler.ExponentialLR`
        - "noam": see `joeynmt.builders.NoamScheduler`
        - "warmupexponentialdecay": see `joeynmt.builders.WarmupExponentialDecayScheduler`

    If no scheduler is specified, return (None, None) which will result in a
    constant learning rate

    :param config: training configuration
    :param optimizer: optimizer for the scheduler, determines the set of parameters
                    which the scheduler sets the learning rate for

    :param scheduler_mode: "min" or "max", depending on whether the validation score
                    should be minimized or maximized, only relevant for "plateau"
    :param hidden_size: encode hidden size (required for NoamScheduler)
    :return:
        - scheduler: scheduler object
        - scheduler_step_at: either "validation" or 'epoch' or 'step'
    """

    scheduler, scheduler_step_at = None, None

    if "scheduling" in config.keys() and config['scheduling']:
        if config['scheduling'].lower() == 'plateau':
            # learning rate scheduler
            scheduler = ReduceLROnPlateau(optimizer=optimizer, mode=scheduler_mode, verbose=False,
                                          threshold_mode='abs', factor=config.get("decrease_factor", 0.1),
                                          patience=config.get('patience', 10))
            scheduler_step_at = 'validation'
        if config['scheduling'].lower() == 'decaying':
            scheduler = StepLR(optimizer=optimizer, step_size=config.get('decaying_step_size', 1))
            scheduler_step_at = 'epoch'
        if config['scheduling'].lower() == 'exponential':
            scheduler = ExponentialLR(optimizer=optimizer, gamma=config.get('decrease_factor', 0.99))
            scheduler_step_at = 'epoch'
        if config['scheduling'].lower() == 'noam':
            factor = config.get('learning_rate_factor', 1)
            warmup = config.get('learning_rate_warmup', 4000)
            scheduler = NoamScheduler(hidden_size=hidden_size, optimizer=optimizer, factor=factor,
                                      warmup=warmup)
            scheduler_step_at = 'step'
        if config['scheduling'].lower() == 'warmupexponentialdecay':
            min_rate = config.get("learning_rate_min", 1.0e-5)
            decay_rate = config.get("learning_rate_decay", 0.1)
            warmup = config.get("learning_rate_warmup", 4000)
            peak_rate = config.get("learning_rate_peak", 1.0e-3)
            decay_length = config.get("learning_rate_decay_length", 10000)
            scheduler = WarmupExponentialDecayScheduler(
                min_rate=min_rate, decay_rate=decay_rate,
                warmup=warmup, optimizer=optimizer, peak_rate=peak_rate,
                decay_length=decay_length)
            scheduler_step_at = "step"

    return scheduler, scheduler_step_at


class NoamScheduler:
    """
    The Noam learning rate scheduler used in "Attention is all you need"

    """

    def __init__(self, hidden_size: int, optimizer: Optimizer, factor: float = 1,
                 warmup: int = 4000):
        """
        warm-up, followed by learning rate decay

        :param hidden_size:
        :param optimizer:
        :param factor:  decay factor
        :param warmup:  number of warmup steps
        :return:
        """
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.hidden_size = hidden_size
        self._rate = 0

    def step(self):
        """
        update parameters and rate
        :return:
        """
        self._step += 1
        rate = self._compute_rate()
        for p in self.optimizer.param_group:
            p['lr'] = rate
        self._rate = rate

    def _compute_rate(self):
        """
        implement lrate above
        :return:
        """
        step = self._step
        lr = self.factor * (self.hidden_size ** (-0.5)
                            * min(step ** (-0.5), step * self.warmup ** (-1.5)))
        return lr

    def state_dict(self):
        return None


class WarmupExponentialDecayScheduler:
    """
    A learning rate scheduler similar to Noam, but modify:
    keep the warm up period but make it so that the decay rate can bu tuneable.
    the deacy is exponential up to a given minimun rate
    """

    def __init__(self, optimizer: Optimizer, peak_rate: float = 1.0e-3,
                 decay_length: int = 10000, warmup: int = 4000,
                 decay_rate: float = 0.5, min_rate: float = 1.0e-5):
        """
        warm up, followed by exponential lr decay

        :param optimizer:
        :param peak_rate: maximum learning rate at peak after warmup
        :param decay_length: decay length after warmup
        :param warmup: number of warmup steps
        :param decay_rate: decay rate after warmup
        :param min_rate: minimum learning rate
        """
        self.optimizer = optimizer
        self._rate = 0
        self._step = 0
        self.warmup = warmup
        self.decay_length = decay_length
        self.decay_rate = decay_rate
        self.min_rate = min_rate
        self.peak_rate = peak_rate

    def step(self):
        """
        update parameters and rate
        :return:
        """
        self._step += 1

        rate = self._compute_rate()
        for p in self.optimizer.param_group:
            p['lr'] = rate

        self._rate = rate

    def _compute_rate(self):
        """
        implement le above
        :return:
        """
        step = self._step
        warmup = self.warmup

        if self._step < warmup:
            rate = step * self.peak_rate / warmup
        else:
            exponent = (step - warmup) / self.decay_length
            rate = self.peak_rate * (self.decay_rate ** exponent)

        return max(rate, self.min_rate)

    def state_dict(self):
        return None
