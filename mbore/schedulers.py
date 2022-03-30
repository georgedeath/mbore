class BaseScheduler:
    def __init__(self, gamma_start: float, gamma_end: float, total_steps: int):
        self.gamma_start = gamma_start
        self.gamma_end = gamma_end
        self.T = total_steps

    def __call__(self, t) -> float:
        raise NotImplementedError


class FixedScheduler(BaseScheduler):
    def __init__(self, gamma_start: float, gamma_end: float, total_steps: int):
        super(FixedScheduler, self).__init__(
            gamma_start, gamma_end, total_steps
        )

    def __call__(self, t: int) -> float:
        return self.gamma_start


class LinearScheduler(BaseScheduler):
    def __init__(self, gamma_start: float, gamma_end: float, total_steps: int):
        super(LinearScheduler, self).__init__(
            gamma_start, gamma_end, total_steps
        )

    def __call__(self, t: int) -> float:
        alpha = t / self.T
        gamma = self.gamma_end * alpha + self.gamma_start * (1.0 - alpha)
        return gamma
