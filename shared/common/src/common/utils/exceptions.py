class SubmittedWeightsError(Exception):
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


class LayerStateException(Exception):
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


class RunFullException(Exception):
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


class MinerNotRegisteredException(Exception):
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


class APIException(Exception):
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


class NanInfWarning(Exception):
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


class NanInfException(Exception):
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


class SpecVersionException(Exception):
    def __init__(self, expected_version: int, actual_version: str):
        self.expected_version = expected_version
        self.actual_version = actual_version
        self.message = f"Spec version mismatch. Expected: {expected_version}, Received: {actual_version}"
        super().__init__(self.message)


class WeightPartitionException(Exception):
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


class RateLimitException(Exception):
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


class MinerBlockedException(Exception):
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


class MinerFrozenException(MinerBlockedException):
    """Raised when orchestrator reports miner is frozen."""


class MinerInitializingException(MinerBlockedException):
    """Raised when orchestrator reports miner is still initializing."""
