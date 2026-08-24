"""Public package interface for pyNDUS."""

from .sandwich import Sandwich, SandwichError

__all__ = [
            "Covariance", "CovarianceError", "Sensitivity", "SensitivityError",
            "SensitivityAlgebraError", "SensitivityChannel", "Sandwich",
            "SandwichError",
            ]


def __getattr__(name):
    """Load public classes on first access."""
    if name == "Covariance":
        from .covariance import Covariance
        return Covariance

    if name == "CovarianceError":
        from .covariance import CovarianceError
        return CovarianceError

    if name == "Sensitivity":
        from .sensitivity import Sensitivity
        return Sensitivity

    if name == "SensitivityError":
        from .sensitivity import SensitivityError
        return SensitivityError

    if name == "SensitivityAlgebraError":
        from ._sensitivity_algebra import SensitivityAlgebraError
        return SensitivityAlgebraError

    if name == "SensitivityChannel":
        from .channels import SensitivityChannel
        return SensitivityChannel

    if name == "Sandwich":
        return Sandwich

    if name == "SandwichError":
        return SandwichError

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
