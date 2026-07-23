"""Public package interface for pyNDUS.

The public classes are loaded lazily so modules that do not use covariance
processing can be imported without the optional NJOY/SANDY runtime available.
"""

__all__ = [
            "Covariance", "CovarianceError", "Sensitivity", "SensitivityError",
            "SensitivityAlgebraError", "Sandwich", "SandwichError",
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
    if name == "Sandwich":
        from .sandwich import Sandwich

        return Sandwich
    if name == "SandwichError":
        from .sandwich import SandwichError

        return SandwichError
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
