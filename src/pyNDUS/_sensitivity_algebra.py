"""Algebra for multi-group sensitivity profiles."""

from collections import OrderedDict
from copy import copy
from itertools import count
import operator

import numpy as np


class SensitivityAlgebraError(ValueError):
    """Raised when sensitivity objects cannot be combined."""


class SensitivityAlgebraMixin:
    """Algebraic operations shared by :class:`Sensitivity` objects."""

    algebra_policy = "raise"
    _source_ids = count()
    _AXES = ("responses", "materials", "zaid", "MTs")

    def with_algebra_policy(self, policy):
        """Return a shallow copy whose binary operations use *policy*."""
        components = self._uncertainty_components()
        result = copy(self)
        result.algebra_policy = self._validate_algebra_policy(policy)
        result._algebra_uncertainty_components = {
            source: component.copy()
            for source, component in components.items()
        }
        result._algebra_provenance_average = np.array(self.sens, copy=True)
        return result

    def combine(self, other, operation="add", policy=None):
        """Combine two sensitivities using an explicit metadata policy.

        Parameters
        ----------
        other : Sensitivity
            The second sensitivity object.
        operation : {"add", "subtract", "multiply", "divide"}
            ``multiply`` and ``divide`` refer to the underlying responses,
            hence are aliases for sensitivity addition and subtraction.
        policy : {"raise", "intersect", "zero"}, optional
            ``raise`` requires identical response, material, ZAID and MT
            sets. ``intersect`` keeps only entries common to both objects.
            ``zero`` keeps their union and fills entries missing from either
            object with zero average and zero standard deviation.
            Energy-group boundaries must always be identical.
        """
        operations = {
            "add": operator.add,
            "+": operator.add,
            "subtract": operator.sub,
            "sub": operator.sub,
            "-": operator.sub,
            "multiply": operator.add,
            "mul": operator.add,
            "*": operator.add,
            "divide": operator.sub,
            "div": operator.sub,
            "/": operator.sub,
        }
        try:
            function = operations[operation]
        except (KeyError, TypeError):
            allowed = "add, subtract, multiply, divide"
            raise ValueError(f"Unknown operation {operation!r}; expected one of {allowed}.") from None
        return self._combine_sensitivity(other, function, policy)

    def __add__(self, other):
        if np.isscalar(other):
            return NotImplemented
        return self._combine_sensitivity(other, operator.add)

    def __sub__(self, other):
        if np.isscalar(other):
            return NotImplemented
        return self._combine_sensitivity(other, operator.sub)

    def __mul__(self, other):
        if np.isscalar(other):
            return self._scale(other)
        return self._combine_sensitivity(other, operator.add)

    def __rmul__(self, other):
        if np.isscalar(other):
            return self._scale(other)
        return NotImplemented

    def __truediv__(self, other):
        if np.isscalar(other):
            if other == 0:
                raise ZeroDivisionError("Cannot divide a Sensitivity by zero.")
            return self._scale(1 / other)
        return self._combine_sensitivity(other, operator.sub)

    def __pow__(self, exponent):
        if not np.isscalar(exponent):
            return NotImplemented
        return self._scale(exponent)

    def __neg__(self):
        return self._scale(-1)

    @classmethod
    def _validate_algebra_policy(cls, policy):
        if policy not in {"raise", "intersect", "zero"}:
            raise ValueError(
                f"Unknown sensitivity algebra policy {policy!r}; "
                "expected 'raise', 'intersect', or 'zero'."
            )
        return policy

    @staticmethod
    def _metadata_keys(sensitivity, name):
        value = getattr(sensitivity, name)
        if name == "responses":
            return tuple(value)
        return tuple(value.keys())

    def _selection(self, other, policy):
        if not isinstance(other, SensitivityAlgebraMixin):
            return NotImplemented

        policy = self._validate_algebra_policy(policy or self.algebra_policy)
        left_grid = np.asarray(self.group_structure)
        right_grid = np.asarray(other.group_structure)
        if left_grid.shape != right_grid.shape or not np.array_equal(left_grid, right_grid):
            raise SensitivityAlgebraError(
                "Sensitivity group structures differ; energy groups cannot be aligned."
            )

        selected = {}
        indexes = {}
        mismatches = []
        for name in self._AXES:
            left = self._metadata_keys(self, name)
            right = self._metadata_keys(other, name)
            if policy == "raise":
                if set(left) != set(right):
                    only_left = tuple(item for item in left if item not in set(right))
                    only_right = tuple(item for item in right if item not in set(left))
                    mismatches.append(
                        f"{name}: only left={only_left}, only right={only_right}"
                    )
                common = left
            elif policy == "intersect":
                right_set = set(right)
                common = tuple(item for item in left if item in right_set)
                if not common:
                    raise SensitivityAlgebraError(
                        f"No common {name} available with policy='intersect'."
                    )
            else:
                left_set = set(left)
                common = left + tuple(item for item in right if item not in left_set)

            selected[name] = common
            left_index = {item: index for index, item in enumerate(left)}
            right_index = {item: index for index, item in enumerate(right)}
            indexes[name] = (
                [left_index.get(item) for item in common],
                [right_index.get(item) for item in common],
            )

        if mismatches:
            details = "; ".join(mismatches)
            raise SensitivityAlgebraError(
                "Sensitivity metadata differ with policy='raise': " + details
            )
        return selected, indexes, policy

    @staticmethod
    def _slice(array, indexes, side):
        if array is None:
            return None
        array = np.asarray(array)
        aligned_shape = tuple(
            len(indexes[name][side]) for name in SensitivityAlgebraMixin._AXES
        ) + (array.shape[4],)
        aligned = np.zeros(aligned_shape, dtype=array.dtype)

        target_indexes = []
        source_indexes = []
        for name in SensitivityAlgebraMixin._AXES:
            mapping = indexes[name][side]
            target_indexes.append(
                [target for target, source in enumerate(mapping) if source is not None]
            )
            source_indexes.append([source for source in mapping if source is not None])

        if all(target_indexes):
            groups = range(array.shape[4])
            aligned[np.ix_(*target_indexes, groups)] = array[
                np.ix_(*source_indexes, groups)
            ]
        return aligned

    def _uncertainty_components(self):
        current_average = np.asarray(self.sens)
        provenance_average = getattr(self, "_algebra_provenance_average", None)
        components = getattr(self, "_algebra_uncertainty_components", None)
        provenance_is_current = (
            provenance_average is not None
            and provenance_average.shape == current_average.shape
            and np.array_equal(provenance_average, current_average)
        )
        if components is not None and provenance_is_current:
            return components
        if not provenance_is_current:
            self._algebra_source_id = next(self._source_ids)
        if self.sens_rsd is None:
            return {}
        source_id = getattr(self, "_algebra_source_id", None)
        if source_id is None:
            source_id = next(self._source_ids)
            self._algebra_source_id = source_id
        self._algebra_provenance_average = np.array(current_average, copy=True)
        absolute_std = np.abs(current_average) * np.asarray(self.sens_rsd)
        return {source_id: absolute_std}

    @staticmethod
    def _rsd_from_components(average, components):
        if not components:
            return None
        variance = np.zeros_like(average, dtype=float)
        for component in components.values():
            variance += np.square(component)
        absolute_std = np.sqrt(variance)
        result = np.zeros_like(absolute_std)
        np.divide(absolute_std, np.abs(average), out=result, where=average != 0)
        result[(average == 0) & (absolute_std != 0)] = np.inf
        return result

    def _new_result(self, average, selected=None, other=None):
        result = copy(self)
        self._store_result_attribute(result, "sens", np.array(average, copy=True))
        if selected is not None:
            self._store_result_attribute(result, "responses", tuple(selected["responses"]))
            self._store_result_attribute(result, "materials", OrderedDict(
                (value, index) for index, value in enumerate(selected["materials"])
            ))
            self._store_result_attribute(result, "zaid", OrderedDict(
                (value, index) for index, value in enumerate(selected["zaid"])
            ))
            left_zais = tuple(self.zais.keys())
            left_zaids = tuple(self.zaid.keys())
            zai_by_zaid = dict(zip(left_zaids, left_zais))
            if other is not None:
                right_zais = tuple(other.zais.keys())
                right_zaids = tuple(other.zaid.keys())
                for zaid, zais in zip(right_zaids, right_zais):
                    zai_by_zaid.setdefault(zaid, zais)
            self._store_result_attribute(result, "zais", OrderedDict(
                (zai_by_zaid.get(value, value), index)
                for index, value in enumerate(selected["zaid"])
            ))
            self._store_result_attribute(result, "MTs", OrderedDict(
                (value, index) for index, value in enumerate(selected["MTs"])
            ))
        return result

    @staticmethod
    def _store_result_attribute(result, name, value):
        """Bypass parser-oriented setters while supporting simple test doubles."""
        descriptor = getattr(type(result), name, None)
        if isinstance(descriptor, property):
            setattr(result, f"_{name}", value)
        else:
            setattr(result, name, value)

    def _scale(self, factor):
        if not np.isscalar(factor):
            return NotImplemented
        factor = float(factor)
        average = np.asarray(self.sens) * factor
        components = {
            source: component * factor
            for source, component in self._uncertainty_components().items()
        }
        result = self._new_result(average)
        result._algebra_uncertainty_components = components
        result._algebra_provenance_average = np.array(average, copy=True)
        self._store_result_attribute(result, "sens_rsd", self._rsd_from_components(average, components))
        result.algebra_policy = self.algebra_policy
        return result

    def _combine_sensitivity(self, other, function, policy=None):
        selection = self._selection(other, policy)
        if selection is NotImplemented:
            return NotImplemented
        selected, indexes, selected_policy = selection

        left_average = self._slice(self.sens, indexes, 0)
        right_average = self._slice(other.sens, indexes, 1)
        average = function(left_average, right_average)

        components = {}
        for source, component in self._uncertainty_components().items():
            components[source] = self._slice(component, indexes, 0)
        sign = 1 if function is operator.add else -1
        for source, component in other._uncertainty_components().items():
            aligned = self._slice(component, indexes, 1) * sign
            if source in components:
                components[source] = components[source] + aligned
            else:
                components[source] = aligned

        result = self._new_result(average, selected, other)
        result._algebra_uncertainty_components = components
        result._algebra_provenance_average = np.array(average, copy=True)
        self._store_result_attribute(result, "sens_rsd", self._rsd_from_components(average, components))
        result.algebra_policy = selected_policy
        return result
