from __future__ import annotations

import warnings
from collections.abc import Sequence, Sized
from typing import TYPE_CHECKING, Any, Literal, NotRequired, TypedDict

import numpy as np
from numpy.typing import NDArray

from ._chemfiles import ELEMENTS


if TYPE_CHECKING:
    from metatensor import TensorMap
    from metatomic import System


Environment = tuple[int, int, float]


class _Property(TypedDict):
    target: Literal["atom", "structure"]
    values: NDArray[np.floating[Any]]
    units: NotRequired[str]
    description: NotRequired[str]


def _as_numpy(array: Any) -> NDArray[Any]:
    if hasattr(array, "detach"):
        return array.detach().cpu().numpy()
    if hasattr(array, "__array__"):
        return np.asarray(array)
    return np.from_dlpack(array)


def _metatomic_valid_structures(structures: object) -> tuple[object, bool]:
    try:
        from metatomic import System
    except ImportError:
        return structures, False

    if isinstance(structures, System):
        return [structures], True
    if isinstance(structures, Sequence) and structures:
        if isinstance(structures[0], System):
            if not all(isinstance(s, System) for s in structures):
                raise TypeError("all structures must be metatomic.System objects")
            return structures, True
    return structures, False


def _metatomic_to_json(system: System) -> dict[str, Any]:
    from metatomic.utils import unit_conversion_factor

    if not system.length_unit:
        raise ValueError("a metatomic System must have a length unit for visualization")
    factor = unit_conversion_factor(system.length_unit, "angstrom")
    positions = _as_numpy(system.positions) * factor
    types = _as_numpy(system.types)
    data = {
        "size": len(system),
        "names": [ELEMENTS[t] if 0 <= t < len(ELEMENTS) else str(t) for t in types],
        "elements": [ELEMENTS[t] if 0 <= t < len(ELEMENTS) else "X" for t in types],
        "x": positions[:, 0].tolist(),
        "y": positions[:, 1].tolist(),
        "z": positions[:, 2].tolist(),
    }
    cell = _as_numpy(system.cell) * factor
    if np.all(np.linalg.norm(cell, axis=1) != 0):
        data["cell"] = cell.reshape(-1).tolist()
    return data


def _metatomic_all_atomic_environments(
    structures: Sequence[Sized], cutoff: float
) -> list[Environment]:
    return [(i, a, cutoff) for i, s in enumerate(structures) for a in range(len(s))]


def metatensor_to_property(
    tensor: TensorMap,
    structures: Sequence[Sized],
    *,
    environments: Sequence[Environment] | None = None,
) -> _Property:
    """Convert a single-block TensorMap into a Chemiscope property.

    Samples must be labeled ``system`` or ``system, atom``. Values are reordered
    to match ``structures`` or the supplied ``environments`` (including repeats).
    Components and properties are flattened in their stored order. All requested
    samples must exist; extra samples are ignored. Units/description can be added
    to the returned dictionary by the caller. Gradients must be extracted explicitly.
    """
    if len(tensor) != 1:
        raise ValueError("expected a TensorMap with exactly one block")
    block = tensor.block(0)
    if block.gradients_list():
        raise ValueError("extract TensorMap gradients explicitly before conversion")
    names = block.samples.names
    target: Literal["atom", "structure"]
    samples: list[tuple[int, ...]]
    if names == ["system"]:
        target = "structure"
        samples = [(i,) for i in range(len(structures))]
    elif names == ["system", "atom"]:
        target = "atom"
        if environments is None:
            environments = _metatomic_all_atomic_environments(structures, 0)
        samples = [(i, a) for i, a, _ in environments]
        for i, a in samples:
            if not 0 <= i < len(structures) or not 0 <= a < len(structures[i]):
                raise ValueError(f"invalid environment ({i}, {a})")
    else:
        raise ValueError("TensorMap samples must be 'system' or 'system, atom'")
    rows = {tuple(row): i for i, row in enumerate(_as_numpy(block.samples.values))}
    try:
        indices = [rows[sample] for sample in samples]
    except KeyError as error:
        raise ValueError(f"missing TensorMap sample {error.args[0]}") from error
    values = _as_numpy(block.values)
    width = int(np.prod(values.shape[1:]))
    values = values.reshape(len(values), width)[indices]
    return {"target": target, "values": values}


def _metatomic_extract_properties(
    structures: Sequence[System],
    only: Sequence[str] | None,
    environments: Sequence[Environment] | None,
) -> dict[str, _Property]:
    custom_names = [set(s.known_custom_data()) for s in structures]
    names = set.intersection(*custom_names)
    all_names = set.union(*custom_names)
    if names != all_names:
        warnings.warn(
            "only custom data present in every System is extracted", stacklevel=3
        )
    properties: dict[str, _Property] = {}
    for name in sorted(names if only is None else names.intersection(only)):
        tensors = [s.custom_data(name) for s in structures]
        parts = [
            metatensor_to_property(t, [s])
            for t, s in zip(tensors, structures, strict=True)
        ]
        first = tensors[0].block(0)
        components, labels = first.components, first.properties
        for tensor, part in zip(tensors[1:], parts[1:], strict=True):
            block = tensor.block(0)
            if (
                part["target"] != parts[0]["target"]
                or block.components != components
                or block.properties != labels
            ):
                raise ValueError(f"inconsistent custom data layout for '{name}'")
        values = np.concatenate([p["values"] for p in parts])
        target = parts[0]["target"]
        if target == "atom" and environments is not None:
            offsets = np.cumsum([0] + [len(s) for s in structures])
            for i, a, _ in environments:
                if not 0 <= i < len(structures) or not 0 <= a < len(structures[i]):
                    raise ValueError(f"invalid environment ({i}, {a})")
            values = values[[offsets[i] + a for i, a, _ in environments]]
        properties[name] = {"target": target, "values": values}
    return properties
