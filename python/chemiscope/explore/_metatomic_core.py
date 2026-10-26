from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from ..structures import _guess_adapter, metatensor_to_property
from ..structures._metatomic import Environment, _as_numpy


if TYPE_CHECKING:
    from ase import Atoms
    from metatomic import BaseModel, System


class CoreFeaturizer:
    def __init__(
        self,
        model: BaseModel,
        *,
        variant: str | None,
        device: str | None,
        check_consistency: bool | None,
        length_unit: str,
    ) -> None:
        import metatomic as mta

        self.model = model
        if not isinstance(self.model, mta.BaseModel):
            raise TypeError("expected a metatomic.BaseModel")
        self.capabilities = self.model.capabilities()
        self.name = "feature" if variant is None else f"feature/{variant}"
        if not any(q.name == self.name for q in self.capabilities.outputs):
            raise ValueError(f"model does not declare '{self.name}'")
        if (
            device not in (None, "cpu")
            or "cpu" not in self.capabilities.supported_devices
        ):
            raise ValueError("the core featurizer currently requires a CPU model")
        self.check_consistency = bool(check_consistency)
        self.length_unit = length_unit

    def __call__(
        self,
        structures: Sequence[System] | Sequence[Atoms],
        environments: Sequence[Environment] | None,
    ) -> NDArray[np.floating[Any]]:
        import metatomic as mta
        from metatensor import Labels, TensorBlock
        from metatomic.utils import unit_conversion_factor
        from vesin import NeighborList

        kind = "system" if environments is None else "atom"
        matches = [
            q
            for q in self.capabilities.outputs
            if q.name == self.name and q.sample_kind == kind
        ]
        if len(matches) != 1:
            raise ValueError(
                f"model must declare one '{self.name}' output for '{kind}' samples"
            )
        requested = [
            mta.Quantity(
                name=self.name,
                unit=matches[0].unit,
                sample_kind=kind,
            )
        ]
        inputs = self.model.requested_inputs(requested)
        pairs = self.model.requested_pair_lists(requested)
        structures, adapter = _guess_adapter(structures)
        dtype = np.dtype(self.capabilities.dtype)
        systems = []
        for structure in structures:
            if adapter == "metatomic":
                types, positions, cell, pbc = (
                    _as_numpy(getattr(structure, name))
                    for name in ("types", "positions", "cell", "pbc")
                )
                unit = structure.length_unit
            elif adapter == "ASE":
                types, positions, cell, pbc = (
                    structure.numbers,
                    structure.positions,
                    structure.cell.array,
                    structure.pbc,
                )
                unit = self.length_unit
            else:
                raise TypeError(
                    "the core featurizer accepts metatomic Systems or ASE Atoms"
                )
            if not unit:
                raise ValueError("a system length unit is required")
            factor = unit_conversion_factor(unit, self.capabilities.length_unit)
            positions = np.asarray(positions * factor, dtype=dtype)
            cell = np.asarray(cell * factor, dtype=dtype)
            cell = np.where(np.asarray(pbc)[:, None], cell, 0)
            system = mta.System(
                self.capabilities.length_unit,
                np.array(types, dtype=np.int32),
                positions,
                cell,
                np.array(pbc, dtype=bool),
            )
            for quantity in inputs:
                if (
                    adapter != "metatomic"
                    or quantity.name not in structure.known_custom_data()
                ):
                    raise ValueError(f"missing model input '{quantity.name}'")
                system.add_custom_data(
                    quantity.name, structure.custom_data(quantity.name).copy()
                )
            for options in pairs:
                i, j, shifts, vectors = NeighborList(
                    cutoff=options.cutoff, full_list=options.full_list
                ).compute(positions, cell, pbc, quantities="ijSD")
                system.add_pairs(
                    options,
                    TensorBlock(
                        values=np.asarray(vectors, dtype=dtype).reshape(-1, 3, 1),
                        samples=Labels(
                            [
                                "first_atom",
                                "second_atom",
                                "cell_shift_a",
                                "cell_shift_b",
                                "cell_shift_c",
                            ],
                            np.column_stack((i, j, shifts)).astype(np.int32),
                        ),
                        components=[Labels.range("xyz", 3)],
                        properties=Labels.range("distance", 1),
                    ),
                )
            systems.append(system)
        selected = None
        if environments is not None:
            selected = Labels(
                ["system", "atom"],
                np.array(
                    sorted({(i, a) for i, a, _ in environments}), dtype=np.int32
                ).reshape(-1, 2),
            )
        output = mta.execute_model(
            self.model,
            systems,
            requested,
            selected_atoms=selected,
            check_consistency=self.check_consistency,
        )[0]
        return metatensor_to_property(output, systems, environments=environments)[
            "values"
        ]
