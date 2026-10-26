Metatomic Core
==============

Core ``metatomic.System`` objects work with ``show``, ``create_input``,
``write_input``, ``write_external_structures`` and ``all_atomic_environments``.
ASE is not needed for this path. Install ``chemiscope[metatomic]`` and install
SciPy if you want tensor ellipsoids.

Structures, properties and shapes
---------------------------------

This example constructs a system and attaches a vector property. Custom data uses
local sample labels: the system index is always zero within each System. Atom
indices refer to that System's atoms. Property extraction aligns values by these
labels, even when the stored rows are shuffled.

.. code-block:: python

    import numpy as np
    import chemiscope
    from metatomic import System
    from metatensor import Labels, TensorBlock, TensorMap

    system = System(
        length_unit="angstrom",
        types=np.array([8, 1, 1], dtype=np.int32),
        positions=np.array([[0., 0., 0.], [0.76, 0.59, 0.], [-0.76, 0.59, 0.]]),
        cell=np.zeros((3, 3)),
        pbc=np.zeros(3, dtype=bool),
    )
    forces = TensorMap(Labels.single(), [TensorBlock(
        values=np.array([[0., 0., 0.], [0.1, 0., 0.], [-0.1, 0., 0.]]),
        samples=Labels(["system", "atom"], np.array([[0, 0], [0, 1], [0, 2]])),
        components=[],
        properties=Labels.range("xyz", 3),
    )])
    system.add_custom_data("custom::forces", forces)
    structures = [system]
    properties = {"forces": chemiscope.extract_properties(structures)["custom::forces"]}
    properties["forces"]["units"] = "eV/angstrom"
    arrows = chemiscope.vectors_to_arrows(properties["forces"]["values"], scale=1)
    chemiscope.write_input(
        "water.json", structures=structures, properties=properties,
        environments=chemiscope.all_atomic_environments(structures),
        shapes={"forces": arrows},
        settings={"structure": [{"shape": "forces"}]},
    )

The file uses the standard Chemiscope JSON format and can be loaded on the homepage.
Replace ``write_input("water.json", ...)`` with ``show(...)`` for a notebook widget.
For polarizabilities, pass arrays of shape ``(N, 6)``, ``(N, 9)`` or ``(N, 3, 3)``
to ``tensors_to_ellipsoids``. Shape arrays cover all atoms, even when the map uses
only selected environments. The scale maps property units to display lengths in
Angstroms. Energy position gradients must be negated to obtain forces.

Conversion conventions
----------------------

* Positions and cells are converted to Angstroms; an empty length unit is rejected.
* Types from 0 to 118 use the atomic-number convention (0 is ``X``). Other integer
  types keep their integer names and use element ``X``. For a model using a different
  type convention, supply explicit ``names`` and ``elements`` through Chemiscope
  structure dictionaries instead.
* Cells are included when all three cell vectors are nonzero, as for ASE inputs.
  Otherwise only atomic positions are displayed. Per-axis periodicity is not stored.
* Custom data must contain one block, with ``system`` or ``system, atom`` samples.
  Components and properties are flattened in stored order; layouts must agree across
  systems. Only names present in every system are extracted. Missing selected samples
  raise an error. Units and descriptions are supplied explicitly in the property dict.
* Use ``metatensor_to_property(tensor, structures, environments=...)`` for model
  outputs covering multiple systems. Gradients and multiple blocks require explicit
  selection/conversion; they are not silently discarded.
* NumPy and host-convertible Torch/JAX arrays are supported. Conversion does not
  change the input System's array backend. Raw DLPack arrays must be host-accessible.
