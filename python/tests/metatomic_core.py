import unittest
from unittest.mock import patch

import numpy as np

import chemiscope


try:
    import metatomic as mta
    from metatensor import Labels, TensorBlock, TensorMap

    HAVE_CORE = hasattr(mta, "System")
except ImportError:
    HAVE_CORE = False


def system(unit="angstrom", pbc=(False, False, False)):
    return mta.System(
        unit,
        np.array([8, 1], dtype=np.int32),
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        np.diag(pbc).astype(float) * 5,
        np.array(pbc),
    )


def tensor(values, samples, names=("system", "atom"), components=()):
    values = np.asarray(values, dtype=float)
    return TensorMap(
        Labels.single(),
        [
            TensorBlock(
                values,
                Labels(list(names), np.asarray(samples, dtype=np.int32)),
                list(components),
                Labels.range("feature", values.shape[-1]),
            )
        ],
    )


@unittest.skipUnless(HAVE_CORE, "requires metatomic-core Python bindings")
class TestCore(unittest.TestCase):
    def test_structures_and_units(self):
        s = system("nm", (True, True, True))
        data = chemiscope.create_input([s])["structures"][0]
        self.assertEqual(data["names"], ["O", "H"])
        self.assertEqual(data["x"], [0.0, 10.0])
        self.assertEqual(data["cell"], (50 * np.eye(3)).reshape(-1).tolist())
        self.assertEqual(s.length_unit, "nm")
        self.assertEqual(chemiscope.create_input(s)["structures"], [data])
        self.assertNotIn("cell", chemiscope.create_input([system()])["structures"][0])
        self.assertEqual(
            chemiscope.all_atomic_environments([s], 2), [(0, 0, 2), (0, 1, 2)]
        )
        partial = chemiscope.create_input([system(pbc=(True, False, False))])
        self.assertNotIn("cell", partial["structures"][0])
        with self.assertRaisesRegex(TypeError, "all structures"):
            chemiscope.create_input([s, {}])

    def test_ase_cell_compatibility(self):
        try:
            from ase import Atoms
        except ImportError:
            self.skipTest("requires ASE")
        for pbc in ((False, False, False), (True, False, False), (True, True, True)):
            s = system(pbc=pbc)
            atoms = Atoms(
                numbers=s.types, positions=s.positions, cell=s.cell, pbc=s.pbc
            )
            core = chemiscope.create_input([s])["structures"][0]
            ase = chemiscope.create_input([atoms])["structures"][0]
            core.pop("elements")
            self.assertEqual(core, ase)

    def test_properties_and_selection(self):
        structures = [system(), system()]
        for i, s in enumerate(structures):
            s.add_custom_data(
                "test::charge", tensor([[10 * i + 2], [10 * i + 1]], [[0, 1], [0, 0]])
            )
        envs = [(1, 1, 3), (0, 0, 3), (1, 1, 3)]
        prop = chemiscope.extract_properties(structures, environments=envs)[
            "test::charge"
        ]
        np.testing.assert_array_equal(prop["values"].ravel(), [12, 1, 12])
        self.assertEqual(prop["target"], "atom")
        self.assertEqual(chemiscope.extract_properties(structures, only=["absent"]), {})
        with patch("chemiscope.explore.show") as show:
            chemiscope.explore(structures, environments=envs, mode="structure")
        np.testing.assert_array_equal(
            show.call_args.kwargs["properties"]["test::charge"]["values"],
            prop["values"],
        )

    def test_tensor_validation(self):
        structures = [system()]
        t = tensor([[4], [3]], [[0, 1], [0, 0]])
        np.testing.assert_array_equal(
            chemiscope.metatensor_to_property(t, structures)["values"].ravel(), [3, 4]
        )
        with self.assertRaisesRegex(ValueError, "missing TensorMap sample"):
            chemiscope.metatensor_to_property(tensor([[1]], [[0, 0]]), structures)
        with self.assertRaisesRegex(ValueError, "samples must"):
            chemiscope.metatensor_to_property(
                tensor([[1]], [[0]], names=["atom"]), structures
            )
        with self.assertRaisesRegex(ValueError, "invalid environment"):
            chemiscope.metatensor_to_property(t, structures, environments=[(0, -1, 3)])
        structures[0].add_custom_data("test::x", t)
        with self.assertWarnsRegex(UserWarning, "every System"):
            self.assertEqual(chemiscope.extract_properties(structures + [system()]), {})


if __name__ == "__main__":
    unittest.main()
