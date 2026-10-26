import inspect
import json
import tempfile
import textwrap
import unittest
from pathlib import Path
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

    def test_shapes_and_serialization(self):
        s = system()
        arrows = chemiscope.vectors_to_arrows([[1, 2, 3], [4, 5, 6]], scale=2)
        self.assertEqual(arrows["parameters"]["atom"][0]["vector"], [2, 4, 6])
        ellipsoids = chemiscope.tensors_to_ellipsoids([np.eye(3), 4 * np.eye(3)])
        np.testing.assert_allclose(
            ellipsoids["parameters"]["atom"][1]["semiaxes"], [2, 2, 2]
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.json"
            chemiscope.write_input(
                str(path), [s], shapes={"forces": arrows, "alpha": ellipsoids}
            )
            data = json.loads(path.read_text())
            self.assertEqual(data["structures"][0]["size"], 2)
            external = chemiscope.write_external_structures(
                [s], prefix=str(Path(directory) / "system")
            )
            self.assertEqual(external[0]["size"], 2)
        for values in ([[1, 2]], [1, 2, 3]):
            with self.assertRaises(ValueError):
                chemiscope.vectors_to_arrows(values)
        with self.assertRaises(ValueError):
            chemiscope.tensors_to_ellipsoids([[1, 2, 3]])

    def test_features_and_batches(self):
        class Model(mta.NumpyModel):
            def requested_pair_lists(self, outputs):
                return [mta.PairListOptions(cutoff=1.5, full_list=False)]

            def capabilities(self):
                return mta.ModelCapabilities(
                    atomic_types=[1, 8],
                    interaction_range=1.5,
                    length_unit="angstrom",
                    supported_devices=["cpu"],
                    dtype="float64",
                    outputs=[
                        mta.Quantity(name="feature", unit="", sample_kind=kind)
                        for kind in ("atom", "system")
                    ],
                )

            def metadata(self):
                return mta.ModelMetadata()

            def execute_inner(self, systems, selected_atoms, requested_outputs):
                for s in systems:
                    pairs = s.pairs(self.requested_pair_lists(requested_outputs)[0])
                    assert len(pairs.samples) == int(s.positions[1, 0] < 1.5)
                if requested_outputs[0].sample_kind == "system":
                    rows = [(i,) for i in reversed(range(len(systems)))]
                    values = [systems[i].positions.sum(axis=0) for (i,) in rows]
                    return [tensor(values, rows, names=["system"])]
                rows = list(reversed(selected_atoms.values.tolist()))
                return [tensor([systems[i].positions[a] for i, a in rows], rows)]

        structures = [system(), system("nm"), system()]
        for batch_size in (None, 1, 2):
            f = chemiscope.metatomic_featurizer(
                Model(), batch_size=batch_size, check_consistency=True
            )
            np.testing.assert_allclose(
                f(structures, None), [[1, 0, 0], [10, 0, 0], [1, 0, 0]]
            )
            np.testing.assert_allclose(
                f(structures, [(1, 1, 3), (0, 0, 3), (1, 1, 3)]),
                [[10, 0, 0], [0, 0, 0], [10, 0, 0]],
            )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.py"
            path.write_text(
                "import numpy as np\nimport metatomic as mta\n"
                "from metatensor import Labels, TensorBlock, TensorMap\n"
                + inspect.getsource(tensor)
                + "\n"
                + textwrap.dedent(inspect.getsource(Model))
                + "\nmodel = Model()\n"
            )
            mta.load_plugin(mta.utils.python_plugin_path)
            f = chemiscope.metatomic_featurizer(mta.load_model(str(path)))
            np.testing.assert_allclose(f([system()], None), [[1, 0, 0]])
        with self.assertRaisesRegex(ValueError, "CPU"):
            chemiscope.metatomic_featurizer(Model(), device="cuda")
        with self.assertRaisesRegex(ValueError, "does not declare"):
            chemiscope.metatomic_featurizer(Model(), variant="missing")

    def test_batched_properties(self):
        structures = [system(), system()]
        for i, s in enumerate(structures):
            s.add_custom_data("test::x", tensor([[i], [i + 2]], [[0, 1], [0, 0]]))
        batched = tensor([[1], [2], [3], [0]], [[1, 1], [0, 0], [1, 0], [0, 1]])
        for envs in (None, [(1, 0, 3), (0, 1, 3), (1, 0, 3)]):
            expected = chemiscope.extract_properties(structures, environments=envs)
            actual = chemiscope.metatensor_to_property(
                batched, structures, environments=envs
            )
            np.testing.assert_array_equal(
                actual["values"], expected["test::x"]["values"]
            )

    def test_custom_data_validation(self):
        structures = [system(), system()]
        t = tensor([[1], [2]], [[0, 0], [0, 1]])
        structures[0].add_custom_data("test::x", t.copy())
        changed_labels = TensorMap(
            Labels.single(),
            [
                TensorBlock(
                    np.ones((2, 1)), t.block(0).samples, [], Labels.range("other", 1)
                )
            ],
        )
        structures[1].add_custom_data("test::x", changed_labels)
        with self.assertRaisesRegex(ValueError, "inconsistent custom data layout"):
            chemiscope.extract_properties(structures)
        multiple = TensorMap(
            Labels.range("key", 2), [t.block(0).copy(), t.block(0).copy()]
        )
        with self.assertRaisesRegex(ValueError, "exactly one block"):
            chemiscope.metatensor_to_property(multiple, [structures[0]])
        s = system()
        s.add_custom_data("test::multiple", multiple)
        with self.assertRaisesRegex(ValueError, "exactly one block"):
            chemiscope.extract_properties([s])
        block = t.block(0).copy()
        block.add_gradient(
            "test",
            TensorBlock(
                np.ones((2, 1)), Labels.range("sample", 2), [], t.block(0).properties
            ),
        )
        with self.assertRaisesRegex(ValueError, "gradients explicitly"):
            chemiscope.metatensor_to_property(
                TensorMap(Labels.single(), [block]), [structures[0]]
            )


if __name__ == "__main__":
    unittest.main()
