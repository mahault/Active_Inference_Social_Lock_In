# Ported from David Hyland's epistemic-pluralism tests/test_topology.py (commit 6772b24).
# Only imports and the reference-fixture path changed.
"""Checks of network structure, role assignment and the signal-change protocol."""

import json
from pathlib import Path
import sys
import unittest

import numpy as np


from tests.pluralist_x64 import import_pluralist, restore, x64_only_here  # noqa: F401

model, experiments, _ = import_pluralist()
restore()


class NetworkTopologies(unittest.TestCase):
    def test_fixed_graphs_and_self_contact_weights(self):
        expected_edges = {"cycle": 24, "wheel": 46, "community": 203, "complete": 276}
        for name, edge_count in expected_edges.items():
            with self.subTest(topology=name):
                adjacency = experiments.network_adjacency(name)
                self.assertEqual(adjacency.shape, (24, 24))
                np.testing.assert_array_equal(adjacency, adjacency.T)
                np.testing.assert_array_equal(np.diag(adjacency), 0)
                self.assertEqual(adjacency.sum(), edge_count * 2)
                self_contact = adjacency + np.eye(24)
                W = self_contact / self_contact.sum(1, keepdims=True)
                np.testing.assert_allclose(W.sum(1), 1., atol=2e-15, rtol=0)
                self.assertTrue(np.all(np.diag(W) > 0))
                # Every agent can reach every other agent within 23 edges.
                self.assertTrue(np.all(np.linalg.matrix_power(self_contact, 23) > 0))
        cycle = experiments.network_adjacency("cycle")
        np.testing.assert_array_equal(cycle.sum(1), 2)
        self.assertEqual(cycle[0, 23], 1)
        wheel = experiments.network_adjacency("wheel")
        np.testing.assert_array_equal(wheel.sum(1), [23] + [3] * 23)
        self.assertEqual(wheel[1, 23], 1)
        community = experiments.network_adjacency("community")
        np.testing.assert_array_equal(community[:4, :4], np.ones((4, 4)) - np.eye(4))
        np.testing.assert_array_equal(community[4:, 4:], np.ones((20, 20)) - np.eye(20))
        bridges = {(int(a), int(b) + 4) for a, b in zip(*np.nonzero(community[:4, 4:]))}
        self.assertEqual(bridges, {(0, 7), (1, 7), (1, 19), (2, 8),
                                  (2, 10), (2, 22), (3, 23)})
        complete = experiments.network_adjacency("complete") + np.eye(24)
        np.testing.assert_array_equal(complete / 24, experiments.social_matrix(24))

    def test_topology_probabilities_and_individual_traces_across_shift(self):
        fixture = json.loads(Path(__file__).with_name("pluralist_reference.json").read_text())
        cfg = model.Config(*model.candidate_priors())
        observed = experiments.topology(cfg, [fixture["seed"]], fixture["steps"])
        np.testing.assert_array_equal(observed["full_observers"], [0, 1, 2, 3])
        states = np.asarray(fixture["topology_completed_states"]) - 1
        columns = np.asarray(fixture["topology_individual_nodes"]) - 4
        for name, expected in fixture["topologies"].items():
            with self.subTest(topology=name):
                np.testing.assert_allclose(observed[name + "__q_B"][0, states], expected["q_B"],
                                           atol=2e-11, rtol=1e-10)
                individual = observed[name + "__individual_B"]
                self.assertEqual(individual.shape, (fixture["steps"], 20))
                np.testing.assert_allclose(individual[np.ix_(states, columns)], expected["individual_B"],
                                           atol=2e-11, rtol=1e-10)
                np.testing.assert_allclose(individual.mean(1), observed[name + "__q_B"][0],
                                           atol=2e-15, rtol=0)
        # The complete graph has the same sharing and attention as Figure 4's
        # N11 arm; its separately constructed trajectory must agree.
        np.testing.assert_allclose(observed["complete__q_B"][0, 79:84],
                                   fixture["conditions"]["N11"]["q_B"],
                                   atol=2e-11, rtol=1e-10)


if __name__ == "__main__":
    unittest.main()
