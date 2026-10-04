#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""evacrl.casebuild (the part that needs only NumPy and SciPy): network clean-up, actions, shortest paths, agents,
reading and writing, and the validator. The synthetic network of the tests (metres):

         5 (100,100)
        /  \\                             link lengths: 0-1 100, 1-2 100 (and a second, 150, between the same two),
   0 --- 1 --- 2 - 3 --- 4 (shelter)               2-3 3 (too short to keep), 3-4 100, 1-5 100, 5-4 250
 (0,0) (100,0) (200,0) (203,0) (303,0)

Run: python -m unittest discover tests
"""
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from evacrl.casebuild import (NO_PATH, Network, actions_and_transitions, agents_table, apportion, build_tables,  # noqa: E402
                              candidate_nodes, distance_to_shelter, merge_short_links, network_from_edges, next_nodes,
                              prune_excess_links, read_raw,
                              read_tables, start_nodes, start_nodes_per_node, start_nodes_proportional, validate_case,
                              validate_tables, write_provenance, write_raw, write_tables)
from evacrl.casebuild import cli  # noqa: E402
from evacrl.tables import load_table  # noqa: E402

XY = [(0, 0), (100, 0), (200, 0), (203, 0), (303, 0), (100, 100)]
EDGES = [(10, 11, 100.4), (11, 12, 100.9), (12, 11, 150.2), (12, 13, 3.2), (13, 14, 100.0), (11, 15, 100.0), (15, 14, 250.0)]
OSMID = [10, 11, 12, 13, 14, 15]


def raw_network(shelters=(14,), rounding="floor"):
    return network_from_edges(OSMID, XY, EDGES, evacuation_osmids=shelters, length_rounding=rounding, crs="EPSG:32653")


def cleaned():
    return merge_short_links(raw_network())[0]


class FromEdges(unittest.TestCase):
    def test_nodes_numbers_shelter_and_rewards(self):
        net = raw_network()
        np.testing.assert_array_equal(net.nodes[:, 0], np.arange(6))
        np.testing.assert_array_equal(net.nodes[:, 3], [0, 0, 0, 0, 1, 0])
        np.testing.assert_array_equal(net.nodes[:, 4], [1, 1, 1, 1, 1000, 1])
        self.assertEqual(net.shelters.tolist(), [4])

    def test_lengths_are_truncated_like_the_2024_study_or_rounded(self):
        floor = raw_network(rounding="floor").links[:, 3]
        near = raw_network(rounding="round").links[:, 3]
        np.testing.assert_array_equal(floor, [100, 100, 150, 3, 100, 100, 250])
        np.testing.assert_array_equal(near, [100, 101, 150, 3, 100, 100, 250])

    def test_loops_and_links_under_a_metre_are_dropped(self):
        net = network_from_edges([1, 2], [(0, 0), (5, 0)], [(1, 1, 10.0), (1, 2, 0.6), (1, 2, 7.9)])
        self.assertEqual(net.links[:, 1:4].tolist(), [[0, 1, 7]])

    def test_unknown_rounding_is_rejected(self):
        with self.assertRaises(ValueError):
            raw_network(rounding="up")


class MergeShortLinks(unittest.TestCase):
    def test_a_short_link_merges_its_two_nodes_into_one_at_their_mean_position(self):
        net, report = merge_short_links(raw_network())
        self.assertEqual(net.num_nodes, 5)
        self.assertEqual(net.nodes[2, 1:3].tolist(), [201.5, 0.0])     # nodes 2 and 3
        self.assertEqual(net.nodes[:, 0].tolist(), [0, 1, 2, 3, 4])    # renumbered, no gap
        self.assertEqual(net.osmid.tolist(), [10, 11, 12, 14, 15])     # the smallest member names the cluster
        self.assertEqual((report["short_links"], report["nodes_before"], report["nodes_after"]), (1, 6, 5))
        self.assertEqual(net.num_links, 6)                              # the short link is gone
        self.assertTrue((net.links[:, 3] >= 5).all())
        self.assertEqual(net.crs, "EPSG:32653")

    def test_no_node_is_left_without_links_and_every_link_points_at_a_node(self):
        net = cleaned()
        used = set(net.links[:, 1:3].astype(int).ravel())
        self.assertEqual(used, set(range(net.num_nodes)))

    def test_the_merged_node_is_a_shelter_if_any_member_is(self):
        net, report = merge_short_links(network_from_edges(OSMID, XY, EDGES, evacuation_osmids=[13]))
        self.assertEqual(net.shelters.tolist(), [2])
        self.assertEqual((report["shelters_before"], report["shelters_after"]), (1, 1))

    def test_a_chain_of_short_links_becomes_one_node(self):
        xy = [(0, 0), (3, 0), (7, 0), (107, 0)]
        edges = [(1, 2, 3.0), (2, 3, 4.0), (3, 4, 100.0)]
        net, report = merge_short_links(network_from_edges([1, 2, 3, 4], xy, edges, evacuation_osmids=[4]))
        self.assertEqual(net.num_nodes, 2)
        self.assertEqual(net.links[:, 1:4].tolist(), [[0, 1, 100]])
        self.assertEqual(report["clusters_of_3_or_more"], 1)
        self.assertAlmostEqual(net.nodes[0, 1], 10 / 3)

    def test_a_link_inside_a_cluster_is_dropped_not_kept_as_a_loop(self):
        # 1-2 and 2-3 are short; 1-3 is long (a bend) but both its ends are now the same node
        xy = [(0, 0), (3, 0), (6, 0)]
        edges = [(1, 2, 3.0), (2, 3, 3.0), (1, 3, 20.0)]
        net, report = merge_short_links(network_from_edges([1, 2, 3], xy, edges))
        self.assertEqual((net.num_nodes, net.num_links, report["links_dropped_as_loops"]), (1, 0, 1))

    def test_a_longer_threshold_merges_more(self):
        # every link but the 150 m and the 250 m one is shorter than 101 m and joins the nodes into one cluster
        net, report = merge_short_links(raw_network(), threshold=101)
        self.assertEqual((net.num_nodes, net.num_links, report["links_dropped_as_loops"]), (1, 0, 2))
        self.assertEqual(net.shelters.tolist(), [0])

    def test_the_input_is_not_changed(self):
        raw = raw_network()
        before = raw.nodes.copy(), raw.links.copy()
        merge_short_links(raw)
        np.testing.assert_array_equal(raw.nodes, before[0])
        np.testing.assert_array_equal(raw.links, before[1])

    def test_legacy_method_keeps_the_defects_of_the_2024_study(self):
        # the pair 12-13 is merged but its merged position goes to a row nobody uses: an orphan node with no links
        net, report = merge_short_links(raw_network(), method="legacy")
        degree = np.bincount(net.links[:, 1:3].astype(int).ravel(), minlength=net.num_nodes)
        self.assertEqual(net.num_nodes, 6)
        self.assertEqual(report["orphan_rows"], 1)
        self.assertEqual(int((degree == 0).sum()), 1)

    def test_legacy_method_cuts_a_chain_off_from_its_shelter_where_clusters_do_not(self):
        # 1-2 and 2-3 are short (a chain), 3-4 is long and 4 is the shelter. Legacy merges pairs, not chains: the real
        # node 1 keeps no link and the orphan it leaves behind has none either
        xy = [(0, 0), (3, 0), (7, 0), (107, 0)]
        edges = [(1, 2, 3.0), (2, 3, 4.0), (3, 4, 100.0)]
        raw = network_from_edges([1, 2, 3, 4], xy, edges, evacuation_osmids=[4])
        legacy, _ = merge_short_links(raw, method="legacy")
        clusters, _ = merge_short_links(raw)
        without_a_way = lambda net: int((next_nodes(net)[:, 1] == NO_PATH).sum())
        self.assertEqual(without_a_way(clusters), 0)
        self.assertEqual(without_a_way(legacy), 2)

    def test_unknown_method_is_rejected(self):
        with self.assertRaises(ValueError):
            merge_short_links(raw_network(), method="x")


class ActionsAndTransitions(unittest.TestCase):
    def test_rows_by_hand(self):
        net = cleaned()   # nodes 0..4 = osm 10, 11, 12+13, 14 (shelter), 15; links in order of EDGES minus the short one
        # links: 0: 0-1, 1: 1-2, 2: 2-1 (the 150 m one), 3: 2-3 (was 13-14), 4: 1-4 (was 11-15), 5: 4-3 (was 15-14)
        self.assertEqual(net.links[:, 1:3].astype(int).tolist(), [[0, 1], [1, 2], [2, 1], [2, 3], [1, 4], [4, 3]])
        a, t = actions_and_transitions(net)
        self.assertEqual(a.shape, (5, 12))
        self.assertEqual(a[0].tolist(), [0, 1, 0] + [0] * 9)
        # node 1: links that start there in table order (1, 4), then those that end there (0, 2)
        self.assertEqual(a[1].tolist(), [1, 4, 1, 4, 0, 2] + [0] * 6)
        self.assertEqual(t[1].tolist(), [1, 4, 2, 4, 0, 2] + [0] * 6)
        # the shelter: one choice, stay
        self.assertEqual(a[3].tolist(), [3, 1, -1] + [0] * 9)
        self.assertEqual(t[3].tolist(), [3, 1, 3] + [0] * 9)

    def test_an_isolated_node_has_no_choices(self):
        net = Network([[0, 0, 0, 0, 1], [1, 5, 0, 0, 1], [2, 9, 9, 1, 1]], [[0, 0, 1, 5, 3]])
        a, _ = actions_and_transitions(net)
        self.assertEqual(a[:, 1].tolist(), [1, 1, 1])  # the lone shelter still has its "stay"
        net.nodes[2, 3] = 0
        self.assertEqual(actions_and_transitions(net)[0][2, 1], 0)

    def test_more_links_than_the_model_can_hold_is_an_error(self):
        n = 12
        nodes = [[i, i, 0, 0, 1] for i in range(n)]
        links = [[i, 0, i + 1, 5, 3] for i in range(n - 1)]   # node 0 with 11 links
        with self.assertRaisesRegex(ValueError, "more than 10 links"):
            actions_and_transitions(Network(nodes, links))


class ExcessLinks(unittest.TestCase):
    def star(self, n=13, shelter=None):
        """Node 0 in the middle with n links to nodes 1..n, of length 10 + k (the last, n, is the longest), and a ring
        round the others so that every outer node keeps a second link."""
        nodes = [[i, i, 0, 0, 1] for i in range(n + 1)]
        links = [[k - 1, 0, k, 10 + k, 3] for k in range(1, n + 1)]
        links += [[n + k - 1, k, k % n + 1, 50, 3] for k in range(1, n + 1)]
        if shelter is not None:
            nodes[shelter][3] = 1
        return Network(nodes, links)

    def test_the_longest_links_of_the_busiest_node_go_first(self):
        net, removed = prune_excess_links(self.star(13))
        degree = np.bincount(net.links[:, 1:3].astype(int).ravel(), minlength=14)
        self.assertEqual(degree.max(), 10)
        self.assertEqual(sorted(removed), [10, 11, 12])                      # links 0-11, 0-12, 0-13 (lengths 21, 22, 23)
        self.assertEqual(net.links[:, 0].tolist(), list(range(net.num_links)))   # renumbered, no gap
        self.assertEqual(net.num_links, 26 - 3)

    def test_a_network_within_the_limit_is_returned_as_it_is(self):
        net = self.star(8)
        out, removed = prune_excess_links(net)
        self.assertEqual(removed, [])
        np.testing.assert_array_equal(out.links, net.links)

    def test_evacuation_nodes_are_not_counted_because_they_ignore_their_links(self):
        _, removed = prune_excess_links(self.star(13, shelter=0))
        self.assertEqual(removed, [])

    def test_the_model_tables_refuse_an_excess_node_unless_told_to_prune(self):
        raw = self.star(12)
        raw.nodes[1][3] = 1
        with self.assertRaisesRegex(ValueError, "more than 10 links"):
            build_tables(raw, threshold=1)
        t = build_tables(raw, threshold=1, excess="prune")
        self.assertEqual(t["merge_report"]["links_pruned_for_degree"], 2)
        self.assertEqual(int(t["actions"][:, 1].max()), 10)
        with self.assertRaises(ValueError):
            build_tables(raw, excess="x")


class ShortestPaths(unittest.TestCase):
    def test_next_nodes_by_hand(self):
        net = cleaned()
        table = next_nodes(net)
        # shelter 3; 4 -> 3 (250); 1 -> 2 -> 3 is 100+100 +? the merged 2-3 link is 100: 1-2 100, 2-3 100 = 200 vs 1-4-3 = 350
        self.assertEqual(table[:, 1].tolist(), [1, 2, 3, 3, 3])
        np.testing.assert_array_equal(table[:, 0], np.arange(5))

    def test_the_shortest_of_several_links_between_two_nodes_is_used(self):
        # shelter 2. Two links between 1 and 2 (100 m and 10 m), 0-1 90 m, 0-2 100 m
        net = Network([[0, 0, 0, 0, 1], [1, 10, 0, 0, 1], [2, 20, 0, 1, 1]],
                      [[0, 0, 1, 90, 3], [1, 1, 2, 100, 3], [2, 1, 2, 10, 3], [3, 0, 2, 100, 3]])
        self.assertEqual(distance_to_shelter(net, parallel="min").tolist(), [100.0, 10.0, 0.0])
        self.assertEqual(distance_to_shelter(net, parallel="last").tolist(), [100.0, 10.0, 0.0])
        # make the second link between 1 and 2 long (500 m): the shortest of the two is now the first (100 m); the
        # 2024 study kept the last one (500 m) and so sent node 1 the long way round, 1 -> 0 -> 2 (190 m)
        net.links[2, 3] = 500
        self.assertEqual(distance_to_shelter(net, parallel="min").tolist(), [100.0, 100.0, 0.0])
        self.assertEqual(distance_to_shelter(net, parallel="last").tolist(), [100.0, 190.0, 0.0])
        np.testing.assert_array_equal(next_nodes(net, parallel="min")[:, 1], [2, 2, 2])
        np.testing.assert_array_equal(next_nodes(net, parallel="last")[:, 1], [2, 0, 2])

    def test_a_node_without_a_way_to_a_shelter_has_no_path(self):
        net = Network([[0, 0, 0, 0, 1], [1, 5, 0, 1, 1], [2, 50, 0, 0, 1], [3, 55, 0, 0, 1]],
                      [[0, 0, 1, 5, 3], [1, 2, 3, 5, 3]])
        self.assertEqual(next_nodes(net)[:, 1].tolist(), [1, 1, NO_PATH, NO_PATH])
        self.assertTrue(np.isinf(distance_to_shelter(net)[2:]).all())

    def test_no_shelter_at_all(self):
        net = Network([[0, 0, 0, 0, 1], [1, 5, 0, 0, 1]], [[0, 0, 1, 5, 3]])
        self.assertEqual(next_nodes(net)[:, 1].tolist(), [NO_PATH, NO_PATH])

    def test_both_methods_give_walks_of_the_same_length_on_random_networks(self):
        rng = np.random.default_rng(5)
        for trial in range(6):
            n = 60
            xy = rng.uniform(0, 500, size=(n, 2))
            pairs = {tuple(sorted(p)) for p in rng.integers(0, n, size=(130, 2)) if p[0] != p[1]}
            links = [[i, a, b, max(int(np.hypot(*(xy[a] - xy[b]))), 1), 3] for i, (a, b) in enumerate(sorted(pairs))]
            nodes = [[i, xy[i, 0], xy[i, 1], 1 if i in (3, 17, 41) else 0, 1] for i in range(n)]
            net = Network(nodes, links)
            fast, slow = next_nodes(net, "nearest"), next_nodes(net, "allpairs")
            dist = distance_to_shelter(net)
            self.assertEqual((fast[:, 1] == NO_PATH).tolist(), (slow[:, 1] == NO_PATH).tolist())
            self.assertEqual(np.isinf(dist).tolist(), (fast[:, 1] == NO_PATH).tolist())
            self.assertTrue(validate_tables(net.nodes, net.links, *actions_and_transitions(net), fast).ok)
            self.assertTrue(validate_tables(net.nodes, net.links, *actions_and_transitions(net), slow).ok)

    def test_unknown_options_are_rejected(self):
        net = cleaned()
        with self.assertRaises(ValueError):
            next_nodes(net, method="x")
        with self.assertRaises(ValueError):
            next_nodes(net, parallel="x")


class Degenerate(unittest.TestCase):
    def test_a_network_with_a_shelter_and_no_links(self):
        net = Network([[0, 0, 0, 1, 1000]], np.zeros((0, 5)))
        self.assertEqual(next_nodes(net).tolist(), [[0, 0]])
        self.assertEqual(distance_to_shelter(net).tolist(), [0.0])
        a, t = actions_and_transitions(net)
        self.assertTrue(validate_tables(net.nodes, net.links, a, t, next_nodes(net)).ok)

    def test_two_nodes_and_a_link_that_is_not_a_link(self):
        net = Network([[0, 0, 0, 1, 1], [1, 5, 0, 0, 1]], [[0, 0, 1, 0, 3]])    # zero length
        self.assertEqual(next_nodes(net)[:, 1].tolist(), [0, NO_PATH])

    def dense_last(self, net):
        """The 2024 study's rule, written the way it did it: a dense matrix filled link by link."""
        n = net.num_nodes
        dense = np.zeros((n, n))
        for _, a, b, length, _ in net.links:
            dense[int(a), int(b)] = dense[int(b), int(a)] = length
        return dense

    def test_the_last_of_several_links_wins_as_in_the_dense_version_but_in_sparse_form(self):
        from evacrl.casebuild.routing import _adjacency
        rng = np.random.default_rng(11)
        for _ in range(5):
            n = 40
            pairs = rng.integers(0, n, size=(160, 2))
            links = [[i, a, b, int(rng.integers(1, 50)), 3] for i, (a, b) in enumerate(pairs) if a != b]   # many repeated pairs
            net = Network([[i, i, 0, 1 if i == 5 else 0, 1] for i in range(n)], links)
            np.testing.assert_array_equal(_adjacency(net, "last").toarray(), self.dense_last(net))

    def test_the_shortest_rule_never_exceeds_the_last_rule(self):
        from evacrl.casebuild.routing import _adjacency
        net = Network([[i, i, 0, 0, 1] for i in range(3)], [[0, 0, 1, 9, 3], [1, 1, 0, 4, 3], [2, 1, 2, 7, 3]])
        self.assertEqual(_adjacency(net, "min")[0, 1], 4)
        self.assertEqual(_adjacency(net, "last")[0, 1], 4)    # the last of the two links 0-1 happens to be the short one
        net.links[1, 3] = 40
        self.assertEqual(_adjacency(net, "min")[0, 1], 9)
        self.assertEqual(_adjacency(net, "last")[0, 1], 40)

    def test_a_big_grid_is_routed_in_sparse_form_even_with_the_last_link_rule(self):
        n = 140                                                    # 19,600 nodes: a dense matrix would take 3 GB
        idx = lambda r, c: r * n + c
        nodes = [[idx(r, c), c * 10, r * 10, 1 if idx(r, c) == 0 else 0, 1] for r in range(n) for c in range(n)]
        links = []
        for r in range(n):
            for c in range(n):
                if c + 1 < n:
                    links.append([len(links), idx(r, c), idx(r, c + 1), 10, 3])
                if r + 1 < n:
                    links.append([len(links), idx(r, c), idx(r + 1, c), 10, 3])
        table = next_nodes(Network(nodes, links), parallel="last")
        self.assertEqual(int((table[:, 1] == NO_PATH).sum()), 0)
        self.assertEqual(distance_to_shelter(Network(nodes, links))[idx(n - 1, n - 1)], 2 * (n - 1) * 10)


class Agents(unittest.TestCase):
    def setUp(self):
        self.net = cleaned()
        self.next = next_nodes(self.net)

    def test_candidates_exclude_shelters_and_nodes_without_a_way_out(self):
        self.assertEqual(candidate_nodes(self.net, self.next).tolist(), [0, 1, 2, 4])
        self.assertEqual(candidate_nodes(self.net, self.next, exclude_shelters=False).tolist(), [0, 1, 2, 3, 4])
        cut = self.next.copy()
        cut[0, 1] = NO_PATH
        self.assertEqual(candidate_nodes(self.net, cut).tolist(), [1, 2, 4])

    def test_random_start_nodes_are_reproducible_and_stay_among_the_candidates(self):
        a = start_nodes(self.net, self.next, 200, np.random.default_rng(1))
        b = start_nodes(self.net, self.next, 200, np.random.default_rng(1))
        np.testing.assert_array_equal(a, b)
        self.assertTrue(set(a) <= {0, 1, 2, 4})
        self.assertTrue((np.diff(a) >= 0).all())

    def test_weights_steer_the_draw(self):
        w = [0, 0, 5, 0, 0]
        got = start_nodes(self.net, self.next, 50, np.random.default_rng(2), weights=w)
        self.assertEqual(set(got), {2})
        with self.assertRaises(ValueError):
            start_nodes(self.net, self.next, 5, np.random.default_rng(2), weights=[0, 0, 0, 5, 0])

    def test_per_node(self):
        self.assertEqual(start_nodes_per_node(self.net, self.next, 3).tolist(), [0] * 3 + [1] * 3 + [2] * 3 + [4] * 3)

    def test_apportion_adds_up_and_is_proportional_without_randomness(self):
        self.assertEqual(apportion([1, 1, 1], 10).tolist(), [4, 3, 3])   # the extra one goes to the lowest node
        counts = apportion([0.2, 0.5, 0.3], 1000)
        self.assertEqual(counts.tolist(), [200, 500, 300])
        for total in (1, 7, 99, 12345):
            self.assertEqual(int(apportion(np.random.default_rng(total).random(17), total).sum()), total)
        with self.assertRaises(ValueError):
            apportion([0, 0], 3)
        for bad in ([1, -1, 3], [1, np.nan, 3], [1, np.inf, 3]):
            with self.assertRaisesRegex(ValueError, "finite and not negative"):
                apportion(bad, 4)

    def test_bad_weights_are_refused_for_the_random_draw_too(self):
        with self.assertRaisesRegex(ValueError, "finite and not negative"):
            start_nodes(self.net, self.next, 5, np.random.default_rng(0), weights=[1, -1, 1, 1, 1])

    def test_proportional_start_nodes(self):
        starts = start_nodes_proportional(self.net, self.next, [10, 0, 30, 99, 60], 100)
        self.assertEqual(np.bincount(starts, minlength=5).tolist(), [10, 0, 30, 0, 60])   # node 3, a shelter, gets none

    def test_agents_table(self):
        table = agents_table([4, 0, 1])
        self.assertEqual(table.shape, (3, 5))
        self.assertEqual(table[:, 4].tolist(), [4, 0, 1])
        self.assertEqual(int(table[:, :4].sum()), 0)


class ReadWrite(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.net = cleaned()
        self.actions, self.trans = actions_and_transitions(self.net)
        self.next = next_nodes(self.net)
        self.agents = agents_table(start_nodes_per_node(self.net, self.next, 2))

    def tearDown(self):
        self._tmp.cleanup()

    def test_tables_round_trip_and_are_what_the_model_reads(self):
        write_tables(self.dir / "data", self.net, self.actions, self.trans, self.next, self.agents)
        back = read_tables(self.dir / "data")
        np.testing.assert_allclose(back["nodes"], self.net.nodes, atol=5e-7)
        np.testing.assert_array_equal(back["links"], self.net.links)
        for key, expected in (("actions", self.actions), ("transitions", self.trans), ("nextnode", self.next), ("agents", self.agents)):
            np.testing.assert_array_equal(back[key], expected, err_msg=key)
        # headers are comments ("# ...") like the rest of the repository
        self.assertTrue((self.dir / "data" / "nodesdb.csv").read_text().startswith("# number,coord_x"))

    def test_the_model_can_load_what_was_written(self):
        from evacrl.sarsa import SARSA
        write_tables(self.dir / "data", self.net, self.actions, self.trans, self.next, self.agents)
        d = self.dir / "data"
        model = SARSA(agentsProfileName=str(d / "agentsdb.csv"), nodesdbFile=str(d / "nodesdb.csv"),
                      linksdbFile=str(d / "linksdb.csv"), transLinkdbFile=str(d / "actionsdb.csv"),
                      transNodedbFile=str(d / "transitionsdb.csv"), meanRayleigh=10)
        self.assertEqual(model.numPedestrian, len(self.agents))
        model.loadShortestPathDB(str(d / "nextnode.csv"))
        for t in range(int(min(model.pedDB[:, 9])), 2000):
            model.initEvacuationAtTime()
            model.stepForward()
            model.checkTargetShortestPath()
            if not t % 10:
                model.computePedHistDenVelAtLinks()
                model.updateVelocityAllPedestrians()
        self.assertEqual(model.getNumberEvacuatedPed(), len(self.agents))

    def test_raw_network_round_trip_keeps_ids_and_crs(self):
        raw = raw_network()
        write_raw(self.dir / "raw", raw)
        back = read_raw(self.dir / "raw")
        np.testing.assert_allclose(back.nodes, raw.nodes, atol=5e-7)
        np.testing.assert_array_equal(back.links, raw.links)
        np.testing.assert_array_equal(back.osmid, raw.osmid)
        self.assertEqual(back.crs, "EPSG:32653")

    def test_raw_coordinates_keep_nine_decimals_so_that_a_rebuild_is_exact(self):
        xy = [(366011.123456789, 3704469.987654321), (366111.123456789, 3704469.987654321)]
        net = network_from_edges([1, 2], xy, [(1, 2, 100.0)], evacuation_osmids=[2])
        write_raw(self.dir / "raw2", net)
        np.testing.assert_allclose(read_raw(self.dir / "raw2").nodes[:, 1:3], net.nodes[:, 1:3], rtol=0, atol=2e-9)

    def test_rebuilding_from_the_raw_files_gives_the_same_tables(self):
        write_raw(self.dir / "raw", raw_network())
        again = merge_short_links(read_raw(self.dir / "raw"))[0]
        np.testing.assert_allclose(again.nodes, self.net.nodes, atol=5e-7)
        np.testing.assert_array_equal(again.links, self.net.links)

    def test_provenance_is_json(self):
        write_provenance(self.dir, {"seed": np.int64(3), "counts": np.array([1, 2]), "note": "高知"})
        info = json.loads((self.dir / "provenance.json").read_text(encoding="utf-8"))
        self.assertEqual(info, {"seed": 3, "counts": [1, 2], "note": "高知"})


class Validation(unittest.TestCase):
    def setUp(self):
        self.net = cleaned()
        self.a, self.t = actions_and_transitions(self.net)
        self.next = next_nodes(self.net)
        self.agents = agents_table(start_nodes_per_node(self.net, self.next, 1))

    def check(self, **changes):
        args = dict(nodes=self.net.nodes.copy(), links=self.net.links.copy(), actions=self.a.copy(),
                    transitions=self.t.copy(), nextnode=self.next.copy(), agents=self.agents.copy())
        args.update(changes)
        return validate_tables(**args)

    def test_a_good_case_has_no_errors_and_no_warnings(self):
        r = self.check()
        self.assertTrue(r.ok, str(r))
        self.assertEqual(r.warnings, [])
        self.assertEqual((r.stats["nodes"], r.stats["shelters"], r.stats["components"]), (5, 1, 1))

    def test_the_numbers_of_nodes_and_links_must_be_in_order(self):
        nodes = self.net.nodes.copy()
        nodes[[0, 1], 0] = [1, 0]
        self.assertIn("node numbers", " ".join(self.check(nodes=nodes).errors))
        links = self.net.links.copy()
        links[2, 0] = 9
        self.assertIn("link numbers", " ".join(self.check(links=links).errors))

    def test_links_must_point_at_nodes_and_have_a_length(self):
        links = self.net.links.copy()
        links[0, 2] = 77
        self.assertIn("do not exist", " ".join(self.check(links=links).errors))
        links = self.net.links.copy()
        links[0, 3] = 0
        self.assertIn("no length", " ".join(self.check(links=links).errors))
        links = self.net.links.copy()
        links[0, 3] = 12.5
        self.assertIn("whole numbers", " ".join(self.check(links=links).errors))
        links = self.net.links.copy()
        links[0, 2] = links[0, 1]
        self.assertIn("join a node to itself", " ".join(self.check(links=links).errors))

    def test_a_case_without_a_shelter_is_an_error(self):
        nodes = self.net.nodes.copy()
        nodes[:, 3] = 0
        self.assertIn("no node is an evacuation node", " ".join(self.check(nodes=nodes).errors))

    def test_actions_that_do_not_match_the_links_are_an_error_but_another_order_is_a_warning(self):
        a = self.a.copy()
        a[1, 2:4] = a[1, 3], a[1, 2]
        r = self.check(actions=a)
        self.assertTrue(r.ok)
        self.assertIn("another order", " ".join(r.warnings))
        a = self.a.copy()
        a[1, 2] = 5
        self.assertIn("does not match the links table", " ".join(self.check(actions=a).errors))

    def test_next_nodes_must_lead_to_a_shelter_along_links(self):
        nxt = self.next.copy()
        nxt[0, 1] = 3                                   # not a neighbour of node 0
        self.assertIn("not neighbours", " ".join(self.check(nextnode=nxt).errors))
        nxt = self.next.copy()
        nxt[1, 1], nxt[2, 1] = 2, 1                     # 1 -> 2 -> 1: a cycle
        self.assertIn("never reach a shelter", " ".join(self.check(nextnode=nxt).errors))
        nxt = self.next.copy()
        nxt[0, 1] = NO_PATH                             # a way exists but the table says none
        self.assertIn("no next node", " ".join(self.check(nextnode=nxt).errors))
        nxt = self.next.copy()
        nxt[3, 1] = 0                                   # a shelter must stay
        self.assertIn("shelter itself", " ".join(self.check(nextnode=nxt).errors))

    def test_next_nodes_that_are_negative_or_too_big_are_an_error_not_a_crash(self):
        for value in (-1, -5, 99):
            nxt = self.next.copy()
            nxt[0, 1] = value
            r = self.check(nextnode=nxt)
            self.assertIn("do not exist", " ".join(r.errors), value)

    def test_a_next_node_off_the_shortest_walk_is_a_warning(self):
        nxt = self.next.copy()
        nxt[1, 1] = 4                                   # 1 -> 4 -> 3 (350) instead of 1 -> 2 -> 3 (200)
        r = self.check(nextnode=nxt)
        self.assertTrue(r.ok, str(r))
        self.assertIn("not on a shortest walk", " ".join(r.warnings))

    def test_agents(self):
        ag = self.agents.copy()
        ag[0, 4] = 3                                    # a shelter
        r = self.check(agents=ag)
        self.assertTrue(r.ok)
        self.assertIn("start at a shelter", " ".join(r.warnings))
        ag = self.agents.copy()
        ag[0, 4] = 99
        self.assertIn("do not exist", " ".join(self.check(agents=ag).errors))
        net = Network(np.vstack([self.net.nodes, [5, 400, 400, 0, 1]]), self.net.links)   # an isolated node 5
        a, t = actions_and_transitions(net)
        nxt = next_nodes(net)
        ag = self.agents.copy()
        ag[0, 4] = 5
        r = validate_tables(net.nodes, net.links, a, t, nxt, ag)
        self.assertIn("no links", " ".join(r.errors))
        self.assertEqual(r.stats["isolated_nodes"], 1)

    def test_a_folder_is_validated_as_a_whole(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_tables(tmp, self.net, self.a, self.t, self.next, self.agents)
            self.assertTrue(validate_case(tmp).ok)
            (Path(tmp) / "actionsdb.csv").unlink()
            self.assertIn("missing", " ".join(validate_case(tmp).errors))


class CommandLine(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        write_raw(self.dir / "raw", raw_network())

    def tearDown(self):
        self._tmp.cleanup()

    def run_cli(self, *argv):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = cli.main([str(a) for a in argv])
        return code, out.getvalue()

    def test_from_raw_builds_a_valid_case_and_records_how(self):
        code, out = self.run_cli("from-raw", self.dir / "raw", self.dir / "case", "--agents", 40, "--seed", 7)
        self.assertEqual(code, 0, out)
        self.assertIn("40 agents", out)
        data = self.dir / "case" / "data"
        self.assertEqual(sorted(p.name for p in data.iterdir()),
                         ["actionsdb.csv", "agentsdb.csv", "linksdb.csv", "nextnode.csv", "nodesdb.csv", "transitionsdb.csv"])
        info = json.loads((self.dir / "case" / "provenance.json").read_text())
        self.assertEqual(info["settings"]["population"]["seed"], 7)
        self.assertEqual(info["settings"]["merge"], "clusters")
        self.assertEqual(info["merge"]["nodes_after"], 5)
        self.assertTrue((self.dir / "case" / "raw" / "nodes.csv").exists())
        self.assertEqual(self.run_cli("validate", self.dir / "case")[0], 0)

    def test_the_same_seed_gives_the_same_case_and_another_seed_another_population(self):
        for name, seed in (("a", 3), ("b", 3), ("c", 4)):
            self.assertEqual(self.run_cli("from-raw", self.dir / "raw", self.dir / name, "--agents", 60, "--seed", seed)[0], 0)
        read = lambda n: (self.dir / n / "data" / "agentsdb.csv").read_text()
        self.assertEqual(read("a"), read("b"))
        self.assertNotEqual(read("a"), read("c"))

    def test_per_node_population(self):
        self.assertEqual(self.run_cli("from-raw", self.dir / "raw", self.dir / "p", "--strategy", "per_node", "--per-node", 3)[0], 0)
        self.assertEqual(len(load_table(self.dir / "p" / "data" / "agentsdb.csv", dtype=int)), 12)   # 4 start nodes x 3

    def test_legacy_reproduces_the_2024_choices(self):
        self.assertEqual(self.run_cli("from-raw", self.dir / "raw", self.dir / "l", "--agents", 10, "--legacy")[0], 0)
        info = json.loads((self.dir / "l" / "provenance.json").read_text())
        self.assertEqual((info["settings"]["merge"], info["settings"]["parallel"]), ("legacy", "last"))
        self.assertEqual(info["merge"]["orphan_rows"], 1)
        self.assertFalse(info["settings"]["population"]["exclude_shelters"])
        self.assertTrue(info["validation"]["warnings"])   # the orphan row

    def test_rebuilding_a_case_from_its_own_raw_folder_changes_nothing_and_keeps_its_origin(self):
        self.run_cli("from-raw", self.dir / "raw", self.dir / "case", "--agents", 30, "--seed", 2)
        path = self.dir / "case" / "provenance.json"
        info = json.loads(path.read_text())
        info.update(census={"area_total": 30.0}, shelters={"points": 4}, command="python -m evacrl.casebuild from-snapshot ORIGINAL")
        path.write_text(json.dumps(info))
        (self.dir / "case" / "raw").mkdir(exist_ok=True)
        write_raw(self.dir / "case" / "raw", raw_network())
        with open(self.dir / "case" / "raw" / "nodes.csv", "a") as f:
            f.write("# touched by hand: a rewrite of this folder would lose it\n")
        before = {p.name: p.read_bytes() for p in (self.dir / "case" / "data").iterdir()}
        raw_before = {p.name: p.read_bytes() for p in (self.dir / "case" / "raw").iterdir()}
        code, out = self.run_cli("from-raw", self.dir / "case" / "raw", self.dir / "case", "--agents", 30, "--seed", 2)
        self.assertEqual(code, 0, out)
        self.assertEqual({p.name: p.read_bytes() for p in (self.dir / "case" / "data").iterdir()}, before)
        self.assertEqual({p.name: p.read_bytes() for p in (self.dir / "case" / "raw").iterdir()}, raw_before)
        again = json.loads(path.read_text())
        self.assertEqual((again["census"], again["shelters"], again["command"]), (info["census"], info["shelters"], info["command"]))
        self.assertIn("rebuilt_with", again)

    def test_a_uniform_population_needs_a_size(self):
        with self.assertRaises(SystemExit):
            self.run_cli("from-raw", self.dir / "raw", self.dir / "x")

    def test_validate_says_what_is_wrong_and_exits_with_1(self):
        self.run_cli("from-raw", self.dir / "raw", self.dir / "case", "--agents", 5)
        (self.dir / "case" / "data" / "nextnode.csv").write_text("# node,next node\n0,3\n1,2\n2,3\n3,3\n4,3\n")
        code, out = self.run_cli("validate", self.dir / "case" / "data")
        self.assertEqual(code, 1)
        self.assertIn("not neighbours", out)

    def test_an_invalid_case_is_not_written(self):
        write_raw(self.dir / "bad", network_from_edges([1, 2], [(0, 0), (10, 0)], [(1, 2, 10.0)]))   # no shelter
        with self.assertRaisesRegex(ValueError, "no node is an evacuation node"):
            self.run_cli("from-raw", self.dir / "bad", self.dir / "nope", "--agents", 3)
        self.assertFalse((self.dir / "nope").exists())


class TheRepositoryTables(unittest.TestCase):
    def test_the_shipped_cases_that_have_a_next_node_table_are_consistent(self):
        found = 0
        for area in (REPO / "cases").iterdir():
            data = area / "data"
            if (data / "nextnode.csv").exists() and (data / "actionsdb.csv").exists():
                found += 1
                r = validate_case(data)
                self.assertTrue(r.ok, f"{area.name}: {r}")
        self.assertGreaterEqual(found, 4)   # cases/kochi_area0, 1, 2 and 4


if __name__ == "__main__":
    unittest.main()
