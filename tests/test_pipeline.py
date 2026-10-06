"""Offline correctness/integration tests; no pretrained downloads or benchmark claims."""
import copy
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import torch
from transformers import RobertaConfig, RobertaModel

from evaluation.metrics import build_filter_index, evaluate_filtered
from models.kg_encoder_rgat import RGATEncoder
from models.reverse_jaket import ReverseJAKET
from models.scorer import ComplExScorer
from preprocessing.dataset import load_synthetic_toy_graph
from preprocessing.experiment_data import EntityTokens, align_relation_texts, dataset_manifest
from preprocessing.sampling import build_incoming_adjacency, sample_local_graph, split_epoch_train
from training.config import ExperimentConfig, semantic_enabled
from training.losses import bce_loss
from training.negative_sampling import sample_negatives
from training.train_pipeline import run_experiment


def tiny_lm():
    with torch.random.fork_rng():
        torch.manual_seed(17)
        return RobertaModel(RobertaConfig(vocab_size=40, hidden_size=16, num_hidden_layers=1,
                            num_attention_heads=2, intermediate_size=24, max_position_embeddings=32),
                            add_pooling_layer=False)


def tiny_dataset():
    return load_synthetic_toy_graph(num_entities=10, num_relations=2, num_train=24,
                                     num_valid=4, num_test=4, seed=42)


def tiny_tokens(n=10):
    ids = torch.tensor([[0, 4 + i, 5, 2, 1, 1] for i in range(n)])
    relations = torch.tensor([[0, 20, 5, 2, 1, 1], [0, 21, 5, 2, 1, 1]])
    return EntityTokens(ids, (ids != 1).long(), torch.ones(n, dtype=torch.bool),
                        EntityTokens(relations, (relations != 1).long(), torch.ones(2, dtype=torch.bool)))


def tiny_config(output_dir="unused", **updates):
    values = dict(dataset="synthetic", kg_dim=8, rgat_heads=2, layers_per_phase=1,
                  block_target_width=4, max_text_length=6, lm_micro_batch=3,
                  lm_name="offline-random-roberta-test-only", max_epochs=3,
                  warmup_epochs=1, mp_ratio=0.7, train_batch_size=4,
                  neighbors_per_hop=2, eval_every=1, eval_node_batch_size=3,
                  eval_triple_batch_size=3, output_dir=output_dir,
                  early_stopping_patience=0, use_amp=False, device="cpu")
    values.update(updates)
    return ExperimentConfig(**values).validate()


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_complex_matches_complex_arithmetic_and_candidate_scores(self):
        torch.manual_seed(4)
        scorer = ComplExScorer(3, 8).double()
        entities = torch.randn(5, 8, dtype=torch.double)
        def complex_value(x):
            return torch.complex(*x.chunk(2, -1))
        for h in range(5):
            for r in range(3):
                rid = torch.tensor([r])
                tails = scorer.score_all_tails(entities, entities[h:h+1], rid)[0]
                for t in range(5):
                    reference = (complex_value(entities[h]) * complex_value(scorer.relation_emb.weight[r])
                                 * complex_value(entities[t]).conj()).sum().real
                    actual = scorer.score(entities[h:h+1], rid, entities[t:t+1])[0]
                    heads = scorer.score_all_heads(entities, rid, entities[t:t+1])[0]
                    torch.testing.assert_close(actual, reference)
                    torch.testing.assert_close(tails[t], reference)
                    torch.testing.assert_close(heads[h], reference)
        with torch.no_grad():
            scorer.relation_emb.weight[0] = torch.tensor([0., 0., 0., 0., 1., 0., 0., 0.])
        h = torch.tensor([[1., 0., 0., 0., 0., 0., 0., 0.]], dtype=torch.double)
        t = torch.tensor([[0., 0., 0., 0., 1., 0., 0., 0.]], dtype=torch.double)
        self.assertNotEqual(scorer.score(h, torch.tensor([0]), t).item(),
                            scorer.score(t, torch.tensor([0]), h).item())
        with self.assertRaises(ValueError):
            ComplExScorer(2, 7)

    def test_filtered_ranks_ties_and_known_true_exclusion(self):
        class Model:
            scorer = ComplExScorer(1, 2)
        model = Model()
        with torch.no_grad():
            model.scorer.relation_emb.weight.zero_()
        entities = torch.zeros(3, 2)
        # Correct tail ties with one remaining candidate; another true tail is filtered.
        filters = build_filter_index([(0, 0, 1), (0, 0, 2)])
        metrics = evaluate_filtered(model, entities, [(0, 0, 1)], *filters)
        self.assertAlmostEqual(metrics["Tail_MRR"], 1 / 1.5)
        self.assertAlmostEqual(metrics["Head_MRR"], 1 / 2)
        self.assertEqual(metrics["Hits@1"], 0)

    def test_split_sampling_and_balanced_negatives(self):
        ds = tiny_dataset()
        mp, sup = split_epoch_train(ds.train, 0.7, 9)
        self.assertFalse(set(map(tuple, mp.tolist())) & set(map(tuple, sup.tolist())))
        self.assertEqual(len(mp) + len(sup), len(ds.train))
        neg = sample_negatives(sup, ds.num_entities, set(ds.train), 4, random.Random(0),
                               balanced_head_tail=True)
        self.assertTrue(torch.equal(neg[:, 0::2, 2], sup[:, 2:3].expand(-1, 2)))
        self.assertTrue(torch.equal(neg[:, 1::2, 0], sup[:, 0:1].expand(-1, 2)))
        self.assertFalse(set(map(tuple, neg.reshape(-1, 3).tolist())) & set(ds.train))
        incoming = build_incoming_adjacency(mp, ds.num_entities, ds.num_relations)
        seeds = torch.cat([sup[:, [0, 2]].flatten(), neg[..., [0, 2]].flatten()]).unique().tolist()
        graph = sample_local_graph(incoming, seeds, 2, 2, random.Random(4))
        mapped = graph.remap(sup)
        self.assertTrue(torch.equal(graph.global_nodes[mapped[:, 0]], sup[:, 0]))
        allowed = {(h, t, r) for h, r, t in mp.tolist()} | {(t, h, r + ds.num_relations) for h, r, t in mp.tolist()}
        actual = {(int(graph.global_nodes[s]), int(graph.global_nodes[t]), r)
                  for (s, t), r in zip(graph.edge_index.t().tolist(), graph.edge_type.tolist())}
        self.assertTrue(actual <= allowed)

    def test_chunked_rgat_matches_full_attention_including_isolated_nodes(self):
        torch.manual_seed(5)
        encoder = RGATEncoder(8, 4, num_layers=2, heads=2).eval()
        x = torch.randn(7, 8)
        edges = torch.tensor([[0, 1, 2, 4, 3, 0], [1, 2, 1, 1, 4, 4]])
        types = torch.tensor([0, 1, 2, 3, 2, 0])
        with torch.no_grad():
            torch.testing.assert_close(encoder(x, edges, types), encoder.encode_full(x, edges, types, 2))
            empty_edges, empty_types = torch.empty(2, 0, dtype=torch.long), torch.empty(0, dtype=torch.long)
            torch.testing.assert_close(encoder(x, empty_edges, empty_types),
                                       encoder.encode_full(x, empty_edges, empty_types, 2))

    def test_bridge_schedule_and_full_gradient_flow(self):
        cfg = tiny_config()
        self.assertEqual(semantic_enabled(1, cfg), 0)
        self.assertEqual(semantic_enabled(2, cfg), True)
        model = ReverseJAKET(10, 2, cfg, lm=tiny_lm()).train()
        nodes = torch.arange(10)
        edges = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]])
        types = torch.tensor([0, 1, 2, 3])
        tokens = tiny_tokens()
        representations = model.encode_local(nodes, edges, types, tokens, use_text=True)
        positives = torch.tensor([[0, 0, 1], [2, 1, 3]])
        negatives = torch.tensor([[0, 0, 4], [2, 1, 5]])
        loss = bce_loss(model.score_triples(representations, positives),
                                    model.score_triples(representations, negatives).unsqueeze(-1))
        loss.backward()
        for name in ("encoder1", "encoder2", "scorer", "bridge.kg_to_lm", "bridge.lm_to_kg",
                     "encoder1.relation_update", "encoder2.relation_update"):
            grads = [p.grad for n, p in model.named_parameters() if n.startswith(name) and p.requires_grad]
            self.assertTrue(any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads), name)
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in model.bridge.lm.parameters()))
        self.assertFalse(model.bridge.lm.lm.training)
        model.eval()
        with torch.no_grad():
            torch.testing.assert_close(model.encode_local(nodes, edges, types, tokens, True),
                                       model.encode_full(edges, types, tokens, True))

    def test_all_variants_write_real_artifacts_on_synthetic_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            for variant in ("baseline", "text_enhanced", "proposed"):
                out = Path(tmp) / variant
                cfg = tiny_config(str(out), variant=variant)
                result = run_experiment(cfg, dataset=tiny_dataset(), tokens=tiny_tokens(),
                                        lm=tiny_lm() if variant != "baseline" else None)
                self.assertTrue(result["results"]["synthetic"])
                self.assertGreater(result["results"]["test"]["MRR"], 0)
                if variant != "baseline":
                    self.assertEqual(result["results"]["best_semantic_enabled"], True)
                for filename in ("config.json", "data_manifest.json", "environment.json", "history.csv",
                                 "last_checkpoint.pt", "best_checkpoint.pt", "warmup_checkpoint.pt", "final_results.json"):
                    self.assertTrue((out / filename).exists(), filename)
                saved = torch.load(out / "last_checkpoint.pt", weights_only=True)
                self.assertFalse(any(k.startswith("bridge.lm.") for k in saved["model_state"]))
                self.assertEqual(len(json.loads((out / "final_results.json").read_text())["test"]), 6)
                if variant == "proposed":
                    from evaluate_checkpoint import evaluate_checkpoint
                    evaluated = evaluate_checkpoint(out / "best_checkpoint.pt", device="cpu",
                        dataset=tiny_dataset(), tokens=tiny_tokens(), lm=tiny_lm())
                    self.assertEqual(evaluated["test"], result["results"]["test"])


    def test_semantic_resume_matches_uninterrupted_training_and_rejects_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            full_cfg = tiny_config(str(Path(tmp) / "full"))
            run_experiment(full_cfg, dataset=tiny_dataset(), tokens=tiny_tokens(), lm=tiny_lm())
            partial_cfg = tiny_config(str(Path(tmp) / "resumed"), max_epochs=2)
            run_experiment(partial_cfg, dataset=tiny_dataset(), tokens=tiny_tokens(), lm=tiny_lm())
            resumed_cfg = copy.deepcopy(partial_cfg)
            resumed_cfg.max_epochs = 3
            checkpoint = Path(resumed_cfg.output_dir) / "last_checkpoint.pt"
            run_experiment(resumed_cfg, resume=checkpoint, dataset=tiny_dataset(), tokens=tiny_tokens(), lm=tiny_lm())
            full = torch.load(Path(full_cfg.output_dir) / "last_checkpoint.pt", weights_only=True)
            resumed = torch.load(checkpoint, weights_only=True)
            for name in full["model_state"]:
                self.assertTrue(torch.equal(full["model_state"][name], resumed["model_state"][name]), name)
            changed_cfg = copy.deepcopy(resumed_cfg)
            changed_cfg.lr *= 2
            with self.assertRaisesRegex(ValueError, "Resume changes"):
                run_experiment(changed_cfg, resume=checkpoint, dataset=tiny_dataset(), tokens=tiny_tokens(), lm=tiny_lm())

    def test_checkpoint_evaluation_warm_start_and_summary(self):
        from evaluate_checkpoint import evaluate_checkpoint
        from summarize_results import summarize
        from training.train_pipeline import _warm_start
        with tempfile.TemporaryDirectory() as tmp:
            cfg = tiny_config(str(Path(tmp) / "baseline"), variant="baseline")
            result = run_experiment(cfg, dataset=tiny_dataset())
            path = Path(cfg.output_dir) / "best_checkpoint.pt"
            evaluated = evaluate_checkpoint(path, device="cpu", dataset=tiny_dataset())
            self.assertEqual(evaluated["test"], result["results"]["test"])
            state = torch.load(path, weights_only=True)
            model = ReverseJAKET(10, 2, tiny_config(), lm=tiny_lm())
            _warm_start(model, path, state["manifest"])
            for name, value in state["model_state"].items():
                self.assertTrue(torch.equal(model.state_dict()[name], value), name)
            wrong = ReverseJAKET(10, 2, tiny_config(kg_dim=16), lm=tiny_lm())
            with self.assertRaisesRegex(ValueError, "kg_dim"):
                _warm_start(wrong, path, state["manifest"])
            with self.assertRaisesRegex(ValueError, "No completed eligible"):
                summarize(tmp, Path(tmp) / "report", plots=False)
            rows, summary = summarize(tmp, Path(tmp) / "report", include_synthetic=True, plots=True)
            self.assertEqual(len(rows), 1)
            self.assertIsNone(summary[0]["test_MRR_std"])
            self.assertTrue((Path(tmp) / "report" / "test_mrr.png").exists())

    def test_width_and_prepended_soft_prompt(self):
        cfg = tiny_config(kg_dim=32)
        model = ReverseJAKET(10, 2, cfg, lm=tiny_lm()).eval()
        self.assertEqual(model.encoder1.num_blocks, 8)
        self.assertEqual(model.scorer.relation_emb.weight.shape, (2, 32))
        tokens = tiny_tokens()
        prompt = torch.randn(2, 16)
        wrapper = model.bridge.lm
        with patch.object(wrapper.lm, "forward", wraps=wrapper.lm.forward) as forward:
            wrapper(prompt, tokens.input_ids[:2], tokens.attention_mask[:2])
            actual = forward.call_args.kwargs["inputs_embeds"]
            raw = wrapper.lm.get_input_embeddings()(tokens.input_ids[:2])
            self.assertEqual(actual.shape[1], raw.shape[1] + 1)
            torch.testing.assert_close(actual[:, 0], prompt)
            torch.testing.assert_close(actual[:, 1:], raw)
            mask = forward.call_args.kwargs["attention_mask"]
            torch.testing.assert_close(mask[:, 1:], tokens.attention_mask[:2])
            self.assertTrue((mask[:, 0] == 1).all())
        with torch.no_grad():
            result = model.encode_local(torch.arange(10), torch.tensor([[0, 1], [1, 0]]),
                                         torch.tensor([0, 2]), tokens, use_text=True)
        self.assertEqual(result.entities.shape, (10, 32))
        self.assertEqual(result.relations.shape, (2, 32))
        self.assertTrue(torch.isfinite(result.entities).all())
        self.assertTrue(torch.isfinite(result.relations).all())

    def test_plain_bce_matches_binary_classification_and_gradients(self):
        pos = torch.tensor([1.2, -0.4], requires_grad=True)
        neg = torch.tensor([[0.2, -2.0], [1.1, -0.1]], requires_grad=True)
        scores = torch.cat([pos, neg.flatten()])
        labels = torch.tensor([1., 1., 0., 0., 0., 0.])
        reference = torch.nn.functional.binary_cross_entropy_with_logits(scores, labels)
        loss = bce_loss(pos, neg)
        torch.testing.assert_close(loss, reference)
        loss.backward()
        self.assertTrue((pos.grad < 0).all())
        self.assertTrue((neg.grad > 0).all())

    def test_direct_projection_enters_encoder2_for_both_kinds(self):
        model = ReverseJAKET(10, 2, tiny_config(), lm=tiny_lm()).eval()
        tokens = tiny_tokens()
        projected, received = [], []
        bridge_hook = model.bridge.register_forward_hook(lambda m, args, output: projected.append(output))
        encoder_hook = model.encoder2.register_forward_pre_hook(
            lambda m, args, kwargs: received.append((args[0], kwargs["relations"])), with_kwargs=True)
        try:
            output = model.encode_local(torch.arange(10), torch.tensor([[0, 1], [1, 0]]),
                                        torch.tensor([0, 2]), tokens, use_text=True)
            self.assertEqual(len(projected), 2)
            self.assertIs(received[0][0], projected[0])
            self.assertIs(received[0][1], projected[1])
            grads = torch.autograd.grad(output.entities.square().mean() + output.relations[:, 0].sum(),
                                        projected)
            self.assertTrue(all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads))
            self.assertFalse(any("gate" in name for name, _ in model.named_parameters()))
        finally:
            bridge_hook.remove()
            encoder_hook.remove()

    def test_relation_text_changes_scores_and_evaluation_uses_encoded_relations(self):
        model = ReverseJAKET(10, 2, tiny_config(), lm=tiny_lm()).eval()
        tokens, changed = tiny_tokens(), tiny_tokens()
        changed.relations.input_ids[:, 1] = torch.tensor([30, 31])
        edges, types = torch.tensor([[0, 1, 2], [1, 2, 0]]), torch.tensor([0, 1, 2])
        triples = torch.tensor([[0, 0, 1], [1, 1, 2]])
        with torch.no_grad():
            a = model.encode_full(edges, types, tokens, True)
            b = model.encode_full(edges, types, changed, True)
            self.assertGreater((a.relations - b.relations).abs().max().item(), 0)
            self.assertGreater((model.score_triples(a, triples) - model.score_triples(b, triples)).abs().max().item(), 0)
            manual = model.scorer.score(a.entities[triples[:, 0]], triples[:, 1],
                                        a.entities[triples[:, 2]], relations=a.relations)
            torch.testing.assert_close(model.score_triples(a, triples), manual)
            filters = build_filter_index(triples.tolist())
            with patch.object(model.scorer, "score_all_tails", wraps=model.scorer.score_all_tails) as score:
                evaluate_filtered(model, a, triples.tolist(), *filters)
                self.assertIs(score.call_args.kwargs["relations"], a.relations)

    def test_relation_alignment_missing_text_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "relation2text.txt"
            path.write_text("/r/b\tsecond description\n/r/a\tfirst description\n", encoding="utf-8")
            self.assertEqual(align_relation_texts({"/r/a": 0, "/r/b": 1}, path),
                             ["first description", "second description"])
            with self.assertRaisesRegex(ValueError, "Missing relation descriptions"):
                align_relation_texts({"/r/c": 0}, path)
        tokens, changed = tiny_tokens(), tiny_tokens()
        changed.relations.input_ids[0, 1] += 1
        first = dataset_manifest(tiny_config(), tiny_dataset(), tokens)
        second = dataset_manifest(tiny_config(), tiny_dataset(), changed)
        self.assertNotEqual(first["relation_tokens_sha256"], second["relation_tokens_sha256"])
        tokens.relations = None
        with self.assertRaisesRegex(ValueError, "Relation text tokens"):
            tokens.validate(10, 2)

    def test_preflight_exercises_semantic_step_offline(self):
        import contextlib
        import io
        import run_colab_preflight
        cfg = tiny_config()
        output = io.StringIO()
        def make_model(num_entities, num_relations, settings):
            return ReverseJAKET(num_entities, num_relations, settings, lm=tiny_lm())
        with patch("sys.argv", ["run_colab_preflight.py"]), \
             patch.object(run_colab_preflight, "load_config", return_value=cfg), \
             patch.object(run_colab_preflight, "load_experiment_dataset", return_value=tiny_dataset()), \
             patch.object(run_colab_preflight, "prepare_text_tokens", return_value=tiny_tokens()), \
             patch.object(run_colab_preflight, "ReverseJAKET", side_effect=make_model), \
             contextlib.redirect_stdout(output):
            run_colab_preflight.main()
        result = json.loads(output.getvalue())
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["relations_with_text"], 2)
        self.assertIn("bridge.kg_to_lm", result["gradient_paths"])

    def test_comparison_plan_keeps_controls_matched(self):
        from run_comparisons import comparison_plan
        plan = comparison_plan(ExperimentConfig(), "unused", [42, 43, 44], [128, 200, 256])
        self.assertEqual(len(plan), 30)
        configs = dict(plan)
        a, b = configs["baseline_complex_seed42"], configs["proposed_complex_seed42"]
        differences = {k for k in a.to_dict() if a.to_dict()[k] != b.to_dict()[k]}
        self.assertEqual(differences, {"variant", "output_dir"})
        for _, config in plan:
            self.assertEqual(config.dataset, "fb15k237")
            self.assertEqual(config.scorer, "complex")
        with self.assertRaisesRegex(ValueError, "Only the ComplEx scorer"):
            ExperimentConfig(scorer="unsupported").validate()

    def test_semantic_early_stopping_waits_for_warmup_and_resume_rejects_changed_text(self):
        metrics = {"MRR": 0.5, "Hits@1": 0.25, "Hits@3": 0.5, "Hits@10": 1.0,
                   "Head_MRR": 0.5, "Tail_MRR": 0.5}
        with tempfile.TemporaryDirectory() as tmp:
            cfg = tiny_config(str(Path(tmp) / "run"), max_epochs=7, warmup_epochs=2,
                               early_stopping_patience=1)
            with patch("training.train_pipeline.evaluate_filtered", return_value=metrics):
                result = run_experiment(cfg, dataset=tiny_dataset(), tokens=tiny_tokens(), lm=tiny_lm())
            self.assertEqual(result["results"]["best_epoch"], 3)
            self.assertEqual(result["results"]["epochs_completed"], 4)
            self.assertEqual(result["results"]["best_semantic_enabled"], True)
            checkpoint = Path(cfg.output_dir) / "last_checkpoint.pt"
            changed = tiny_tokens()
            changed.input_ids[0, 1] += 1
            with self.assertRaisesRegex(ValueError, "tokenized text changed"):
                run_experiment(cfg, resume=checkpoint, dataset=tiny_dataset(), tokens=changed, lm=tiny_lm())
            changed_relations = tiny_tokens()
            changed_relations.relations.input_ids[0, 1] += 1
            with self.assertRaisesRegex(ValueError, "tokenized text changed"):
                run_experiment(cfg, resume=checkpoint, dataset=tiny_dataset(), tokens=changed_relations, lm=tiny_lm())
            # A directory containing stale final_results must be ignored while
            # an interrupted continuation is marked running.
            from summarize_results import collect_results
            (Path(cfg.output_dir) / "run_status.json").write_text('{"status": "running"}')
            self.assertEqual(collect_results(tmp, include_synthetic=True), [])


if __name__ == "__main__":
    unittest.main()
