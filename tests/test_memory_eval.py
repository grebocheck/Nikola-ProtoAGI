import unittest

from protoagi.evals.memory import EvalFact, EvalQuery, build_eval_service, evaluate, load_corpus


class MemoryEvalTests(unittest.TestCase):
    def test_default_corpus_loads(self) -> None:
        facts, queries = load_corpus()
        self.assertGreater(len(facts), 5)
        self.assertGreater(len(queries), 5)
        sections = {query.section for query in queries}
        self.assertIn("contradictions", sections)
        self.assertIn("negative", sections)
        self.assertIn("multimodal_caption", sections)

    def test_evaluate_recall_at_k(self) -> None:
        facts = [
            EvalFact(text="користувач любить чорний чай зранку"),
            EvalFact(text="користувач не любить, коли запитують про погоду"),
        ]
        queries = [
            EvalQuery(query="чай", expected_substrings=["чорний чай"]),
            EvalQuery(query="не існує", expected_substrings=["нічого"]),
        ]
        _, service = build_eval_service(facts)
        report = evaluate(queries, service, k_values=(1, 3))
        self.assertEqual(len(report.queries), 2)
        self.assertGreaterEqual(report.recall_at_k[1], 0.5)
        self.assertEqual(report.recall_at_k[1], 0.5)
        self.assertGreater(report.mrr, 0.0)
        self.assertIn("friendly", report.section_metrics)

    def test_default_corpus_has_reasonable_recall(self) -> None:
        facts, queries = load_corpus()
        _, service = build_eval_service(facts)
        report = evaluate(queries, service, k_values=(3, 5))
        # FTS-only is enough to retrieve at least half the queries within
        # top-5 from the bundled corpus.
        self.assertGreaterEqual(report.recall_at_k[5], 0.5)
        self.assertIn("contradictions", report.section_metrics)
        self.assertIn("negative", report.section_metrics)
        self.assertIn("multimodal_caption", report.section_metrics)

    def test_superseded_eval_fact_is_hidden(self) -> None:
        facts = [
            EvalFact(text="User used to prefer espresso at night"),
            EvalFact(
                text="User now prefers mint tea at night",
                supersedes=["used to prefer espresso"],
            ),
        ]
        queries = [
            EvalQuery(
                query="what does user prefer at night",
                expected_substrings=["mint tea"],
                section="contradictions",
            )
        ]
        _, service = build_eval_service(facts)
        report = evaluate(queries, service, k_values=(1,))
        self.assertEqual(report.recall_at_k[1], 1.0)
        self.assertEqual(report.section_metrics["contradictions"]["recall_at_k"]["1"], 1.0)


if __name__ == "__main__":
    unittest.main()
