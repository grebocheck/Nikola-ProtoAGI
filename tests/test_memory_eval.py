import unittest

from protoagi.memory_eval import EvalFact, EvalQuery, build_eval_service, evaluate, load_corpus


class MemoryEvalTests(unittest.TestCase):
    def test_default_corpus_loads(self) -> None:
        facts, queries = load_corpus()
        self.assertGreater(len(facts), 5)
        self.assertGreater(len(queries), 5)

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

    def test_default_corpus_has_reasonable_recall(self) -> None:
        facts, queries = load_corpus()
        _, service = build_eval_service(facts)
        report = evaluate(queries, service, k_values=(3, 5))
        # FTS-only is enough to retrieve at least half the queries within
        # top-5 from the bundled corpus.
        self.assertGreaterEqual(report.recall_at_k[5], 0.5)


if __name__ == "__main__":
    unittest.main()
