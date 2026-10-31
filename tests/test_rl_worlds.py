import pytest

pytest.importorskip("cvtestbed")
modal = pytest.importorskip("modal")


def test_prompts_and_questions_for_any_world():
    from lvpm.rl import worlds as W

    p = W.prompt(100)
    assert p[0]["role"] == "system" and "LAB NOTEBOOK" in p[1]["content"]
    qs = W.questions(100, 16)
    assert len(qs) == 16 and all(0 < q["p_yes"] < 1 for q in qs) and not any(q["shifted"] for q in qs)
    assert W.questions(100, 16) == qs  # deterministic


def test_parse_report():
    from lvpm.rl.grpo_modal import parse_report

    t, w = parse_report('<think>x</think> {"hypotheses": [{"text": "a", "weight": 3}, {"text": "b", "weight": 1}]}')
    assert t == ["a", "b"] and w == [0.75, 0.25]
    assert parse_report("no json") is None
