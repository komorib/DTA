from dta.rank_unlocked_fusion_v7.run_pipeline import (
    build_arg_parser,
    test_improves_v5,
)


def test_official_gate_requires_strict_ci_improvement():
    baseline = {"CI": 0.7086996133}
    assert test_improves_v5({"CI": 0.7088}, baseline)
    assert not test_improves_v5({"CI": baseline["CI"]}, baseline)
    assert not test_improves_v5({"CI": 0.70}, baseline)


def test_parser_exposes_smoke_and_resume_controls():
    parser = build_arg_parser()
    destinations = {action.dest for action in parser._actions}
    assert {"limit", "smoke", "resume", "maximum_epochs"}.issubset(destinations)
