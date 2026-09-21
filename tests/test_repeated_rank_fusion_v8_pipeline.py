from dta.repeated_rank_fusion_v8.run_pipeline import (
    build_arg_parser,
    test_improves_v7,
)


def test_v8_gate_requires_strict_ci_improvement_over_v7():
    baseline = {"CI": 0.7163453001243019}
    assert test_improves_v7({"CI": 0.717}, baseline)
    assert not test_improves_v7({"CI": baseline["CI"]}, baseline)
    assert not test_improves_v7({"CI": 0.71}, baseline)


def test_v8_parser_exposes_repeat_and_resume_controls():
    destinations = {action.dest for action in build_arg_parser()._actions}
    assert {
        "v7_root",
        "maximum_epochs",
        "maximum_pairs",
        "limit",
        "smoke",
        "resume",
    }.issubset(destinations)
