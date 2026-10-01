from langgraph.checkpoint.memory import InMemorySaver

from agentic_rag.agent import AgentDeps, build_graph, prompts, run_turn
from tests.conftest import FakeGenerator, FakeJudge


def _graph(retriever, judge, generator=None):
    deps = AgentDeps(
        judge=judge,
        generator=generator or FakeGenerator(),
        retriever=retriever,
        max_query_rewrites=2,
        max_generation_attempts=2,
    )
    return build_graph(deps, checkpointer=InMemorySaver())


def test_happy_path_returns_cited_grounded_answer(retriever):
    result = run_turn(_graph(retriever, FakeJudge()), "What is the on-call stipend?", "t1")
    assert result["route"] == "knowledge_base"
    assert result["grounded"] is True
    assert result["trace"] == [
        "route:knowledge_base",
        "retrieve:2",
        "grade:2",
        "generate",
        "verify:ok",
        "finalize",
    ]
    assert result["citations"][0]["n"] == 1
    assert result["citations"][0]["source"].startswith("corpus/")


def test_irrelevant_results_trigger_rewrites_then_not_found(retriever):
    judge = FakeJudge(grades=[[], [], []])
    result = run_turn(_graph(retriever, judge), "What is the stock vesting schedule?", "t2")
    assert result["answer"] == prompts.NOT_FOUND_REPLY
    assert result["trace"].count("rewrite") == 2
    assert result["trace"][-1] == "not_found"
    assert judge.rewrites == [
        "What is the stock vesting schedule?",
        "What is the stock vesting schedule? policy",
    ]


def test_rewrite_recovers_when_second_search_is_relevant(retriever):
    judge = FakeJudge(grades=[[], [1]])
    result = run_turn(_graph(retriever, judge), "How much time off?", "t3")
    assert "rewrite" in result["trace"]
    assert result["trace"][-1] == "finalize"
    assert result["grounded"] is True


def test_unsupported_claims_are_fed_back_and_regenerated(retriever):
    generator = FakeGenerator()
    judge = FakeJudge(verdicts=[False, True])
    result = run_turn(_graph(retriever, judge, generator), "Meal limit abroad?", "t4")
    assert result["trace"].count("generate") == 2
    assert generator.feedback == [None, "- made-up claim"]
    assert result["grounded"] is True
    assert prompts.UNVERIFIED_CAVEAT not in result["answer"]


def test_persistently_ungrounded_answer_gets_caveat(retriever):
    judge = FakeJudge(verdicts=[False, False])
    result = run_turn(_graph(retriever, judge), "Meal limit abroad?", "t5")
    assert result["grounded"] is False
    assert result["answer"].endswith(prompts.UNVERIFIED_CAVEAT)
    assert "caveat" in result["trace"]


def test_out_of_scope_and_conversational_routes(retriever):
    out = run_turn(_graph(retriever, FakeJudge(route="out_of_scope")), "Capital of Peru?", "t6")
    assert out["answer"] == prompts.OUT_OF_SCOPE_REPLY
    assert out["trace"] == ["route:out_of_scope", "decline"]

    chat = run_turn(_graph(retriever, FakeJudge(route="conversational")), "thanks!", "t7")
    assert chat["trace"] == ["route:conversational", "converse"]


def test_history_is_persisted_per_thread(retriever):
    judge = FakeJudge()
    graph = _graph(retriever, judge)
    run_turn(graph, "What is the on-call stipend?", "same")
    run_turn(graph, "And for the secondary?", "same")
    run_turn(graph, "Unrelated new thread", "other")
    # history = prior user + assistant messages for the thread
    assert judge.route_histories == [0, 2, 0]
