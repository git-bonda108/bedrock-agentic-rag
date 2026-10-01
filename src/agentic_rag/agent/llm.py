"""Model-facing interfaces for the agent, with Amazon Bedrock implementations.

The graph depends on the two small protocols below rather than on chat models directly,
so tests can run the full control flow with deterministic fakes.

Model split:
- Answer model (Claude Opus 5.5): writes the cited answer. Its thinking is always on and it
  rejects sampling parameters and forced tool choice, so it is called with plain messages.
- Judge model (Claude Haiku 4.5): fast structured decisions (routing, grading, verification)
  through `with_structured_output`, at temperature 0.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, Protocol

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from agentic_rag.agent import prompts

Route = Literal["knowledge_base", "conversational", "out_of_scope"]


class RouteDecision(BaseModel):
    route: Route
    standalone_query: str = Field(
        default="", description="Standalone search query; empty unless route is knowledge_base"
    )


class GradeDecision(BaseModel):
    relevant_passages: list[int] = Field(description="1-based numbers of relevant passages")


class VerifyDecision(BaseModel):
    grounded: bool
    unsupported_claims: list[str] = Field(default_factory=list)


class Judge(Protocol):
    def route(self, question: str, history: Sequence[BaseMessage]) -> RouteDecision: ...

    def grade(self, query: str, passages: Sequence[str]) -> list[int]: ...

    def rewrite(self, query: str) -> str: ...

    def verify(self, answer: str, passages: Sequence[str]) -> VerifyDecision: ...


class Generator(Protocol):
    def answer(
        self,
        question: str,
        passages: Sequence[str],
        history: Sequence[BaseMessage],
        feedback: str | None = None,
    ) -> str: ...

    def converse(self, question: str, history: Sequence[BaseMessage]) -> str: ...


def number_passages(passages: Sequence[str]) -> str:
    return "\n\n".join(f"[{i}] {p}" for i, p in enumerate(passages, start=1))


def message_text(msg: BaseMessage) -> str:
    """Text content only; drops reasoning/thinking blocks returned by Bedrock Converse."""
    if isinstance(msg.content, str):
        return msg.content
    return "".join(
        block.get("text", "")
        for block in msg.content
        if isinstance(block, dict) and block.get("type") == "text"
    )


def _history_text(history: Sequence[BaseMessage], limit: int = 6) -> str:
    lines = []
    for msg in list(history)[-limit:]:
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        lines.append(f"{role}: {message_text(msg)}")
    return "\n".join(lines) or "(none)"


class BedrockJudge:
    def __init__(self, chat_model: Any) -> None:
        self._model = chat_model
        self._router = chat_model.with_structured_output(RouteDecision)
        self._grader = chat_model.with_structured_output(GradeDecision)
        self._verifier = chat_model.with_structured_output(VerifyDecision)

    def route(self, question: str, history: Sequence[BaseMessage]) -> RouteDecision:
        return self._router.invoke(
            [
                SystemMessage(prompts.ROUTER_SYSTEM),
                HumanMessage(
                    f"Conversation so far:\n{_history_text(history)}\n\nLatest message: {question}"
                ),
            ]
        )

    def grade(self, query: str, passages: Sequence[str]) -> list[int]:
        decision: GradeDecision = self._grader.invoke(
            [
                SystemMessage(prompts.GRADER_SYSTEM),
                HumanMessage(f"Question: {query}\n\nPassages:\n{number_passages(passages)}"),
            ]
        )
        return sorted({i for i in decision.relevant_passages if 1 <= i <= len(passages)})

    def rewrite(self, query: str) -> str:
        reply = self._model.invoke(
            [SystemMessage(prompts.REWRITE_SYSTEM), HumanMessage(f"Query: {query}")]
        )
        return message_text(reply).strip().strip('"') or query

    def verify(self, answer: str, passages: Sequence[str]) -> VerifyDecision:
        return self._verifier.invoke(
            [
                SystemMessage(prompts.VERIFIER_SYSTEM),
                HumanMessage(f"Passages:\n{number_passages(passages)}\n\nAnswer:\n{answer}"),
            ]
        )


class BedrockGenerator:
    def __init__(self, chat_model: Any) -> None:
        self._model = chat_model

    def answer(
        self,
        question: str,
        passages: Sequence[str],
        history: Sequence[BaseMessage],
        feedback: str | None = None,
    ) -> str:
        prompt = f"Handbook passages:\n{number_passages(passages)}\n\nQuestion: {question}"
        if feedback:
            prompt += (
                "\n\nA reviewer found these statements in your previous draft were not "
                f"supported by the passages. Remove or correct them:\n{feedback}"
            )
        messages: list[BaseMessage] = [
            SystemMessage(prompts.ANSWER_SYSTEM),
            *list(history)[-6:],
            HumanMessage(prompt),
        ]
        return message_text(self._model.invoke(messages)).strip()

    def converse(self, question: str, history: Sequence[BaseMessage]) -> str:
        messages: list[BaseMessage] = [
            SystemMessage(prompts.CONVERSATIONAL_SYSTEM),
            *list(history)[-6:],
            HumanMessage(question),
        ]
        return message_text(self._model.invoke(messages)).strip()


def build_bedrock_models(settings: Any) -> tuple[BedrockJudge, BedrockGenerator]:
    from langchain_aws import ChatBedrockConverse

    judge_model = ChatBedrockConverse(
        model=settings.judge_model_id,
        region_name=settings.region,
        temperature=0,
        max_tokens=1024,
    )
    # No temperature: Claude Opus 5.5 rejects sampling parameters.
    answer_model = ChatBedrockConverse(
        model=settings.answer_model_id,
        region_name=settings.region,
        max_tokens=4096,
    )
    return BedrockJudge(judge_model), BedrockGenerator(answer_model)
