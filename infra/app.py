#!/usr/bin/env python3
"""CDK entry point.

npx aws-cdk deploy AgenticRagStack          # the application
npx aws-cdk deploy AgenticRagGithubOidc     # once: the role GitHub Actions deploys with
"""

import os
import sys
from pathlib import Path

import aws_cdk as cdk

sys.path.insert(0, str(Path(__file__).resolve().parent))

from stacks.agentic_rag_stack import AgenticRagStack
from stacks.github_oidc_stack import GithubOidcStack

app = cdk.App()
env = cdk.Environment(
    account=os.getenv("CDK_DEFAULT_ACCOUNT"),
    region=os.getenv("CDK_DEFAULT_REGION", "us-west-2"),
)

AgenticRagStack(
    app,
    "AgenticRagStack",
    env=env,
    answer_model_id=app.node.try_get_context("answerModelId"),
    judge_model_id=app.node.try_get_context("judgeModelId"),
    rerank_model_id=app.node.try_get_context("rerankModelId"),
    alarm_email=app.node.try_get_context("alarmEmail") or None,
    description="Agentic RAG on AWS: LangGraph + Bedrock, S3 Vectors, Lambda ingestion, ECS API",
)
GithubOidcStack(
    app,
    "AgenticRagGithubOidc",
    env=env,
    github_repo=app.node.try_get_context("githubRepo"),
    create_provider=app.node.try_get_context("createOidcProvider") != "false",
)

cdk.Tags.of(app).add("project", "bedrock-agentic-rag")
app.synth()
