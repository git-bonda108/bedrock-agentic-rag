"""CDK assertions: synthesizes the stacks (no AWS calls) and checks the security and
wiring properties that matter most."""

import sys
from pathlib import Path

import aws_cdk as cdk
import pytest
from aws_cdk.assertions import Match, Template

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "infra"))

from stacks.agentic_rag_stack import AgenticRagStack
from stacks.github_oidc_stack import GithubOidcStack


@pytest.fixture(scope="module")
def template(tmp_path_factory) -> Template:
    code = tmp_path_factory.mktemp("ingest")
    (code / "placeholder.py").write_text("")
    app = cdk.App()
    stack = AgenticRagStack(
        app,
        "TestStack",
        answer_model_id="anthropic.claude-opus-5-5",
        judge_model_id="anthropic.claude-haiku-4-5",
        rerank_model_id="amazon.rerank-v1:0",
        ingest_code_path=str(code),
        env=cdk.Environment(account="123456789012", region="us-west-2"),
    )
    return Template.from_stack(stack)


def test_vector_index_matches_titan_v2(template):
    template.has_resource_properties(
        "AWS::S3Vectors::Index",
        {
            "DataType": "float32",
            "Dimension": 1024,
            "DistanceMetric": "cosine",
            "MetadataConfiguration": {"NonFilterableMetadataKeys": ["text", "title", "section"]},
        },
    )


def test_ingest_queue_has_dlq_and_partial_batch_failures(template):
    template.has_resource_properties(
        "AWS::SQS::Queue",
        {"RedrivePolicy": {"maxReceiveCount": 3, "deadLetterTargetArn": Match.any_value()}},
    )
    template.has_resource_properties(
        "AWS::Lambda::EventSourceMapping",
        {"FunctionResponseTypes": ["ReportBatchItemFailures"], "BatchSize": 5},
    )
    template.has_resource_properties(
        "AWS::Lambda::Function",
        {"Architectures": ["arm64"], "Runtime": "python3.12", "TracingConfig": {"Mode": "Active"}},
    )


def test_documents_bucket_is_private_and_versioned(template):
    template.has_resource_properties(
        "AWS::S3::Bucket",
        {
            "PublicAccessBlockConfiguration": {
                "BlockPublicAcls": True,
                "BlockPublicPolicy": True,
                "IgnorePublicAcls": True,
                "RestrictPublicBuckets": True,
            },
            "VersioningConfiguration": {"Status": "Enabled"},
        },
    )


def test_checkpoint_table_matches_dynamodb_saver_schema(template):
    template.has_resource_properties(
        "AWS::DynamoDB::Table",
        {
            "KeySchema": [
                {"AttributeName": "PK", "KeyType": "HASH"},
                {"AttributeName": "SK", "KeyType": "RANGE"},
            ],
            "TimeToLiveSpecification": {"AttributeName": "ttl", "Enabled": True},
        },
    )


def test_alb_is_not_open_to_the_internet(template):
    for sg in template.find_resources("AWS::EC2::SecurityGroup").values():
        for rule in sg["Properties"].get("SecurityGroupIngress", []):
            assert rule.get("CidrIp") != "0.0.0.0/0"


def test_cloudfront_is_https_only_and_uncached(template):
    template.has_resource_properties(
        "AWS::CloudFront::Distribution",
        {
            "DistributionConfig": Match.object_like(
                {"DefaultCacheBehavior": Match.object_like({"ViewerProtocolPolicy": "https-only"})}
            )
        },
    )


def test_alarms_exist(template):
    template.resource_count_is("AWS::CloudWatch::Alarm", 5)


def test_oidc_role_is_scoped_to_main_branch():
    app = cdk.App()
    stack = GithubOidcStack(app, "Oidc", github_repo="owner/repo")
    Template.from_stack(stack).has_resource_properties(
        "AWS::IAM::Role",
        {
            "AssumeRolePolicyDocument": {
                "Statement": [
                    Match.object_like(
                        {
                            "Condition": {
                                "StringEquals": Match.any_value(),
                                "StringLike": {
                                    "token.actions.githubusercontent.com:sub": [
                                        "repo:owner/repo:ref:refs/heads/main",
                                        "repo:owner/repo:environment:production",
                                    ]
                                },
                            }
                        }
                    )
                ]
            }
        },
    )
