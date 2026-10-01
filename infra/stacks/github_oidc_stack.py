"""IAM role that GitHub Actions assumes through OIDC: no long-lived AWS keys in GitHub.

The role can only (1) assume the CDK bootstrap roles, which do the actual deployment,
(2) read the stack outputs, (3) seed the sample corpus, and (4) read the API key to run
the post-deploy eval. Only workflows on main of the configured repository can assume it.
"""

from __future__ import annotations

from aws_cdk import Aws, CfnOutput, Duration, Stack
from aws_cdk import aws_iam as iam
from constructs import Construct

GITHUB_OIDC_URL = "token.actions.githubusercontent.com"


class GithubOidcStack(Stack):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        github_repo: str,
        create_provider: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        if create_provider:
            provider = iam.OpenIdConnectProvider(
                self,
                "GithubProvider",
                url=f"https://{GITHUB_OIDC_URL}",
                client_ids=["sts.amazonaws.com"],
            )
        else:  # an account can hold only one provider per URL
            provider = iam.OpenIdConnectProvider.from_open_id_connect_provider_arn(
                self,
                "GithubProvider",
                f"arn:{Aws.PARTITION}:iam::{Aws.ACCOUNT_ID}:oidc-provider/{GITHUB_OIDC_URL}",
            )

        role = iam.Role(
            self,
            "DeployRole",
            role_name="agentic-rag-github-deploy",
            max_session_duration=Duration.hours(1),
            assumed_by=iam.WebIdentityPrincipal(
                provider.open_id_connect_provider_arn,
                conditions={
                    "StringEquals": {f"{GITHUB_OIDC_URL}:aud": "sts.amazonaws.com"},
                    "StringLike": {
                        f"{GITHUB_OIDC_URL}:sub": [
                            f"repo:{github_repo}:ref:refs/heads/main",
                            f"repo:{github_repo}:environment:production",
                        ]
                    },
                },
            ),
        )
        account, region, partition = Aws.ACCOUNT_ID, Aws.REGION, Aws.PARTITION
        role.add_to_policy(
            iam.PolicyStatement(
                actions=["sts:AssumeRole"],
                resources=[f"arn:{partition}:iam::{account}:role/cdk-*"],
            )
        )
        role.add_to_policy(
            iam.PolicyStatement(
                actions=["cloudformation:DescribeStacks"],
                resources=[
                    f"arn:{partition}:cloudformation:{region}:{account}:stack/AgenticRagStack/*"
                ],
            )
        )
        role.add_to_policy(
            iam.PolicyStatement(
                actions=["secretsmanager:GetSecretValue"],
                resources=[
                    f"arn:{partition}:secretsmanager:{region}:{account}:secret:agentic-rag/*"
                ],
            )
        )
        role.add_to_policy(
            iam.PolicyStatement(
                actions=["s3:ListBucket", "s3:PutObject", "s3:DeleteObject"],
                resources=[
                    f"arn:{partition}:s3:::agenticragstack-documentsbucket*",
                    f"arn:{partition}:s3:::agenticragstack-documentsbucket*/uploads/corpus/*",
                ],
            )
        )

        CfnOutput(self, "DeployRoleArn", value=role.role_arn)
