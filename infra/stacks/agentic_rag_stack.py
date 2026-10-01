"""The application stack.

    CloudFront (HTTPS) -> ALB (CloudFront prefix list only) -> ECS Fargate: FastAPI + LangGraph
                                                                  | Bedrock: Claude, Titan, Rerank
                                                                  | S3 Vectors: QueryVectors
                                                                  | DynamoDB: checkpoints, registry
    S3 uploads/ -(ObjectCreated/Removed)-> SQS (+DLQ) -> Lambda ingest -> Titan -> S3 Vectors

Everything is pay-per-use except the Fargate task and the ALB.
"""

from __future__ import annotations

from pathlib import Path

from aws_cdk import Aws, CfnOutput, Duration, RemovalPolicy, Stack
from aws_cdk import aws_cloudfront as cloudfront
from aws_cdk import aws_cloudfront_origins as origins
from aws_cdk import aws_cloudwatch as cw
from aws_cdk import aws_cloudwatch_actions as cw_actions
from aws_cdk import aws_dynamodb as dynamodb
from aws_cdk import aws_ec2 as ec2
from aws_cdk import aws_ecs as ecs
from aws_cdk import aws_ecs_patterns as ecs_patterns
from aws_cdk import aws_elasticloadbalancingv2 as elbv2
from aws_cdk import aws_iam as iam
from aws_cdk import aws_lambda as lambda_
from aws_cdk import aws_lambda_event_sources as event_sources
from aws_cdk import aws_logs as logs
from aws_cdk import aws_s3 as s3
from aws_cdk import aws_s3_notifications as s3n
from aws_cdk import aws_s3vectors as s3vectors
from aws_cdk import aws_secretsmanager as secretsmanager
from aws_cdk import aws_sns as sns
from aws_cdk import aws_sns_subscriptions as subs
from aws_cdk import aws_sqs as sqs
from aws_cdk import custom_resources as cr
from constructs import Construct

REPO_ROOT = Path(__file__).resolve().parents[2]
EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v2:0"
EMBEDDING_DIMENSIONS = 1024
VECTOR_INDEX = "chunks"
UPLOAD_PREFIX = "uploads/"
TARGET_5XX = elbv2.HttpCodeTarget.TARGET_5XX_COUNT


class AgenticRagStack(Stack):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        answer_model_id: str,
        judge_model_id: str,
        rerank_model_id: str,
        alarm_email: str | None = None,
        ingest_code_path: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # ------------------------------------------------------------ storage
        documents_bucket = s3.Bucket(
            self,
            "DocumentsBucket",
            encryption=s3.BucketEncryption.S3_MANAGED,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            enforce_ssl=True,
            versioned=True,
            lifecycle_rules=[
                s3.LifecycleRule(noncurrent_version_expiration=Duration.days(30)),
                s3.LifecycleRule(abort_incomplete_multipart_upload_after=Duration.days(1)),
            ],
            cors=[
                s3.CorsRule(
                    allowed_methods=[s3.HttpMethods.PUT],
                    allowed_origins=["*"],
                    allowed_headers=["*"],
                    max_age=3000,
                )
            ],
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True,
        )

        vector_bucket_name = f"agentic-rag-vectors-{Aws.ACCOUNT_ID}"
        vector_bucket = s3vectors.CfnVectorBucket(
            self, "VectorBucket", vector_bucket_name=vector_bucket_name
        )
        vector_index = s3vectors.CfnIndex(
            self,
            "VectorIndex",
            vector_bucket_name=vector_bucket_name,
            index_name=VECTOR_INDEX,
            data_type="float32",
            dimension=EMBEDDING_DIMENSIONS,
            distance_metric="cosine",
            # Chunk text is returned with results but never filtered on, so it does not
            # count against the 2 KB filterable-metadata limit.
            metadata_configuration=s3vectors.CfnIndex.MetadataConfigurationProperty(
                non_filterable_metadata_keys=["text", "title", "section"]
            ),
        )
        vector_index.node.add_dependency(vector_bucket)

        documents_table = dynamodb.Table(
            self,
            "DocumentsTable",
            partition_key=dynamodb.Attribute(name="doc_id", type=dynamodb.AttributeType.STRING),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            point_in_time_recovery_specification=dynamodb.PointInTimeRecoverySpecification(
                point_in_time_recovery_enabled=True
            ),
            removal_policy=RemovalPolicy.DESTROY,
        )
        # Key schema required by langgraph-checkpoint-aws DynamoDBSaver.
        checkpoints_table = dynamodb.Table(
            self,
            "CheckpointsTable",
            partition_key=dynamodb.Attribute(name="PK", type=dynamodb.AttributeType.STRING),
            sort_key=dynamodb.Attribute(name="SK", type=dynamodb.AttributeType.STRING),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            time_to_live_attribute="ttl",
            removal_policy=RemovalPolicy.DESTROY,
        )

        api_key = secretsmanager.Secret(
            self,
            "ApiKey",
            secret_name="agentic-rag/api-key",  # noqa: S106 - a name, the value is generated
            description="x-api-key for the agentic RAG API",
            generate_secret_string=secretsmanager.SecretStringGenerator(
                exclude_punctuation=True, password_length=40
            ),
            removal_policy=RemovalPolicy.DESTROY,
        )

        common_env = {
            "DOCUMENTS_BUCKET": documents_bucket.bucket_name,
            "DOCUMENTS_TABLE": documents_table.table_name,
            "VECTOR_BUCKET": vector_bucket_name,
            "VECTOR_INDEX": VECTOR_INDEX,
            "VECTOR_STORE": "s3vectors",
            "EMBEDDINGS_PROVIDER": "bedrock",
            "EMBEDDING_MODEL_ID": EMBEDDING_MODEL_ID,
            "EMBEDDING_DIMENSIONS": str(EMBEDDING_DIMENSIONS),
        }

        def grant_vectors(principal: iam.IGrantable, actions: list[str]) -> None:
            principal.grant_principal.add_to_principal_policy(
                iam.PolicyStatement(
                    actions=[f"s3vectors:{a}" for a in actions],
                    resources=[vector_index.attr_index_arn],
                )
            )

        def grant_bedrock_invoke(principal: iam.IGrantable, models: list[str]) -> None:
            principal.grant_principal.add_to_principal_policy(
                iam.PolicyStatement(
                    actions=["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"],
                    resources=[
                        *[f"arn:{Aws.PARTITION}:bedrock:*::foundation-model/{m}" for m in models],
                        f"arn:{Aws.PARTITION}:bedrock:*:{Aws.ACCOUNT_ID}:inference-profile/*",
                    ],
                )
            )

        # ------------------------------------------------------------ ingestion
        dlq = sqs.Queue(
            self,
            "IngestDLQ",
            retention_period=Duration.days(14),
            encryption=sqs.QueueEncryption.SQS_MANAGED,
            enforce_ssl=True,
        )
        ingest_queue = sqs.Queue(
            self,
            "IngestQueue",
            visibility_timeout=Duration.minutes(30),  # 6x the Lambda timeout
            dead_letter_queue=sqs.DeadLetterQueue(queue=dlq, max_receive_count=3),
            encryption=sqs.QueueEncryption.SQS_MANAGED,
            enforce_ssl=True,
        )
        for event_type in (s3.EventType.OBJECT_CREATED, s3.EventType.OBJECT_REMOVED):
            documents_bucket.add_event_notification(
                event_type,
                s3n.SqsDestination(ingest_queue),
                s3.NotificationKeyFilter(prefix=UPLOAD_PREFIX),
            )

        ingest_fn = lambda_.Function(
            self,
            "IngestFunction",
            runtime=lambda_.Runtime.PYTHON_3_12,
            architecture=lambda_.Architecture.ARM_64,
            handler="agentic_rag.ingest.handler.handler",
            code=lambda_.Code.from_asset(ingest_code_path or str(REPO_ROOT / "build" / "ingest")),
            memory_size=1024,
            timeout=Duration.minutes(5),
            # Caps concurrent Bedrock embedding traffic; SQS buffers bursts.
            reserved_concurrent_executions=5,
            tracing=lambda_.Tracing.ACTIVE,
            log_group=logs.LogGroup(
                self,
                "IngestLogs",
                retention=logs.RetentionDays.ONE_MONTH,
                removal_policy=RemovalPolicy.DESTROY,
            ),
            environment={
                **common_env,
                "POWERTOOLS_SERVICE_NAME": "ingest",
                "POWERTOOLS_METRICS_NAMESPACE": "AgenticRag",
                "CHUNK_SIZE": "1800",
                "CHUNK_OVERLAP": "250",
            },
        )
        ingest_fn.add_event_source(
            event_sources.SqsEventSource(
                ingest_queue,
                batch_size=5,
                max_batching_window=Duration.seconds(10),
                report_batch_item_failures=True,
            )
        )
        documents_bucket.grant_read(ingest_fn)
        documents_table.grant_read_write_data(ingest_fn)
        grant_vectors(ingest_fn, ["PutVectors", "DeleteVectors", "GetVectors"])
        grant_bedrock_invoke(ingest_fn, ["amazon.titan-embed-text-v2*"])

        # ------------------------------------------------------------ API service
        vpc = ec2.Vpc(
            self,
            "Vpc",
            max_azs=2,
            nat_gateways=0,  # tasks run in public subnets with public IPs: no NAT cost
            subnet_configuration=[
                ec2.SubnetConfiguration(name="public", subnet_type=ec2.SubnetType.PUBLIC)
            ],
        )
        vpc.add_gateway_endpoint("S3Endpoint", service=ec2.GatewayVpcEndpointAwsService.S3)
        vpc.add_gateway_endpoint(
            "DynamoEndpoint", service=ec2.GatewayVpcEndpointAwsService.DYNAMODB
        )
        cluster = ecs.Cluster(
            self, "Cluster", vpc=vpc, container_insights_v2=ecs.ContainerInsights.ENHANCED
        )

        api_logs = logs.LogGroup(
            self,
            "ApiLogs",
            retention=logs.RetentionDays.ONE_MONTH,
            removal_policy=RemovalPolicy.DESTROY,
        )
        service = ecs_patterns.ApplicationLoadBalancedFargateService(
            self,
            "Api",
            cluster=cluster,
            cpu=1024,
            memory_limit_mib=2048,
            desired_count=1,
            public_load_balancer=True,
            open_listener=False,
            assign_public_ip=True,
            task_subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PUBLIC),
            circuit_breaker=ecs.DeploymentCircuitBreaker(enable=True, rollback=True),
            min_healthy_percent=100,
            health_check_grace_period=Duration.seconds(60),
            runtime_platform=ecs.RuntimePlatform(
                cpu_architecture=ecs.CpuArchitecture.X86_64,
                operating_system_family=ecs.OperatingSystemFamily.LINUX,
            ),
            task_image_options=ecs_patterns.ApplicationLoadBalancedTaskImageOptions(
                image=ecs.ContainerImage.from_asset(str(REPO_ROOT), file="Dockerfile"),
                container_port=8000,
                log_driver=ecs.LogDrivers.aws_logs(stream_prefix="api", log_group=api_logs),
                environment={
                    **common_env,
                    "CHECKPOINTS_TABLE": checkpoints_table.table_name,
                    "API_KEY_SECRET_ARN": api_key.secret_arn,
                    "ANSWER_MODEL_ID": answer_model_id,
                    "JUDGE_MODEL_ID": judge_model_id,
                    "RERANK_MODEL_ID": rerank_model_id,
                    "RERANK_ENABLED": "true",
                },
            ),
        )
        service.target_group.configure_health_check(
            path="/healthz", healthy_http_codes="200", interval=Duration.seconds(15)
        )
        service.target_group.set_attribute("deregistration_delay.timeout_seconds", "30")
        # Agent turns can run for tens of seconds.
        service.load_balancer.set_attribute("idle_timeout.timeout_seconds", "120")

        scaling = service.service.auto_scale_task_count(min_capacity=1, max_capacity=4)
        scaling.scale_on_cpu_utilization("Cpu", target_utilization_percent=60)
        scaling.scale_on_request_count(
            "Requests", requests_per_target=40, target_group=service.target_group
        )

        task_role = service.task_definition.task_role
        documents_bucket.grant_read_write(task_role, f"{UPLOAD_PREFIX}*")
        documents_bucket.grant_delete(task_role, f"{UPLOAD_PREFIX}*")
        documents_table.grant_read_data(task_role)
        checkpoints_table.grant_read_write_data(task_role)
        api_key.grant_read(task_role)
        grant_vectors(task_role, ["QueryVectors", "GetVectors"])
        grant_bedrock_invoke(
            task_role,
            [
                "anthropic.claude-*",
                "amazon.titan-embed-text-v2*",
                "amazon.rerank-v1*",
                "cohere.rerank-*",
            ],
        )
        task_role.add_to_principal_policy(
            iam.PolicyStatement(actions=["bedrock:Rerank"], resources=["*"])
        )

        # Only CloudFront may reach the ALB: allow its managed origin-facing prefix list.
        prefix_list = cr.AwsCustomResource(
            self,
            "CloudFrontPrefixList",
            on_update=cr.AwsSdkCall(
                service="EC2",
                action="describeManagedPrefixLists",
                parameters={
                    "Filters": [
                        {
                            "Name": "prefix-list-name",
                            "Values": ["com.amazonaws.global.cloudfront.origin-facing"],
                        }
                    ]
                },
                physical_resource_id=cr.PhysicalResourceId.of("cloudfront-origin-facing"),
            ),
            policy=cr.AwsCustomResourcePolicy.from_sdk_calls(
                resources=cr.AwsCustomResourcePolicy.ANY_RESOURCE
            ),
            install_latest_aws_sdk=False,
        )
        service.load_balancer.connections.allow_from(
            ec2.Peer.prefix_list(prefix_list.get_response_field("PrefixLists.0.PrefixListId")),
            ec2.Port.tcp(80),
            "CloudFront origin-facing",
        )

        distribution = cloudfront.Distribution(
            self,
            "Distribution",
            comment="Agentic RAG API",
            price_class=cloudfront.PriceClass.PRICE_CLASS_100,
            default_behavior=cloudfront.BehaviorOptions(
                origin=origins.LoadBalancerV2Origin(
                    service.load_balancer,
                    protocol_policy=cloudfront.OriginProtocolPolicy.HTTP_ONLY,
                    read_timeout=Duration.seconds(60),
                ),
                viewer_protocol_policy=cloudfront.ViewerProtocolPolicy.HTTPS_ONLY,
                allowed_methods=cloudfront.AllowedMethods.ALLOW_ALL,
                cache_policy=cloudfront.CachePolicy.CACHING_DISABLED,
                origin_request_policy=cloudfront.OriginRequestPolicy.ALL_VIEWER,
            ),
        )

        # ------------------------------------------------------------ observability
        alarm_topic = sns.Topic(self, "AlarmTopic")
        if alarm_email:
            alarm_topic.add_subscription(subs.EmailSubscription(alarm_email))

        def api_metric(name: str, stat: str, label: str = "") -> cw.Metric:
            return cw.Metric(
                namespace="AgenticRag",
                metric_name=name,
                dimensions_map={"Service": "api"},
                statistic=stat,
                period=Duration.minutes(5),
                label=label or name,
            )

        alb = service.load_balancer.metrics
        alarms = [
            cw.Alarm(
                self,
                "IngestDlqNotEmpty",
                metric=dlq.metric_approximate_number_of_messages_visible(),
                threshold=0,
                comparison_operator=cw.ComparisonOperator.GREATER_THAN_THRESHOLD,
                evaluation_periods=1,
                alarm_description="Documents failed ingestion 3 times; inspect the DLQ.",
            ),
            cw.Alarm(
                self,
                "IngestErrors",
                metric=ingest_fn.metric_errors(period=Duration.minutes(5)),
                threshold=1,
                evaluation_periods=1,
                treat_missing_data=cw.TreatMissingData.NOT_BREACHING,
            ),
            cw.Alarm(
                self,
                "Api5xx",
                metric=alb.http_code_target(TARGET_5XX, period=Duration.minutes(5)),
                threshold=5,
                evaluation_periods=1,
                treat_missing_data=cw.TreatMissingData.NOT_BREACHING,
            ),
            cw.Alarm(
                self,
                "ChatLatencyP95",
                metric=api_metric("ChatLatencyMs", "p95"),
                threshold=30_000,
                evaluation_periods=3,
                treat_missing_data=cw.TreatMissingData.NOT_BREACHING,
                alarm_description="p95 agent turn latency above 30s for 15 minutes.",
            ),
            cw.Alarm(
                self,
                "GroundedRateLow",
                metric=api_metric("Grounded", "Average"),
                threshold=0.8,
                comparison_operator=cw.ComparisonOperator.LESS_THAN_THRESHOLD,
                evaluation_periods=3,
                treat_missing_data=cw.TreatMissingData.NOT_BREACHING,
                alarm_description="Under 80% of answers passed the groundedness check.",
            ),
        ]
        for alarm in alarms:
            alarm.add_alarm_action(cw_actions.SnsAction(alarm_topic))

        dashboard = cw.Dashboard(self, "Dashboard", dashboard_name="AgenticRag")
        dashboard.add_widgets(
            cw.GraphWidget(
                title="Agent turn latency (ms)",
                left=[
                    api_metric("ChatLatencyMs", "p50", "p50"),
                    api_metric("ChatLatencyMs", "p95", "p95"),
                ],
            ),
            cw.GraphWidget(
                title="Grounded answer rate",
                left=[api_metric("Grounded", "Average")],
                left_y_axis=cw.YAxisProps(min=0, max=1),
            ),
            cw.GraphWidget(
                title="API requests and 5xx",
                left=[alb.request_count(), alb.http_code_target(TARGET_5XX)],
            ),
        )
        dashboard.add_widgets(
            cw.GraphWidget(
                title="Ingestion",
                left=[
                    cw.Metric(
                        namespace="AgenticRag",
                        metric_name="ChunksIndexed",
                        dimensions_map={"service": "ingest"},
                        statistic="Sum",
                    ),
                    ingest_fn.metric_errors(),
                ],
                right=[ingest_queue.metric_approximate_number_of_messages_visible()],
            ),
            cw.GraphWidget(
                title="ECS CPU / memory",
                left=[
                    service.service.metric_cpu_utilization(),
                    service.service.metric_memory_utilization(),
                ],
            ),
            cw.AlarmStatusWidget(title="Alarms", alarms=alarms),
        )

        # ------------------------------------------------------------ outputs
        CfnOutput(self, "ApiUrl", value=f"https://{distribution.distribution_domain_name}")
        CfnOutput(self, "ApiKeySecretArn", value=api_key.secret_arn)
        CfnOutput(self, "DocumentsBucketName", value=documents_bucket.bucket_name)
        CfnOutput(self, "VectorBucketName", value=vector_bucket_name)
        CfnOutput(self, "IngestDlqUrl", value=dlq.queue_url)
        CfnOutput(self, "AlarmTopicArn", value=alarm_topic.topic_arn)
