.PHONY: install lint test eval build-lambda synth bootstrap deploy seed eval-live destroy

CDK := npx --yes aws-cdk@2

install:
	uv venv --python 3.12 && uv pip install -e ".[dev]"

lint:
	ruff check . && ruff format --check .

test:
	pytest --cov=agentic_rag --cov-report=term-missing

eval:
	python -m evals.run_evals offline

build-lambda:
	./scripts/build_lambda.sh

synth: build-lambda
	$(CDK) synth --quiet

bootstrap:
	$(CDK) bootstrap

deploy: build-lambda
	$(CDK) deploy AgenticRagStack --require-approval never --outputs-file cdk.out/outputs.json

seed:
	python scripts/ops.py seed

eval-live:
	@eval "$$(python scripts/ops.py outputs)"; \
	API_KEY=$$(aws secretsmanager get-secret-value --secret-id $$API_KEY_SECRET_ARN --query SecretString --output text) \
	API_URL=$$API_URL python -m evals.run_evals live

destroy:
	$(CDK) destroy AgenticRagStack
