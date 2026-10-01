# Data Classification and Retention Policy

Owner: Security Engineering and Legal (#privacy).

## Classification levels

- **Public**: approved for publication, such as marketing pages and public docs.
- **Internal**: default for company information not meant for the public.
- **Confidential**: customer contracts, financials, employee records.
- **Restricted**: customer production data, credentials, encryption keys and any personal data.

Restricted data must be encrypted at rest and in transit and may only be processed in approved production accounts.

## Use of AI tools

Restricted and Confidential data must never be pasted into public AI tools. It may only be sent to AI models running in Northwind Cloud's approved Amazon Bedrock accounts, which do not use inputs for model training.

## Retention periods

- Customer data is deleted within 30 days of contract termination, unless the contract states otherwise.
- Database backups are retained for 35 days.
- Application and audit logs are retained for 13 months.
- Employee records are retained for 7 years after employment ends.

## Deletion requests

Data subject requests (access or deletion) received from customers' end users must be completed within 30 days. Forward them to privacy@northwind.cloud immediately; do not delete data yourself.
