# Incident Response Runbook

Owner: Site Reliability Engineering (#sre). This runbook defines how Northwind Cloud detects, manages and learns from production incidents.

## Severity levels

- **SEV1**: complete outage of a customer-facing service, data loss, or a confirmed security breach. Paging is immediate and the on-call engineer must acknowledge within 5 minutes.
- **SEV2**: major degradation affecting many customers, or a full outage of a non-critical service. Acknowledge within 15 minutes.
- **SEV3**: minor degradation with a workaround available. Handle within business hours.
- **SEV4**: cosmetic issues or internal tooling problems. Track as a normal ticket.

Any engineer can declare an incident. When in doubt, declare at the higher severity; downgrading later is cheap.

## Roles

Every SEV1 and SEV2 incident has an Incident Commander (IC), a Communications Lead and subject-matter responders. The IC coordinates and makes decisions but does not debug. The primary on-call engineer is IC until they hand the role over explicitly.

## Communication

Open a dedicated Slack channel named `#inc-YYYYMMDD-short-name`. For SEV1 incidents the Communications Lead updates the public status page at least every 30 minutes, and every 60 minutes for SEV2. Customer Support must be told within 15 minutes of declaring a SEV1.

Security incidents follow this runbook and additionally involve the Security on-call. Possible personal data breaches must be escalated to Legal within 24 hours so regulatory deadlines can be met.

## After the incident

A blameless postmortem is required for every SEV1 and SEV2 incident. The draft is due within 5 business days of resolution and is reviewed at the weekly reliability review. Action items are tracked as tickets with owners and due dates; SEV1 action items must be completed within 30 days.
