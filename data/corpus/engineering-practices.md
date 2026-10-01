# Engineering Practices

Owner: Engineering Leadership (#eng-practices).

## Code review

Every change requires at least one approving review before merge. Changes to infrastructure code, authentication, or anything touching Restricted data require two approvals, one of them from a code owner.

## Testing

New code must reach at least 80% line coverage, and every bug fix includes a regression test. Flaky tests are quarantined within 24 hours and fixed or deleted within 2 weeks.

## Deployments

Production deploys go through the CI/CD pipeline only; manual changes in the AWS console are not allowed outside break-glass incidents. Risky changes ship behind feature flags.

There is a deploy freeze every Friday from 15:00 local time until Monday, and a company-wide change freeze from December 20 to January 2. Exceptions need approval from the VP of Engineering.

## Design documents

Changes expected to take more than 2 weeks of engineering effort need a written RFC, reviewed by at least two senior engineers from outside the authoring team.
