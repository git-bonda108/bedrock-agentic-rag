# Information Security Policy

Owner: Security Engineering (#sec-help). Applies to all Northwind Cloud employees and contractors.

## Authentication

Multi-factor authentication (MFA) is mandatory for every company account. Production access, the AWS console and the source code host require a hardware security key (YubiKey); SMS codes are not accepted for any system.

All passwords must be stored in the company 1Password vault. Passwords must be at least 16 characters and unique per system. Sharing credentials, including in chat or tickets, is a policy violation.

## Production access

Standing production access is limited to the Site Reliability team. Other engineers request break-glass access through the `prod-access` workflow, which requires a linked incident or change ticket. Break-glass sessions expire automatically after 4 hours and every command is recorded.

Access to production systems is reviewed quarterly by each team's manager. Access that is not re-approved within 10 business days of the review is revoked.

## Devices

Company laptops must have full-disk encryption (FileVault or BitLocker), automatic screen lock after 5 minutes of inactivity, and the endpoint agent installed. Operating system security updates must be applied within 14 days of release.

Personal devices may access email and chat through the managed mobile app only. Personal devices must never store customer data.

## Reporting phishing and security incidents

Report suspected phishing with the "Report phish" button or by forwarding to security@northwind.cloud. If you clicked a link or entered credentials, report it in #sec-report within 1 hour; you will not be penalised for reporting quickly.

Lost or stolen devices must be reported to the IT service desk immediately so the device can be remotely wiped.
