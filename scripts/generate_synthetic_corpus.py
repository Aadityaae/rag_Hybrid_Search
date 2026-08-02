"""Generates a small synthetic internal-engineering-docs corpus for a
fictional company (Northwind Analytics) so the pipeline can be built and
tested end to end without needing real proprietary documents. Docs
deliberately cross-reference each other so multi-hop eval questions have
a real answer spread across two files.

Run: python scripts/generate_synthetic_corpus.py
"""
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

DOCS = {
"onboarding.md": """# Engineering Onboarding Guide

Welcome to the Northwind Analytics engineering team. This guide covers your first two weeks.

## Day 1: Accounts and Access

Your manager will file an access request in the Identity Portal on your behalf. You should
receive credentials for: GitHub (org: northwind-eng), the internal Confluence-equivalent wiki,
PagerDuty, and the staging AWS account. Production AWS access is granted separately after you
complete the security training in week two.

## Week 1: Environment Setup

Install the `nw-cli` tool via `brew install northwind/tap/nw-cli`. Run `nw-cli setup` to clone
the core repositories and configure your local `.env` file. The core services run locally via
`docker-compose up` from the `platform` repo root. If containers fail to start, check that
ports 5432 (Postgres) and 6379 (Redis) are free.

## Week 1: Your First Pull Request

Every new engineer ships a small documentation fix or test-coverage improvement in their first
week. Pull requests require two approvals and a passing CI run before merge. See the
Deployment Pipeline doc for what happens after merge.

## Week 2: On-Call Shadowing

New engineers shadow one on-call rotation before joining the rotation themselves. See the
On-Call Rotation doc for how shifts are scheduled and what tooling is used.

## Security Training

Security training is mandatory before production access is granted. It covers credential
handling, the incident response process, and secure coding basics. It takes about two hours
and must be renewed annually.
""",

"deployment-pipeline.md": """# Deployment Pipeline

## Overview

Northwind Analytics deploys via a trunk-based workflow. Merges to `main` trigger the CI/CD
pipeline defined in `platform/.github/workflows/deploy.yml`.

## Stages

1. **Build**: Compiles the service and builds a container image tagged with the git SHA.
2. **Test**: Runs unit tests, integration tests against a disposable Postgres instance, and
   a lint/type-check pass. A failing stage blocks promotion.
3. **Staging Deploy**: The image is automatically deployed to staging. Smoke tests run against
   the staging environment.
4. **Canary**: 5% of production traffic is routed to the new version for 15 minutes. Error
   rates and p99 latency are compared against the baseline automatically.
5. **Full Rollout**: If canary metrics are healthy, the deploy proceeds to 100% of production
   traffic over 10 minutes. If canary metrics regress, the pipeline automatically rolls back
   and pages the on-call engineer.

## Rollbacks

Any engineer can trigger a manual rollback with `nw-cli deploy rollback --service <name>`.
Rollbacks restore the previous container image and complete in under two minutes. A rollback
should always be followed by an incident write-up if it was triggered by a production issue --
see the Incident Response doc for the write-up template.

## Feature Flags

Risky changes should ship behind a feature flag rather than relying on canary alone. See the
Feature Flags doc for how flags are created and targeted.

## Database Migrations

Migrations are NOT run automatically as part of this pipeline. See the Database Migrations doc
for the separate, manual migration process and why it's kept separate from code deploys.
""",

"incident-response.md": """# Incident Response Process

## Severity Levels

- **SEV1**: Full outage or data loss affecting all customers. Page immediately, all-hands.
- **SEV2**: Significant degradation affecting a subset of customers or a core feature.
- **SEV3**: Minor issue, workaround available, no customer-facing impact yet.

## Who Responds

The on-call engineer (see the On-Call Rotation doc for the current schedule) is the first
responder for any page. For SEV1 incidents, the on-call engineer immediately loops in the
Incident Commander on duty, who coordinates the response and communication.

## Response Steps

1. Acknowledge the page in PagerDuty within 5 minutes.
2. Create an incident channel in Slack named `#incident-<date>-<short-name>`.
3. Post a status update every 15 minutes for SEV1/SEV2 incidents.
4. Mitigate first, root-cause later. A rollback (see the Deployment Pipeline doc) is often the
   fastest mitigation for a bad deploy.
5. Once mitigated, downgrade severity if applicable and continue investigation during business
   hours.

## Postmortems

Every SEV1 and SEV2 incident requires a postmortem within 3 business days, using the template
in the wiki. Postmortems are blameless and focus on system and process gaps, not individual
error. The postmortem must include a timeline, root cause, customer impact, and at least two
concrete follow-up action items with owners.
""",

"api-authentication.md": """# API Authentication

## Overview

All external API requests to Northwind Analytics services must be authenticated using an API
key issued through the Developer Portal. Internal service-to-service calls use short-lived
mTLS certificates instead, issued automatically by the service mesh.

## Issuing API Keys

Customers generate API keys in the Developer Portal under Settings > API Keys. Each key is
scoped to a set of permissions (read-only, read-write, admin) chosen at creation time. Keys do
not expire by default but can be revoked instantly from the portal.

## Rate Limits

Read-only keys are limited to 600 requests per minute. Read-write keys are limited to 120
requests per minute. Exceeding the limit returns an HTTP 429 with a `Retry-After` header.
Enterprise customers can request higher limits by contacting support.

## Key Rotation

We recommend rotating API keys every 90 days. The portal supports creating a new key alongside
an existing one so traffic can be migrated before the old key is revoked, avoiding downtime.

## Common Errors

A 401 response means the key is missing, malformed, or revoked. A 403 means the key is valid
but lacks the required scope for that endpoint. These are logged and visible in the Developer
Portal's request log for debugging.
""",

"database-migrations.md": """# Database Migrations

## Why Migrations Are Manual

Unlike the rest of the deployment pipeline (see the Deployment Pipeline doc), schema migrations
are run manually and deliberately kept out of the automated CI/CD flow. This is because a bad
migration is much harder to roll back than a bad code deploy, and migrations often need to be
sequenced carefully around a deploy rather than bundled with it.

## Migration Process

1. Write the migration using the `nw-cli migrate new <name>` scaffold, which generates a
   timestamped file in `db/migrations/`.
2. Migrations must be backward-compatible with the currently deployed code for at least one
   release cycle -- this allows safe rollbacks without a data mismatch.
3. Run the migration against staging first via `nw-cli migrate up --env staging`.
4. Request review from a member of the Data Platform team before running against production.
5. Run against production during the weekly migration window (Tuesdays 10am-12pm PT) unless
   it's fixing a live incident.

## Rollback

Every migration must include a corresponding `down` migration. If a migration causes issues,
run `nw-cli migrate down` to revert. Because migrations are decoupled from code deploys, a
migration rollback does not require a code rollback and vice versa.
""",

"on-call-rotation.md": """# On-Call Rotation

## Schedule

On-call is a weekly rotation managed in PagerDuty, covering all engineers who have completed
security training and shadowed at least one prior rotation (see the Onboarding Guide). The
rotation covers 24/7 primary coverage plus a secondary backup who is paged if the primary
doesn't acknowledge within 10 minutes.

## Tooling

Pages arrive via PagerDuty, which integrates with our monitoring stack (see the Logging and
Monitoring doc) to automatically page on threshold breaches. On-call engineers are expected to
keep the PagerDuty mobile app notifications enabled during their shift.

## Compensation

On-call engineers receive a stipend for each week of primary coverage, and receive equivalent
time off if a shift involves more than 3 pages between 10pm and 7am local time.

## Handoff

Handoff happens every Monday at 10am via a short sync where the outgoing on-call engineer
briefs the incoming one on any open issues, recent incidents, and anything flaky to watch.
""",

"logging-and-monitoring.md": """# Logging and Monitoring

## Stack

Northwind Analytics uses a centralized logging stack (Loki) and a metrics stack (Prometheus +
Grafana). All services must emit structured JSON logs and expose a `/metrics` endpoint in the
Prometheus exposition format.

## Alerting

Alert thresholds are defined per-service in `platform/monitoring/alerts/`. When a threshold is
breached, an alert fires to PagerDuty, which pages whoever is on-call (see the On-Call Rotation
doc). Alerts are tuned to minimize false pages -- if you're getting paged for noise, file a
ticket to adjust the threshold rather than snoozing it repeatedly.

## Dashboards

Every service should have a Grafana dashboard covering request rate, error rate, and latency
(the "RED" method). Dashboards are provisioned as code in `platform/monitoring/dashboards/` and
reviewed in the same PR as the service change that needs them.

## Log Retention

Logs are retained for 30 days in hot storage and 1 year in cold storage for compliance. Cold
storage logs require a support ticket to access, with a 24-hour SLA.
""",

"feature-flags.md": """# Feature Flags

## Overview

Feature flags let engineers ship code to production dark and roll it out gradually, independent
of the deploy pipeline's canary stage (see the Deployment Pipeline doc). This is the preferred
mechanism for de-risking large or behavior-changing releases.

## Creating a Flag

Flags are created in the LaunchDarkly-equivalent internal tool, `nw-flags`, via
`nw-cli flags create <flag-name> --default off`. Every flag must have an owner and an expiry
date -- flags older than 90 days are automatically flagged for cleanup in a weekly report.

## Targeting

Flags can be targeted by customer segment, percentage rollout, or individual account for
internal dogfooding. Percentage rollouts are the most common pattern for de-risking a new code
path: start at 1%, monitor error rates and relevant metrics, and increase gradually.

## Cleanup

Once a flag is fully rolled out and stable, the flag and its associated conditional code must
be removed within one sprint. Stale flags are a common source of confusing bugs and are treated
as tech debt.
""",
}

def main():
    for filename, content in DOCS.items():
        (RAW_DIR / filename).write_text(content.strip() + "\n", encoding="utf-8")
    print(f"Wrote {len(DOCS)} synthetic docs to {RAW_DIR}")

if __name__ == "__main__":
    main()
