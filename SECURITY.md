# Security policy

## Supported code

Security fixes target the current `main` branch. Historical research snapshots and their frozen environments remain provenance records; they are not automatically upgraded or represented as supported production deployments. Published versions will list their support status when versioned releases are established.

## Report a vulnerability privately

Use [GitHub private vulnerability reporting](https://github.com/thinktraveller/YONOD/security/advisories/new). Include the affected commit, operating system, dependency versions, a minimal non-sensitive reproduction, and the expected impact. Do not post exploit details, tokens, private datasets or patient/personal information in public issues. Ordinary installation and usage bugs belong in the issue tracker with redacted logs.

Reports are reviewed by the repository maintainer. This project does not promise a fixed response time or a paid support service. Follow-up and any coordinated disclosure can take place in the private advisory.

## Dependency updates and credentials

Dependabot alerts are reviewed as dependency findings. Security-update pull requests require compatibility review and testing; they are not automatically merged. An empty alert list only describes the known dependency graph at that time.

Validate upgrades in an isolated development/CI environment. Keep frozen research dependencies, configurations and results identifiable; do not silently replace their environment or combine results across changed model versions.

Ignore rules and push protection are safeguards, not a substitute for removing credentials from shared files. If a real credential is exposed, revoke or rotate it first, then assess the affected commits and assets. Do not rewrite repository history based only on a filename or a suspected cache display.
