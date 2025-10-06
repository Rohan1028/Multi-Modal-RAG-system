# Contributing to Multimodal RAG System

Thanks for your interest in contributing! Please follow these guidelines to keep contributions smooth and high quality.

## Getting Started

1. Fork the repository and clone your fork.
2. Create a feature branch off `main`: `git checkout -b feature/my-change`.
3. Run `make setup` and ensure `make test` passes before submitting a PR.

## Development Workflow

- Keep changes focused; open draft PRs early for feedback.
- Write tests for new functionality or bug fixes.
- Run `make lint` and `make test` locally before pushing.
- Update documentation (README, CHANGELOG) when behavior changes.

## Commit & PR Conventions

- Use conventional commit prefixes when possible (e.g., `feat:`, `fix:`, `docs:`).
- Reference related issues and milestones in the PR description.
- Ensure CI passes; PRs failing CI will not be merged.

## Issue Labels & Milestones

We use the following labels:
- `area:*` for functional areas (ingestion, retrieval, generation, api, ui, evaluation, infra)
- `kind:*` for change type (feature, bug, chore, docs)
- `priority:*` for urgency (P0, P1, P2)

Milestones map to the project roadmap shared in the README. Attach new issues to the relevant milestone when possible.

## Code of Conduct

Please read our [Code of Conduct](CODE_OF_CONDUCT.md) and follow it in all interactions.

Thanks again for contributing!
