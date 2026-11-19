# Agent Instructions

## Publishing Flow

- **Pack runs end at uploaded drafts.** Build, validate, zip, and upload pack assets to a **DRAFT** GitHub release, then stop. Creating a draft release never starts the scan gate — `workflow_dispatch` is the only trigger for `.github/workflows/scan-gate.yml`.
- **The scan gate is dispatched ONLY after the owner has reviewed and given the OK.** Never dispatch `.github/workflows/scan-gate.yml` before receiving explicit approval from the owner.
- **Dispatching the gate with `owner_approved=true` is the publish pipeline.** Once the owner gives the OK, dispatch `.github/workflows/scan-gate.yml` via `workflow_dispatch` with `owner_approved=true` (and `release_tag` if targeting a specific draft). That run handles the entire publish pipeline: VirusTotal scan, waiting for verdicts, auto-publishing the release on clean, and updating the catalog (`packs.json` and `thumbs/`).
- Dispatches without `owner_approved=true` (missing or `false`) fail immediately with an approval error; nothing is scanned and nothing is published.
- Use `simulate=true` only for dry-run verification (simulation never modifies `packs.json`/`thumbs/` or publishes a release).
