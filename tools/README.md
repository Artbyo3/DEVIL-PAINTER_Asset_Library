# Making Packs

The pack-making pipeline keeps curation and taste manual while automating validation, splitting, zipping, and draft release creation:

**curate folder by hand -> `check_pack` -> `split_pack` (if oversized) -> `make_pack` -> scan gate publishes**

## Pipeline Steps

1. **Curate folder by hand**
   - Put one asset kind per pack folder inside its matching subfolder (`Textures/`, `Alphas/`, `Materials/`, `Brushes/`, `Generators/`, or a custom folder for generic `asset` files).
   - For non-image assets, you may place an optional sibling preview image named `<stem>.preview.<img_ext>` in the same folder.
   - Add `pack.json` with `name`, `description`, `tags`, and `sets` (exact file count).

2. **Validate with `tools/check_pack.py`**
   ```bash
   python tools/check_pack.py <pack_folder>
   ```
   Checks all standing rules (`<= 200` files, kind-pure, valid `pack.json`, readable images, preview convention, no stray or hidden files) and exits `0` only if every rule passes.

3. **Split oversized packs with `tools/split_pack.py`** (only if > 200 files)
   ```bash
   python tools/split_pack.py <pack_folder> [--max 200]
   ```
   Moves whole pattern families into balanced sibling folders (`<= --max` files each, never splitting a family across packs) and writes a `pack.json` in each new folder. Stops with an error if any single family alone exceeds `--max`.

4. **Build zip and draft release with `tools/make_pack.py`**
   ```bash
   python tools/make_pack.py <pack_folder>
   ```
   Runs `check_pack` first, builds `<PackFolder>.zip`, and creates a **DRAFT** GitHub release containing the zip plus every individual catalog file using the repository's tag and naming conventions. Stops at the draft.

5. **Publish via Release Scan Gate**
   Run the **Release Scan Gate** GitHub Actions workflow (`workflow_dispatch` with `release_tag`) once the draft upload is complete. The gate scans every asset with VirusTotal, builds catalog entries and thumbnails (`build_pack_entry.py`, `build_thumbs.py`), updates `packs.json`, attaches `PACK-REPORT.txt`, and publishes the release.
