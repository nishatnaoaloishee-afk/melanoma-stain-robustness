# Final checks before making this public

- [ ] Confirm code ownership and permission to share legacy scripts.
- [ ] Confirm no credentials, personally identifying information, private data, or reviewer correspondence.
- [ ] Copy locked 205-ROI split (`inputs/splits/*.npz`) from the original experiment.
- [ ] Copy compact original sampled-pixel NPZ, calibration factors and five enhancement targets.
- [ ] Copy 4-method Stage 11/12 results, five Stage 15 result/config pairs, Stage 16/17 final outputs.
- [ ] Copy current Stage 20-23 outputs (the earlier project inventory did not list them).
- [ ] Capture original Python and dependency versions (not a new arbitrary install).
- [ ] Verify the original 205 matched image/annotation pair filenames versus the currently downloadable PUMA version.
- [ ] Run `python tools/check_release.py` and verify selected paper-reported numbers manually.
- [ ] Check no data directories or `.venv` are staged: `git status --short` and `git diff --cached --stat`.
- [ ] Verify complete README instructions and final source provenance.
- [ ] Push to GitHub, inspect files in browser, and create a stable version tag/release.
- [ ] Only after public URL works, add code availability sentence in manuscript.
