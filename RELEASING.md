# Development and release routine

Three scenarios, in increasing order of consequence: committing locally,
pushing, and cutting a release.

## One-time setup in a fresh clone

```bash
poetry install
poetry run pre-commit install
```

Without the second command the hooks never run, and problems surface in CI
instead of on your machine.

---

## 1. Committing locally

```bash
git add -A
git commit -m "..."
```

That is the whole routine. The hooks do the rest.

**What the hooks do.** Some of them rewrite files: `ruff format`,
`ruff check --fix`, `end-of-file-fixer`, `trailing-whitespace`, and
`mixed-line-ending`. Others only report: `mypy`, `check-ast`, `check-yaml`,
`check-json`, `check-docstring-first`, `debug-statements`, merge conflict and
case conflict checks, and a 2 MB file size limit. `kacl-verify` runs only when
`CHANGELOG.md` is part of the commit.

**When a commit is rejected**, there are two cases, and they need different
responses.

*Files were reformatted.* The output says "files were modified by this hook".
Nothing is wrong; the formatter edited your files and aborted so you can see
what changed. Stage the changes and commit again:

```bash
git add -A
git commit -m "..."
```

The second attempt normally passes.

*A check failed.* `mypy` reported a type error, `ruff` found something it
cannot fix automatically, or `kacl-verify` rejected the changelog. Fix the
cause and commit again. Do not stage and retry, since nothing was rewritten.

**Escape hatch.** `git commit --no-verify` skips the hooks. Reserve it for
work in progress on a branch you will clean up before pushing; `test.yml` runs
`pre-commit run --all-files` on push regardless, so skipped problems only
resurface later.

**Changelog.** If the change is user-facing, add a line under `## Unreleased`
in `CHANGELOG.md` in the same commit. Use `Added`, `Changed`, `Deprecated`,
`Removed`, `Fixed`, or `Security`. Describe the effect on the user, not the
mechanism. Internal refactors, tests, and CI changes need no entry.

---

## 2. Pushing

```bash
poetry run pytest      # optional, but faster than waiting for CI
git push origin <branch>
```

Pushing to **any** branch triggers `test.yml`, which runs actionlint, a cruft
check, `pre-commit run --all-files`, `pytest` on Python 3.10, 3.11 and 3.12,
and `mkdocs build`.

**Branch or `main`?** Either is fine for ordinary work. What matters is that
a release can only be cut from `main`, so anything intended for the next
release must reach `main` first.

Check the Actions tab after pushing. The multi-version `pytest` matrix
sometimes catches things a local run on one interpreter does not.

**After pulling someone else's changes**, or your own from another machine,
run `poetry install` if `poetry.lock` changed.

---

## 3. Cutting a release

**Before starting.**

1. Everything intended for the release is merged and pushed to `main`.
2. The Test workflow on `main` is green.
3. `## Unreleased` in `CHANGELOG.md` is not empty. The workflow fails if it is.

**Choose the bump.**

- `patch` for fixes and internal changes
- `minor` for new functionality that does not break existing usage
- `major` for breaking changes

**Run it.**

4. GitHub, Actions tab, "Draft a release", Run workflow, on `main`, enter the bump.

   This bumps the version in `pyproject.toml`, moves `Unreleased` into a dated
   section in `CHANGELOG.md`, updates `version` and `date-released` in
   `CITATION.cff`, commits all three, tags, and creates a **draft** release.
   Nothing is published yet.

5. Releases page, open the draft, check the notes read sensibly.

6. Click **Publish**. This triggers, automatically:
   - PyPI upload (`release.yml`)
   - documentation deploy to GitHub Pages
   - Zenodo deposit and DOI

7. `git pull`, to collect the version bump commit the workflow made.

8. Check the Zenodo record. Confirm the licence field: Zenodo cannot parse the
   dual licence and may fall back to CC-BY-4.0, which is wrong. Fix it in the
   Zenodo web interface if so.

**Never edit `version` in `pyproject.toml` or `CITATION.cff` by hand.**
The release workflow owns both, and a manual edit causes a mismatch between
the tag, the PyPI release, and the archived record.

**Never publish with `poetry publish` directly.** It uploads to PyPI without
creating a GitHub release, so no tag, no changelog entry, and no Zenodo
deposit. Always release through the workflow.

### If the release workflow fails

Where it failed determines the cleanup.

- **Before the "Create tag" step**: fix and rerun, nothing to undo.
- **After the tag was pushed**: delete the tag first, or the next run collides.

```bash
git push --delete origin <version>
git tag -d <version>
```

Also revert the version bump commit if one was made, so `pyproject.toml` does
not claim a version that was never released.

---

## Occasional maintenance

- `poetry run pre-commit autoupdate` refreshes hook versions. The pinned
  `pre-commit-hooks` revision is old; update it when convenient, then run
  `poetry run pre-commit run --all-files` to catch anything the newer hooks
  flag.
- Zenodo only archives releases published **after** the repository was enabled
  in Zenodo's GitHub settings.
- The PyPI upload needs the `PYPI_TOKEN` repository secret.
