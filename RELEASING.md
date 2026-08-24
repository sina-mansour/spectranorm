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

Keep nothing below the `## Unreleased` section other than released version
sections. `kacl-cli release` absorbs everything between `## Unreleased` and the
next `##` heading into the new version's release notes, so stray prose ends up
in the GitHub release description.

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

Check the Actions tab, or `gh run watch`, after pushing. The multi-version
`pytest` matrix sometimes catches things a local run on one interpreter does not.

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

### From the terminal

```bash
gh workflow run draft_release.yml --ref main -f version=patch
gh run watch                       # wait for the draft to be created
git pull                           # collect the version bump commit
gh release view <version>          # check the notes read sensibly
gh release edit <version> --draft=false    # publish
```

### From the browser

GitHub, Actions tab, "Draft a release", Run workflow, on `main`, enter the bump.
Then Releases page, open the draft, check it, click **Publish**. Afterwards
`git pull` locally.

### What each step does

The **draft** step bumps the version in `pyproject.toml`, moves `Unreleased`
into a dated section in `CHANGELOG.md`, updates `version` and `date-released`
in `CITATION.cff`, commits all three, tags, and creates a draft release.
Nothing is published yet.

**Publishing** the draft fires the `release: published` event, which triggers,
automatically:

- PyPI upload (`release.yml`)
- documentation deploy to GitHub Pages
- Zenodo deposit and DOI

### Afterwards

Confirm the new version on PyPI, that the docs site rebuilt, and that the
Zenodo record exists.

**Never edit `version` in `pyproject.toml` or `CITATION.cff` by hand.**
The release workflow owns both, and a manual edit causes a mismatch between
the tag, the PyPI release, and the archived record.

**Never publish with `poetry publish` directly.** It uploads to PyPI without
creating a GitHub release, so no tag, no changelog entry, and no Zenodo
deposit. Always release through the workflow.

### Zenodo metadata

Most of the record is populated automatically from `.zenodo.json`, including
the licence, which is set to `other-nc` ("Other (Non-Commercial)"). This is an
approximation of the dual licence, but an accurate one: Zenodo's vocabulary has
no identifier for AGPL-3.0 with a non-commercial restriction, and `other-nc`
is the closest correct choice. Without it, Zenodo defaults to CC-BY-4.0, which
would be wrong.

Two fields cannot be set from `.zenodo.json`, because they are InvenioRDM
custom fields with no equivalent in the legacy metadata schema that
`.zenodo.json` uses:

- **Programming language**: Python
- **Development status**: Active

These should carry over from the previous version, since Zenodo inherits
metadata when creating a new version. Check once and only re-enter them if they
did not.

**Optional refinement.** If you want the full dual licence text on the record
rather than "Other (Non-Commercial)", edit the record and set a custom licence:

- Title: `Dual licence: AGPL-3.0 for non-commercial use, or commercial licence for commercial use`
- Link: `https://github.com/sina-mansour/spectranorm/blob/main/LICENSE`
- Description: `Free for non-commercial use, including academic research and teaching, under the terms of the GNU Affero General Public License v3.0. Commercial use requires a separate licence. See the LICENSE file in the repository for full terms and contact details.`
- Copyright: `© 2025 Sina Mansour L. and collaborators.`

This must be redone after every release, since `.zenodo.json` will overwrite
the licence field each time. Skip it unless the fuller wording matters.

**Verifying.** The public record page is authoritative. API responses may be
served from cache:

```bash
curl -s -H "Accept: application/vnd.inveniordm.v1+json" \
  "https://zenodo.org/api/records/<id>" | python3 -m json.tool | grep -A8 '"rights"'
```

Saving may fail with a generic error if Zenodo's search service is degraded
(check <https://stats.uptimerobot.com/vlYOVuWgM>). Retry later; metadata edits
do not affect the DOI or the files.

**At the 0.1.5 release**, confirm that `other-nc` was applied and that the two
custom fields carried over. If both held, this section can be shortened.

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

- `poetry run pre-commit autoupdate` refreshes hook versions, then
  `poetry run pre-commit run --all-files` to catch anything the newer hooks flag.
- Bump GitHub Action versions in `.github/workflows/` and in
  `.github/actions/python-poetry-env/action.yml` when runner deprecation
  warnings appear.
- Zenodo only archives releases published **after** the repository was enabled
  in Zenodo's GitHub settings.
- The PyPI upload needs the `PYPI_TOKEN` repository secret.
- `.zenodo.json` follows Zenodo's legacy deposit schema, documented at
  <https://developers.zenodo.org/#deposit-metadata>.
- Changes prior to 0.1.4 were not recorded in `CHANGELOG.md`.
