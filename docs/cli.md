# transformers-mic CLI

`transformers-mic` packages the existing MIC runner, fixture builder, replay tools,
and reviewed catalog. The `tmic` command creates a working directory for a model,
proposes additional usage paths when needed, and measures expectations there.
It is an alpha implementation: generated plans still need code review and a
working model-specific build recipe. New artifacts use `hf-internal-testing/mic-{name}-slim` and
`mic-{name}-integration`. The nine legacy `proto-*-slim` repositories were moved
to `mic-*-slim`, preserving suffixes and revisions; historical evidence stays as recorded.

## Install

From this repository (Python 3.10+):

```sh
python -m pip install -e '.[runtime,dev]'
python -m pip install 'hf-doc-builder @ git+https://github.com/huggingface/doc-builder.git'
hf auth login        # publishing needs write access to the target namespace
gh auth login        # history search; or pass --no-history to create
```

## Step by step: one model

**The order is: real checkpoint → contract → fixture.** You never write a fixture
first. You start from a real checkpoint id. `create` writes the contract (the
usage code and manifest), and `build` then derives the slim fixture from that
checkpoint. `build` needs the contract because it accepts weights only after
the contract's inputs move the outputs (the sensitivity gate).

1. **Create the workspace from the real checkpoint.**

   ```sh
   tmic create google-bert/bert-base-uncased --out work/bert --namespace hf-internal-testing
   cd work/bert
   ```

   `--namespace` sets the Hub owner of the new `mic-*-slim` and
   `mic-*-integration` repos. Without it, they belong to whoever is logged in
   (for example `you/mic-bert-base-uncased-integration`). It has no effect on
   which token pays for the LLM call. Set it here, because `create` writes the
   name into the manifest and the lock, and changing it later means creating
   the workspace again. Nothing is uploaded until step 7.

   If the checkpoint is in the bundled catalog, this reuses the published
   contract and needs no LLM call. Add `--regenerate` to build a new
   `mic-*-slim` fixture instead of reusing the published one. If the checkpoint
   is new, the planner drafts the contract on your HF token (see
   [When the LLM is used](#when-the-llm-is-used)). By default the contract
   covers both the real checkpoint and the fixture. Use `--fixtures-only` to
   declare a slim-only contract.

2. **Read the draft.** Open `integration.json` and `usage.md` before running
   anything. They are executable Python, and nothing sandboxes them.

3. **Build the fixture.**

   ```sh
   tmic build
   ```

   This shrinks the real config, copies the processors byte for byte and tries
   random weights first. If the gates fail, it tries a bounded overfit. If both
   fail, the build is rejected. `.mic/project.json` records which weights won
   and why ([Random or overfit?](#random-or-overfit)).

   For models with fixed geometry or processor tensors, add reviewed entries to
   the fixture's `build` recipe before this step. For example:

   ```json
   {
     "copy_assets": ["attention_mask.safetensors"],
     "preserve_frozen_prefixes": ["model.geometry."]
   }
   ```

   `copy_assets` names extra files copied byte for byte from the pinned source.
   `preserve_frozen_prefixes` copies matching frozen parameters and persistent
   buffers only when they have `requires_grad=False`; missing names and shape
   changes fail the build. Their source hashes are included in the fixture key
   and build record. The planner cannot add these options; review and edit the
   recipe yourself.

4. **Record the candidate expectations.**

   ```sh
   tmic record --all-variants
   ```

   Every case runs twice, each time in a fresh process, and the results are
   written to `*.candidate.json`.

5. **Review and approve them.**

   ```sh
   tmic review
   tmic review --approve
   ```

   `--approve` accepts only new and unchanged values. Changed values stay
   pending until someone approves them by hand.

6. **Check the approved expectations.**

   ```sh
   tmic check --all-variants
   tmic status
   ```

7. **Publish to the Hub.**

   ```sh
   tmic publish                 # preview the destinations
   tmic publish --execute       # upload; the lock is rewritten to immutable revisions
   tmic check --all-variants    # publishing invalidates receipts: check the published pins
   ```

   Uploads are public and pinned, so this step is outward-facing.

8. **Export into Transformers.**

   ```sh
   tmic export --transformers-checkout /path/to/transformers
   ```

   This writes `tests/contracts/fleet.lock` and the expectations. It does not
   commit or open a PR.

## Several models at once

List the models in a JSON file shaped like `experiments/proto-models.json`, with
`name`, `model_type` and `checkpoint` for each entry. Then:

```sh
tmic batch --models-file my-models.json --out work/fleet --phase prepare --no-history
tmic batch --models-file my-models.json --out work/fleet --phase run --resume
```

`prepare` runs step 1 for every model, always with `--regenerate`. Read the
drafts in `work/fleet/projects/*/` before you start `run`, which does steps 3–6. `run`
calls `review --approve` itself, so you do not get a manual review there:
inspect the approved expectations before you publish. `batch` keeps going past
failures and records every stage. `--select NAME` reruns one model. `--publish`
adds step 7, and it uploads only after all declared checks pass.

Use a checkout pinned to the workspace's `fleet.lock` dependency environment for
baseline reproduction. Installing the runtime extra alone does not freeze that
environment. The repository's GPU workflow uses `.github/scripts/install_pinned.sh`.
The PyPI release of doc-builder currently lacks the runnable-markdown collector;
install its Git version as above. The installed wheel bundles the backend and
catalog, so it also works outside this checkout; model weights download from the
immutable Hub pins.

`tmic create --offline` reuses a catalog contract without network requests.
Otherwise, `gh` must be installed and authenticated for GitHub history searches.
`--no-history` skips them; `--history-file failures.json` imports a normalized JSON
list from relore or another source. A failure object can contain `url`, `title`,
`body`, and revision information. Searches are bounded candidates, not a complete
inventory of past failures, and do not count as regression detection.

## When the LLM is used

Catalog reuse requires no inference call. For a new model, or with `create --llm`,
the planner uses **Hugging Face Inference Providers**, default model
**`Qwen/Qwen3.8-27B`**, with `provider="auto"`. Set `--provider` or `--llm-model`
on `create` and `update` to override them. Authentication uses the normal HF token
(`HF_TOKEN` or `hf auth login`) with Inference Providers permission and available
credits. Provider inference may be billed to that account.

The planner receives checkpoint config, the manifest schema, and failure
candidates. It proposes a manifest and runnable usage, stored in `.mic/plan.json`.
Invalid drafts receive validation feedback for one repair attempt; persistent
errors stop creation without executing the draft. It cannot generate expected
numeric values, loosen tolerances, or choose overfit
weights. Review generated `integration.json` and `usage.md` before running them:
syntax/schema checks do not sandbox Python. `create` alone does not execute the
proposed model code; `--build` and `--record` are explicit execution shortcuts.

## Random or overfit?

1. Reuse the existing pinned slim first. A new assertion does not automatically
   need new weights: measure and replay it on that fixture.
2. A fresh random fixture is sufficient if it passes the builder's sensitivity
   and finite-output gates for the proposed usage path.
3. The builder tries bounded overfitting when random fails. Use
   `--readable-output` or `--weights overfit` when meaningful decoded output is
   itself a requirement. `--weights random` forbids a silent overfit fallback.
4. If neither passes, the candidate is rejected. Some usage paths need a real
   checkpoint; declaring checkpoint-only targets is an explicit manifest choice.

`.mic/project.json` records the selected weights and reason; `build.json` contains
the gate evidence. A gate pass establishes usefulness for those probes, **not**
coverage of a past failure. Random and overfit can tie on historical coverage;
readability alone does not prove one detects more bugs. Legacy published fixtures
may predate gate metadata: they can be reused with reviewed baselines, but new
builds must pass the current gates.

## Add paths and prove regressions

```sh
tmic update 'Add a padded batch through the public pipeline API'
tmic record --all-variants
tmic review
tmic check --all-variants
tmic replay FIX_SHA --repo /path/to/transformers --match-output answer_logits
```

Update retains the old source under `.mic/history/` and keeps the fixture pin.
It rejects removal of existing output assertions. Retaining every existing call
is also a code-review requirement; output names alone cannot prove that. Use
`update --retrain` or `build --rebuild` only when the existing fixture is inadequate.
Builds are staged into immutable `.mic/artifacts/` directories and do not overwrite
previous approved expectations. LLM updates currently support single-checkpoint
contracts; grouped contracts can be edited directly and checked with the same CLI.

Replay archives bad/fixed Transformers revisions from a local Git repository,
executes the same contract and frozen baselines on each, and exits successfully
only for a failure on the bad revision followed by a pass on the fix. Without
`--match-output`, a catch may be on any output and needs inspection. No automatic
claim converts search results or sensitivity gates into replay evidence.

`record` runs each case twice before writing `*.candidate.json`. `review --approve`
accepts new and unchanged candidates, including added output fields. Changed or
removed existing values remain pending for manual review; the LLM never approves
them. Review new baselines for meaningful assertions before approval. `check`
uses approved files and pins; `--target` allows focused investigation, but cannot
satisfy the publication gate. Publication requires every declared variant checked
on the current sources and approved expectations. These receipts cover the
platforms actually run, not an untested CPU/GPU or dependency matrix.

## Hub and Transformers updates

`publish` previews the destination after validation. `publish --execute` delegates
to the existing checksum-checked Hub publishers and updates the lock to immutable
Hub revisions. It invalidates local check receipts, so check again before export.
Upload requires write access to the integration/fixture namespaces; catalog
artifacts remain under their existing owners.

`export --contract-name mic-bert` gives the destination a distinct name when
retaining a historical `BERT` contract. Export rejects names that differ only
by case, avoiding baseline-folder collisions on macOS and Windows.

`export` writes a reviewable `tests/contracts/fleet.lock` and expectations into the
provided checkout, preserves other contracts, and removes workspace paths from
pins. It does not commit, open a PR, or merge Transformers changes. CI can run:

```sh
tmic check --lock tests/contracts/fleet.lock --contract BERT --all-variants
```

`tmic status` shows project state. Workspaces contain source, history, candidate
expectations, checks and artifacts; keep them outside versioned source or ignore
`work/`. The current release implements Transformers orchestration; the bundled
prototype still includes Transformers.js contracts, but the CLI does not publish
or export a JavaScript fleet.

## Validation of this alpha

The current fleet has **23 published architectures**, each checked on GPU in
three variants. `auto` uses explicit Auto classes for BERT/T5/ViT, without a
separate fixture. All Hub fixtures and contracts use immutable pins.
[Per-model results and scope](../experiments/results/prototype-regeneration.json).
Some GPU checks use documented model-scoped patches; GPT-OSS's slim is dense,
so these results do not establish unpatched or quantized-kernel coverage.

Native m8i CPU recording checked/exported **22 models / 66 variants**. The
four-thread proof-branch push executed **77 cases: 73 pass, four exact Whisper
xfails**. Wav2Vec2 skips only loss/gradients. The PR synthetic merge uses newer
source: Whisper generation succeeds, exposing an obsolete error match and
prototype numeric/missing-BF16-reference failures. It remains red.
[Current proof and remaining work](transformers-ci-proof.md).

Use the recorded runtime and four-thread profile for CPU reproduction. The proof
asserts the checkout import path and thread count. New/unchanged candidates were
approved; numeric tolerances were not loosened. Table Transformer's focused
four-thread repeat exactly matched its original references.

**CSM, Gemma3n and PI0 remain rejected**: real audio processor incompatibility,
nonfinite gradients, and weak state/image sensitivity, respectively. DETR and
SpeechT5 exempt only amplitude scaling canceled by input normalization; their
remaining probes and the .001 threshold stay fixed. The new fleet has no
historical-replay or full-checkpoint claim.

Regenerate or resume the declared fleet:

```sh
tmic batch --models-file experiments/proto-models.json --out work/fleet --phase prepare --no-history
tmic batch --models-file experiments/proto-models.json --out work/fleet --phase run --resume
tmic batch --models-file experiments/proto-models.json --out work/fleet --phase run --resume --select whisper
```

`batch` continues after failures and records each stage. `--publish` explicitly
adds upload and published-pin checks. `mic-published` can check GPU pins only
with `published_gpu_only=true`; native CPU recording/checks remain separate.

31 package tests and ruff pass. Wheel/sdist checks verify bundled resources and
exclude weights, candidate baselines and installed JavaScript dependencies.
The installed CLI creates an offline BERT workspace outside the checkout.
Transformers `make style` passes in the proof worktree. The
[initial package validation](../experiments/results/mic_cli_validation.json)
is separate from the architecture fleet study.
