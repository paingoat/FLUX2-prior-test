# Pico Scribble 300 — SIGMA visual + Qwen

Use `build_pico_scribble_300_colab.ipynb`. Earlier notebooks remain experiment records.
The goal is exactly 100 add + 100 remove + 100 replace, one scribble per unique
source/result pair. No SAM, detector, or FLUX.2 training is included.

## Run

1. Select a GPU runtime. Enable `USE_DRIVE=True` before the first run to preserve
   progress across resets; otherwise `/content` is temporary.
2. Choose `DATA_SOURCE="remote_s3"` or upload a ZIP. Remote mode caches the official
   Pico metadata and Open Images URL-to-ID CSV, then downloads individual S3
   source images. Missing/ambiguous matches are skipped without Flickr fallback.
3. Run all cells. Up to 300 extra candidates per operation are attempted per
   preparation pass, with a pool of 2000 per operation by default. Preparation
   stops early for an operation that reaches 100 accepted samples.
4. Clear cases are `auto_accepted`; broad/empty masks are `rejected`; ambiguous
   geometry or text extraction is `needs_review`.
5. The gallery defaults to `needs_review`. Edit Qwen noun phrases, change styles,
   or click a different region if needed. Add clicks on result; remove/replace
   clicks on original. Approve or reject. A manual rejection overrides auto acceptance.
6. Audit a contact sheet of automatic samples, displaying scribble and target side
   by side. Use the status dropdown to inspect individual automatic cases.
7. If short of quota, click **Bổ sung ứng viên** or increase the candidate pool and
   rerun candidate loading. Rerun export once there are 100 accepted per operation.

## Filtering and grouping

Small components are filtered. Raw component boxes are grouped if they overlap
substantially or are very close; no unlimited dilation is performed. Scribbles
use original group masks, not the artificial bridges between them.

Automatic acceptance requires a compact dominant group, limited secondary areas,
limited merge expansion and acceptable overall changed area. Add masks covering
more than 35% of the image are rejected by default; the other operations use 55%.
Masks covering over 25% or boxes over 35% are sent to review. These are conservative
starting heuristics and may reject valid large objects. Mild aspect-ratio resize
up to 5% may pass automatically; larger shifts need review. These checks do not
prove semantic edit correctness or separate shadows touching the object.

Group/quality parameters are recorded in `group_policy` and affect the automation
signature. Changing them re-evaluates decisions on cached heatmaps. Image/inference
processing parameters remain locked to the run name to protect previous assets.
Previous-version reviews are backed up. Existing manual rejections remain rejected.

## Qwen short prompts

`Qwen/Qwen2.5-0.5B-Instruct` is used independently of SIGMA's text branch. The HF
revision is resolved once and cached. Text inference uses GPU by default, or CPU
with `TEXT_DEVICE="cpu"`. Qwen returns structured noun phrases; code generates the
final sentence from operation, object, replacement, and scribble style.

The parser validates JSON/string fields, operation, grounded words, explicit
quantities and quoted sign text. Replace destinations must come from the clause
after `with`, preventing a location like `window` from becoming the replacement.
Ambiguous or invalid responses go to review; there is no automatic regex fallback.
A 0.5B model can still omit semantic details despite passing these checks.

Examples of generated text:

- `Add a dark green canoe in the marked region.`
- `Add a dark green canoe inside the oval.`
- `Remove the person marked with an X.`
- `Replace the telescope inside the hand-drawn box with a leafy plant.`

Approximately half of samples use a generic reference, half a shape-specific one,
deterministically per ID. X/stroke templates do not say `inside`. Changes to style
reset manual prompt overrides and regenerate text. Slots, model revision, cache
key, prompt version and any explicit override are recorded.

## Scribbles, manual selection, and output

Add/replace: box, ellipse, loose freehand loop. Remove also supports X, hatch and
short interior strokes. Seed, style and algorithm version are saved.

Clicks use `ipyevents` with the Colab custom widget manager. A click chooses one
connected component, not semantic segmentation; object and shadow joined in a
mask cannot be separated by a click. Changing threshold clears the selection.

```text
pico_scribble_300_v1_<export-hash>/
  samples/add/add_pico_28821/{original,scribble,result,mask,change_mask}.png
  samples/remove/...
  samples/replace/...
  metadata.jsonl
  review.csv
  decisions.json
  run_config.json
  README.txt
```

`mask.png` is the selected cue. `change_mask.png` retains filtered changes.
`review_status` distinguishes `auto_accepted` from `manual_approved`; suspect and
rejected records cannot fill quota. `decisions.json` includes decision reasons and
metrics. Original/overlay condition FLUX.2; result is its target. No files are
published externally. Follow Pico-Banana and Open Images data terms.

## Upload format

Upload a ZIP containing `pairs.jsonl` plus images, or the existing Prior_test
metadata/images layout. Example row:

```json
{"operation":"remove","prompt_original":"Remove the red flag.","original_file_name":"original/a.png","result_file_name":"result/a.png"}
```

The 8-pair demonstration ZIP cannot satisfy 300 samples. Change cases are excluded.

## Verification

```powershell
python tests/check_pico_300_notebook.py
python tests/smoke_qwen_pico_prompts.py
```

The first command is offline: grouping, broad-mask rejection, prompt validation,
cache, deduplication, mixed automatic/manual quota/export, and widget callbacks
when widget libraries are installed. The second downloads the public Qwen model
and tests CPU inference on examples and held-out Pico instructions. The full
300-pair GPU/Colab workflow has not been run during authoring.
