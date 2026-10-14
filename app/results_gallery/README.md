# Batch results gallery

Local viewer for `data/VIBE_test` and `data/Prior_test`, merged with images under `output/batch/<model>/<run>/`. Each card is Input + Output for the selected model. On Prior, an eye button on the output pane toggles ground truth. Likes, dislikes, and comments are stored in `app/results_gallery/feedback.json`.

```bash
python app/results_gallery/server.py
python app/results_gallery/server.py --port 8503
```

Open http://127.0.0.1:8502

From a laptop over SSH:

```bash
ssh -L 8502:localhost:8502 user@host
```

Samples still list when result PNGs are missing (empty frames). Click **Reload** after a batch run writes new images. JPEG thumbnails go in `app/results_gallery/.cache/` (Pillow; without Pillow the thumb route serves the original file).
