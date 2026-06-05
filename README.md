# face-recognition-watchlist

A computer vision pipeline that captures video, detects and recognizes faces, tracks them across
frames, and raises alerts when a recognized identity appears on a configured watchlist. A dashboard
provides a live monitoring view.

## Project layout

```
face-recognition-watchlist/
├── src/
│   ├── capture.py      # read frames from a camera or video file
│   ├── detector.py     # locate faces and return bounding boxes
│   ├── recognizer.py   # embed faces and match against enrolled identities
│   ├── tracker.py      # associate detections across frames into stable tracks
│   ├── watchlist.py    # load watchlist config and decide on alerts
│   └── pipeline.py     # orchestrate the end-to-end loop
├── dashboard/
│   └── app.py          # live monitoring web UI
├── data/
│   ├── enrolled/       # consented enrolled face images (gitignored)
│   └── watchlist.json  # watchlist names + similarity threshold
├── docs/
├── requirements.txt
└── README.md
```

## Configuration

`data/watchlist.json` controls who is on the watchlist and how strict matching is:

```json
{
  "watchlist": ["bob"],
  "similarity_threshold": 0.65
}
```

## Running

The project is run as a package:

```bash
python -m src.pipeline
```

## Development setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Privacy

Enrolled face images under `data/enrolled/` are consented personal data and are intentionally
excluded from version control via `.gitignore`. Do not commit them to a public repository.
