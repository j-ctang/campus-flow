# Offline concentration experiment

A disposable local experiment using two existing SAM responses for one SCOUT camera. It produces static exact-frame concentration, one-second mean occupancy, and ten-second mean occupancy images. Geometry remains provisional; these are diagnostic previews, not validated population measurements.

The experiment excludes stairs, ambiguous surface intersections, truncated masks and observations outside a manually selected upper-landing polygon. Colors share one scale across both clips and smoothing settings. Gray means excluded or unobserved. Empty processed frames contribute zero to interval means. Clip-local IDs are never treated as unique people.

## Run locally

Requires Node.js compatible with `@meta-sam/parser` 0.1.0 and Python with NumPy, OpenCV, Trimesh and Rtree. The tested Python environment used NumPy 2.5.4, opencv-python-headless 5.0.0.93, Trimesh 5.2.0 and Rtree 1.4.1.

From the repository root, with the official parser installed:

```sh
node services/perception/concentration/import_cached.mjs RESPONSE_1200.json RESPONSE_3000.json outputs/concentration/cached-observations.json
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 nice -n 15 python services/perception/concentration/experiment.py --region REGION.json
```

`SAM_PARSER_PATH` can point to an existing parser package directory instead of installing another copy. The two input responses must correspond to source frames 1200–1299 and 3000–3099 respectively. The source mapping is fixed for this experiment. Region JSON contains a `pixel_polygon` array of preview-image `[x, y]` vertices. Input video, calibration, ground mesh and official timestamps must be available under ignored `data/scout/`.

Generated artifacts stay under ignored `outputs/concentration/`. `report.json` records input hashes, exclusions, time conventions, smoothing sensitivity and validation status. `mapped-observations.json` preserves source-frame and capture-timestamp provenance. Before ray queries, the mesh is cropped to triangles that selected camera rays can reach (about 5% of the ground mesh for the upper landing). A hashed projection cache avoids repeating mesh queries. There is no inference or network code.

```sh
node --test services/perception/tests/import_cached.test.mjs
python -m unittest discover -s services/perception/tests -p 'test_*.py'
```

## Interpretation limits

Foot locations are heuristic mask-bottom estimates. No calibrated model confidence is available. Preview-time averages use uniform video frames; official capture timestamps are retained separately. Smoothing uses grid cells in a square local-coordinate footprint. The color value is discrete observation mass per cell, not people per square metre. Occupancy rewards longer presence and does not count unique visitors. Two short, disjoint clips do not establish typical campus use, complete detection or cross-clip identity.
