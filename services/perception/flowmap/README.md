# Flow map pipeline

Turns camera footage of people into a crowd concentration heat map and walking-flow arrows drawn on a top-down map of the area.

> **Input format version 0.1. It may change.**

## Input bundle

A bundle is a folder with one JSON file, one top-down map image and one video or frame folder per camera.

```json
{
  "version": "0.1",
  "map": {
    "image": "map.png",
    "scale": { "points": [[120, 400], [192, 400]], "meters": 10 }
  },
  "cameras": [
    {
      "id": "camera1",
      "video": "camera1.mp4",
      "frame_size": [2704, 1520],
      "position": [88, 330],
      "height_m": 8.3,
      "ground_points": [
        { "frame": [512, 1210], "map": [140, 371] },
        { "frame": [2190, 1180], "map": [251, 322] },
        { "frame": [2330, 760], "map": [289, 455] },
        { "frame": [640, 790], "map": [175, 498] }
      ]
    }
  ]
}
```

| Field | Meaning |
| --- | --- |
| `map.image` | Top-down picture of the area, such as an aerial photo or a site plan. Paths are relative to the JSON file. |
| `map.scale` | Two points on the map and the real distance between them in meters. |
| `cameras[].id` | Unique camera name. |
| `cameras[].video` | Video file or frame folder for this camera. |
| `cameras[].frame_size` | Frame width and height in pixels. |
| `cameras[].position` | Where the camera stands on the map, in map pixels. Used for the camera icon. |
| `cameras[].height_m` | Optional. Camera height above the ground in meters. Needed for head points (`--point head`). |
| `cameras[].ground_points` | At least four pairs. Each pair is one spot on the ground, such as a crosswalk corner, marked in a camera frame (`frame`) and on the map (`map`). |
| `cameras[].view_area` | Optional. Polygon in frame pixels of the ground this camera should cover. By default, the area inside the marked ground points is used. |

### Marking ground points

- Pick spots on flat ground that you can find in both the frame and the map, such as road markings, curb corners or manhole covers.
- Spread the spots wide across the ground the camera sees. Avoid spots near the horizon, where one pixel spans many meters.
- Do not put three spots on one straight line.
- Each sloped area, such as stairs, needs its own set of points. Version 0.1 handles flat ground only.

## Detections

Person detection runs separately and writes one JSON file:

```json
{ "camera1": { "0001": [[x1, y1, x2, y2], ...], "0002": [] } }
```

Frame keys must match across cameras: the same key means the same moment. Only keys that every camera has are used.

## Run

```sh
python services/perception/flowmap/heatmap.py BUNDLE.json DETECTIONS.json OUTPUT_DIR --sigma-m 1.5 --point head
```

`--point feet` (default) uses the bottom of each person box. `--point head` uses a spot near the top of the box and needs `height_m` for every camera. Use head points for crowds, where feet are often hidden.

## Output

- `heatmap.png`: concentration over the map. Blue means few people and red means the most. Camera icons and view areas are drawn on the map, and gray areas have no camera coverage.
- `report.json`: frames used, people per frame, covered area in square meters, and the peak density and its location.

## How it works

1. The marked point pairs give each camera a homography, a fixed conversion from frame pixels to map pixels that holds for flat ground.
2. Each person box gives one image point. With feet, the bottom center is converted to the map. With heads, a point 8% of the box height below the top center is converted as if it lay on the ground. That spot is beyond the person, so it is pulled back toward the camera's map position by the ratio of person height (1.75 m) to camera height.
3. Each map pixel belongs to one camera only, the first listed camera that covers it, so overlapping cameras never count a person twice.
4. Every person adds one unit of mass, spread by a Gaussian of `--sigma-m` meters. The mean over frames gives people per square meter.

## Flow arrows

`flow.py` draws where people walk, as arrows on the map.

```sh
python services/perception/flowmap/flow.py bundle.json tracks.json out/ --camera cam1 --fps 10 --cell-m 4
```

Tracks JSON: `{"<camera id>": {"<track id>": [[frame, x, y, top], ...]}}`, with `x, y` the foot point (box bottom center) and `top` the box top, in frame pixels. `top` is optional. A file with only track IDs at the top is read as one camera, named by `--camera`.

1. Hidden feet are fixed. When legs are hidden behind someone, the box bottom jumps up and the foot lands meters away on the map. A box shorter than 85% of its usual height (over 1 s around it) gets its foot at top plus usual height. This needs `top`.
2. Each foot point goes through the camera's homography to the map.
3. A track that ends is joined to one that starts within 1 s, close by. Track IDs that restart (for example in every SAM chunk) then make one walk.
4. Tracks are split at gaps over 1 s and at jumps faster than walking plus a few pixels of noise. Far from the camera one pixel spans more ground, so more noise is allowed there.
5. Each piece is smoothed (median, then mean, over 0.5 s). Pieces shorter than 1.5 s or moving less than 1 m are dropped.
6. The map is cut into cells (`--cell-m`, default 4 m). A track adds one crossing to each cell it walks through, in one of 8 direction bins. Wandering inside a cell does not count.
7. Each cell gets one arrow per main direction. Neighboring bins merge, so a stream between two bins gives one arrow. Opposite streams stay separate and never cancel.
8. An arrow needs 20% of the tracks of a busy cell (the 95th percentile of cells with flow), at least 3, so the threshold grows with the crowd. `--min-count` sets a fixed number instead. An arrow also needs at least 15% of the busiest arrow's rate. Arrows show only where at least half the cell is covered.

On SCOUT camera 10 (20 minutes), 2 m cells with 3 tracks drew 280 arrows and hid the lanes; 4 m cells keep the main lanes. On 8 SCOUT cameras the scaled threshold is 27 tracks and draws 117 arrows; the answer key, with 2.5 times as many people, gets 63 and 110 arrows. A fixed 8 drew 166 arrows, mostly minor streams.

Arrow width is people per minute going that way. Each arrow sits a little to the right of its cell center, so two-way paths show as two lanes. Speed is not shown.

Output: `flow.png` (arrows on the map) and `flow.json` (minutes of footage, tracks kept and dropped, arrow count, top rate).

`scout/demo_map.py --tracks` draws the same arrows over the heat layer.

## Limits

- Flat ground only.
- Strong lens distortion, such as fisheye, bends straight lines, and a homography cannot follow that. Expect about 1 m of error with wide-angle cameras.
- In dense crowds, feet are often hidden, so box bottoms can sit above the real foot position. Head points avoid this.
- Head points assume everyone is 1.75 m tall and need the camera position on the map to be accurate.
- Flow from tracks far from the camera is noisy: one pixel there can be half a meter of ground.
- Tracks are joined by position only. Two people who meet at a track restart can swap.

## Viewer export

```sh
python services/perception/flowmap/export.py BUNDLE.json DETECTIONS.json NEW_OUTPUT_DIR \
  --tracks TRACKS.json --fps 10 --cell-m 4
```

Writes `map.png`, `map.json`, transparent `heat.png`, gray `coverage.png`,
`cameras.json`, `flow.json` (arrow entries), and `report.json`. The existing
`flow.py` command continues to write its summary to its own `flow.json`.
Use separate output folders for these commands.

The export directory must be new. Omit tracks for heat-only results.
Set `--north-deg` only when known; otherwise north is `null`. No API requests,
video decoding, or mesh loading occur. See [the viewer contract](../../../docs/app.md#export-a-result-folder)
for field meanings, color scale, coverage, timing, and overlap rules.
