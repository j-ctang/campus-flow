# How the app works

This page fixes what goes into CrowdMap, what comes out, and how long it takes. Build the viewer against it. The numbers come from real runs on 8 SCOUT cameras and on CityStreet.

## In one line

A planner gives us a top-down map and footage from cameras that watch the same area. After a short setup and some processing time, they get one map that shows where people gather and which way they walk.

## What the user puts in

| Input | Example | Who makes it | Time |
| --- | --- | --- | --- |
| Top-down map image | Aerial photo, site plan, or a render of a 3D scan | User | Minutes, if they already have one |
| Map scale | Two points on the map and the distance between them in meters | User clicks | 1 minute |
| Camera videos | One video per camera, recorded at the same time | User uploads | Upload time only |
| Ground point pairs | 4 spots per camera, each marked once in a video frame and once on the map | User clicks | About 5 minutes per camera |
| Camera position | Where the camera stands on the map. Required; draws the icon and corrects head points | User clicks | Seconds |
| Camera height | Meters above the ground, optional; needed for head points | User types | Seconds |

All of this is one input bundle, described in [`services/perception/flowmap/README.md`](../services/perception/flowmap/README.md). The viewer needs a setup screen to collect it: upload the map, set the scale, then for each camera upload the video, click where the camera stands on the map, and click 4 matching spots in the frame and on the map.

Rules for good ground points: flat ground, spread wide, not on one line, away from the horizon.

## What happens to it

1. Cut each video into 100-frame chunks (10 seconds at 10 fps).
2. Send each chunk to the hosted SAM 3.1 video model, which finds and follows every person.
3. Turn each person's feet or head into a spot on the map, using the 4 ground point pairs.
4. Add up all spots over time into a heat map, in people per square meter.
5. Join each person's spots over time into a track, and turn tracks into flow arrows, in people per minute.
6. Merge the cameras. Ground seen by two cameras belongs to one camera only, so nobody is counted twice.

## How long it takes and what it costs

Measured on SCOUT: 20-minute videos at 10 fps, 852x480, 120 chunks per camera.

| Step | Time | Cost |
| --- | --- | --- |
| Setup by the user | About 5 minutes per camera, plus about 5 minutes for the map | Free |
| SAM, per camera | About 28 minutes for 20 minutes of video (each chunk takes about 10 s of SAM time, then a 10 s pause) | About 2.40 USD per 20 minutes of video |
| Busy cameras | Twice the time and cost: SAM follows at most 16 people per frame, so busy views are split into 2 crops | About 4.80 USD per 20 minutes |
| Heat map, flow arrows and drawing | About 7 seconds for 8 cameras | Free |

Rules of thumb:

- Processing takes about 1.4 times the video length, per camera.
- Cost is about 0.12 USD per minute of video, per camera, or twice that for busy views.
- Cameras run one after another today. 8 cameras of 20 minutes take about 4 hours and cost about 19 USD. Running cameras in parallel would bring this near 30 minutes, if the SAM service allows it. That is not tested yet.

So the app is a job, not an instant answer. The viewer needs a progress state: chunks done out of total, time left, and money spent so far. Results can be shown per camera as each one finishes.

## What comes out

The viewer reads one result folder. `services/perception/flowmap/export.py` writes the layers below from a prepared bundle, cached detections, and optional tracks. It makes no SAM requests. `timeline.json` is not exported yet.

| File | Contents | Layer in the viewer |
| --- | --- | --- |
| `map.png` | The top-down base map | Base layer |
| `map.json` | `{"m_per_px": 0.1, "width": 1753, "height": 3576, "north_deg": 0}` | Scale bar, north arrow |
| `heat.png` | Heat as a transparent color image, same size as `map.png`; transparent where no camera sees | Heat layer |
| `coverage.png` | Ground no camera sees, as a gray mask | Gray layer |
| `cameras.json` | Per camera: id, map position, view direction, height, view area polygon in map pixels, area in m², people per frame | Camera icons, view areas, click-for-numbers |
| `flow.json` | List of arrows: map position, direction, people per minute | Flow layer |
| `report.json` | Totals: frames, minutes, people per frame (mean and peak), covered m², peak people per m² and where, flow summary | Summary numbers |
| `timeline.json` | Optional. People in view per minute, and heat per time window | Time slider |

Example `flow.json` entry: `{"x": 1427, "y": 1842, "dx": 0.0, "dy": -1.0, "people_per_min": 6.2}`. Positions are map pixels; `dx, dy` is a unit vector in map pixels, with y pointing down.

### Export a result folder

```sh
python services/perception/flowmap/export.py BUNDLE.json DETECTIONS.json NEW_OUTPUT_DIR \
  --tracks TRACKS.json --fps 10 --cell-m 4 --north-deg 0
```

The output directory must not already exist. The exporter publishes it only after every file is written. Omit `--tracks` for still-image datasets: `flow.json` is then an empty list and `report.json.flow` is `null`.

All images have the base map's original dimensions, with no legend or border. Map positions use x right and y down. `--north-deg` is the clockwise angle from image up toward north; omit it when unknown (`map.json.north_deg` is `null`). It records orientation and does not rotate the map.

`heat.png` uses the OpenCV JET palette with density normalized to `report.json.heat_max_people_per_m2` (the covered pixels' 99.5th percentile, floored at 1e-9). Alpha is `0.75 * normalized_density^0.6`, zero outside coverage. This is a display layer, not raw density data. `coverage.png` is opaque gray outside coverage and transparent inside.

`cameras.json` is a list. Each camera has `id`, `position`, `direction`, `direction_source`, `height_m` (or `null`), `view_area`, `area_m2`, and `people_per_frame_mean`. Direction points from the camera position toward the view polygon's centroid; it is an approximate display heading, not calibrated optical orientation. Areas and counts use exclusive ownership: the first listed camera owns overlapping pixels. `view_area` describes the full polygon before overlap removal.

Heat, camera totals, and flow use only frame keys shared by every camera. Duration is shared frame count divided by `--fps`, not elapsed wall time. Flow requires integer frame keys and track frame numbers on the same time base. With multiple cameras, tracks must be nested under camera IDs; flat track files are accepted only for a single-camera bundle. Missing frames are excluded, so a gapped recording does not represent its full elapsed interval.

Coverage follows the bundle's flat-ground view polygons. Unlike the SCOUT-specific demo, this generic exporter does not additionally intersect mesh-derived masks; its totals may differ from that demo. It does not infer visibility behind obstacles.

## What the viewer must show

Essential, for the first version:

- The base map, north up, with a scale bar.
- The heat layer and its legend in people per square meter.
- Gray for ground no camera sees.
- Camera icons and view areas. Clicking a camera shows its area and its numbers.
- Layer switches for heat, flow, cameras and gray.
- A summary: people per frame, peak density, covered area, minutes of video.
- Data credits.

Next:

- Flow arrows, with arrow width meaning people per minute.
- A time slider over `timeline.json`.
- The setup screen and job progress described above.

## Measured results to expect

| Run | Cameras | Ground covered | People per frame (mean, peak) | Flow arrows |
| --- | --- | --- | --- | --- |
| SCOUT, 20 minutes | 8 | 3,119 m² | 23.3, 44 | 117 |
| CityStreet, 200 frames | 3 | 882 m² | 68.3, not measured | none: CityStreet is still frames, not video |

On CityStreet, the heat map matches the official ground labels with map correlation 0.81, and 72% of people are placed within 2 m.

## Limits

- Ground must be flat. Each slope, such as stairs, needs its own set of ground points.
- More than 16 people in a view means the view must be split, which doubles time and cost.
- SCOUT images may not be shared. Public demos use CityStreet.
