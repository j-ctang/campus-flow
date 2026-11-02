# Flow map pipeline

Turns camera footage of people into a crowd concentration heat map drawn on a top-down map of the area.

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
python services/perception/flowmap/heatmap.py BUNDLE.json DETECTIONS.json OUTPUT_DIR --sigma-m 1.5
```

## Output

- `heatmap.png`: concentration over the map. Blue means few people and red means the most. Camera icons and view areas are drawn on the map, and gray areas have no camera coverage.
- `report.json`: frames used, people per frame, covered area in square meters, and the peak density and its location.

## How it works

1. The marked point pairs give each camera a homography, a fixed conversion from frame pixels to map pixels that holds for flat ground.
2. The bottom center of each person box is taken as the person's feet and converted to the map.
3. Each map pixel belongs to one camera only, the first listed camera that covers it, so overlapping cameras never count a person twice.
4. Every person adds one unit of mass, spread by a Gaussian of `--sigma-m` meters. The mean over frames gives people per square meter.

## Limits

- Flat ground only.
- Strong lens distortion, such as fisheye, bends straight lines, and a homography cannot follow that. Expect about 1 m of error with wide-angle cameras.
- In dense crowds, feet are often hidden, so box bottoms can sit above the real foot position.
