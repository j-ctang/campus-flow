# Contributing

This page lists open tasks you can pick up now, without waiting for the pipeline work in progress. Each task says what to deliver and when it is done.

## Goal

One top-down map of the whole SCOUT campus scene:

- a heat layer showing where people concentrate, only on ground a camera sees
- flow arrows showing where people walk and in which direction
- every camera as an icon with the ground it sees
- gray for ground no camera covers

CityStreet, a busy Hong Kong street, proves the method on a dense crowd against its official ground labels.

The pipeline (heat map, flow arrows, camera map) is in pull request #2. Camera 10 already has a full 20-minute heat map and flow arrows.

## Tasks

### 1. Build the map viewer (UI and UX, high priority)

Build the page people will look at, in `apps/web/`.

- Show the base map with layers you can turn on and off: heat, flow arrows, cameras and their view areas.
- Add a legend: gray means no camera, heat is people per square meter, arrow width is people per minute.
- Click a camera to see its view area and its numbers.
- Later: a time slider, because the pipeline keeps detections per frame.

You do not need the data to start. Make mock files with this shape:

- `map.json`: `{"m_per_px": 0.1, "width": 1753, "height": 3576}` and a placeholder `map.png`
- `cameras.json`: `{"cameras": {"cam10": {"map_px": [1701, 1726], "forward": [-0.4, -0.86, -0.3]}}}`
- `report.json`: `{"frames": 11970, "people_per_frame_mean": 10.0, "peak_people_per_m2": 0.07}`
- `flow.json`: a list of arrows, each with a map position, a direction and a rate in people per minute

Done when the page shows all layers from the mock files and the legend reads clearly to someone new.

### 2. Mark ground points for more cameras

Each camera needs four ground points that a person marks by hand: one spot seen in the video and the same spot on the map. This is the input for the next 21 cameras.

- Pick cameras 9, 11 and 12 first. They sit next to camera 10.
- For each camera, mark four spots on flat ground, far apart, such as path corners or line ends. Avoid spots near the horizon.
- Write them in the bundle format from pull request #2 (`services/perception/flowmap/README.md`).

Done when each camera has a bundle file and a screenshot of the four spots in the frame and on the map.

### 3. Check the detections by hand

- Pick 20 frames spread over the camera 10 video.
- Count the clearly visible people standing on the plaza in each frame.
- Compare with the detection counts for the same frames.

Done when there is a table with frame number, your count, the detection count, and missed or extra people.

### 4. Check the flow arrows against the video

- Watch 5 minutes of the camera 10 video.
- Sketch the main walking paths and their directions on the map.
- Compare with the flow arrows.

Done when there is a short note: which arrows match, which are missing, which look wrong.

### 5. Turn the map north-up

The map keeps the 3D mesh axes, so north may not point up.

- Find three or more landmarks, such as building corners, both on the map and on OpenStreetMap.
- Work out the rotation angle.

Done when we have the angle and the landmark list.

### 6. Find more crowd data

We need footage of dense crowds, ideally on a campus, with camera calibration or people positions on the ground.

Done when there is a table of five or more datasets with: link, license, whether redistribution is allowed, number of cameras, crowd density, and labels included.

### 7. Prepare the demo story

- Write the three-minute story: the problem, what the map shows, and what a planner could decide from it.
- Pick the views to show and their order.

Done when there is a draft slide outline.

## Setup

The requirements file and tests arrive with pull request #2.

```sh
python3 -m venv .venv
.venv/bin/pip install -r services/perception/requirements.txt
.venv/bin/python -m unittest discover -s services/perception/tests
```

Get the datasets yourself (see `README.md`). They go in the ignored `data/` folder. Outputs go in the ignored `outputs/` folder.

## Data rules

- SCOUT is for academic and research use only. Never commit or upload footage, frames, mesh renders, or files made from them. Request your own SCOUT access.
- CityStreet is CC BY-NC-SA 4.0.
- Person detection runs on paid SAM credits that the team shares. Ask before running new detection, and reuse cached results.

## Machine limits

Mesh and video jobs can overload a laptop. Cap threads (`OMP_NUM_THREADS=2`), run with `nice -n 15`, and run one heavy job at a time.
