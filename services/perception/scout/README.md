# SCOUT demo map

Scripts that turn the SCOUT dataset into the demo map: a top-down render of the whole scene, every camera with the ground it sees, and a crowd heat layer from person detections.

## Steps

```sh
D=data/scout; T=outputs/scout/topdown; C=outputs/scout/cam10
python services/perception/scout/topdown.py $D $T                 # map.png, map.json
python services/perception/scout/cameras.py $D $T                 # cameras.json, view-areas.npz, camera-map.png
python services/perception/scout/make_bundle.py $D $T $D/videos/s2_cam_10_preview.mp4 $C
caffeinate -i python services/perception/scout/sam_video.py CROP.mp4 $C/sam/left100 --ledger LEDGER.json --budget 2.40
python services/perception/scout/sam_tracks.py cam10 $C/detections.json $C/sam/left100:240:100:2:0:600 $C/sam/right100:560:100:2:600:852
python services/perception/scout/demo_map.py $T $C/bundle.json $C/detections.json $C/demo --tracks $C/detections-tracks.json
```

Several cameras: run the steps above per camera, then merge them. List the camera with the best view of shared ground first; it owns that ground.

```sh
python services/perception/scout/merge_cameras.py outputs/scout/multi outputs/scout/cam10 outputs/scout/cam9 outputs/scout/cam11 outputs/scout/cam12
python services/perception/scout/demo_map.py $T outputs/scout/multi/bundle.json outputs/scout/multi/detections.json outputs/scout/multi/demo \
  --tracks outputs/scout/multi/tracks.json --highlight cam9,cam10,cam11,cam12
```

## Person detection with SAM video

`sam_video.py` sends a video to the hosted SAM 3.1 model in short chunks. Rules learned on SCOUT camera 10:

- **Use short chunks (100 frames, 10 s).** SAM picks the people it tracks near the start of a clip and then mostly follows only those. In a 2,000-frame clip, people per frame fell from 8 in the first 200 frames to about 1 afterwards. In 100-frame clips the count stays steady and matches still-image detection.
- **At most 16 people per frame.** Split a busy frame into overlapping crops, one request each. `sam_tracks.py` gives each person to the crop whose band holds their feet, so no one is counted twice.
- **Long requests can fail mid-stream.** A 11,970-frame request failed after 2,322 frames. A failure now costs at most one chunk.
- **Resumable.** Each finished chunk is cached. A rerun skips finished chunks.
- **Keep the Mac awake.** Idle sleep pauses Wi-Fi and breaks long runs. Start SAM runs with `caffeinate -i` (macOS).
- **Network errors resume, API errors stop.** A timeout, reset, broken pipe or server-busy answer (HTTP 429, 502, 503, 504) waits 60 s and retries the same chunk, at most 5 times per run. A request that takes over 5 minutes counts as a timeout. Any other error answer from the API stops the run at once.
- **Spending.** One request at a time, a pause between requests. Every request goes into the usage ledger as soon as it ends. `--budget` refuses a run whose nominal cost is higher. Price: 0.20 USD per 1,000 frames, at least 50 frames per request.
- Track IDs restart in every chunk. The flow step joins tracks across chunk boundaries.

## Limits

- Camera heights are only 2.5 to 3.3 m, so use feet points (`--point feet`), not head points.
- The ground mesh has no ground under roofs. Camera 10 covers only the far plaza.
- View areas skip an occlusion test, so they can include ground hidden behind walls.
- The map keeps the mesh axes. North may not point up.
