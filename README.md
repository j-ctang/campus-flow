# Campus Flow

Campus Flow studies how people move around EPFL’s campus in Switzerland. It maps observed pedestrian movement and summarizes flows between campus areas to help inform future transit planning.

## Data credits

Campus Flow uses the [SCOUT dataset](https://scout.epfl.ch/) from EPFL’s Computer Vision Laboratory (CVLAB), including video, camera calibration, and scene geometry.

Dataset reference: Martin Engilberge, Ivan Vrkic, Friedrich Wilke Grosche, Julien Pilet, Engin Turetken, and Pascal Fua. *[One Graph to Track Them All: Dynamic GNNs for Single- and Multi-View Tracking](https://arxiv.org/abs/2507.08494)* (2025).

SCOUT is available for academic and research use under its [terms of use](https://scout.epfl.ch/download). Dataset files are not included in this repository; teammates should request access directly from SCOUT.

The [CityStreet dataset](https://visal.cs.cityu.edu.hk/research/citystreet/) from City University of Hong Kong's VISAL lab supplies a denser crowd for testing. It contains 500 synchronized multi-view frames of a busy Hong Kong street intersection, with camera calibration and ground-plane people labels. CityStreet is released under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) and its bundled copyright notice.

Dataset reference: Qi Zhang and Antoni B. Chan. *[Wide-Area Crowd Counting via Ground-Plane Density Maps and Multi-View Fusion CNNs](https://arxiv.org/abs/2012.00946)* (CVPR 2019).

To download CityStreet from its official source into the ignored `data/citystreet/` folder, run `scripts/get-citystreet.sh`. The script requires [uv](https://docs.astral.sh/uv/) and about 32 GB of free disk space. It skips the DukeMTMC and PETS2009 archives bundled in the same folder.
