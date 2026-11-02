# Campus Flow

A hackathon prototype exploring how aggregate pedestrian observations can inform campus transit planning.

## What we're building

- Measure pedestrian crossings from authorized video using SAM 3.1.
- Connect observations to calibrated scene coordinates.
- Compare shuttle scenarios under an equal operating budget.

UCI motivates the problem. EPFL's SCOUT dataset is a candidate perception benchmark; its observations are not UCI travel demand. Simulated demand and results must be labeled explicitly.

## Current status

The repository is collaboration scaffolding. No SAM inference or transit improvement has been demonstrated yet. Dataset access and use permissions, including hosted processing, remain under review.

## Working areas

| Area | Directory | Responsibility |
| --- | --- | --- |
| Demo interface | `apps/web/` | Video replay, scene map, result presentation |
| Perception and geometry | `services/perception/` | SAM adapter, calibration, aggregate counts |
| Transit scenarios | `services/planner/` | Demand assumptions, equal-budget comparisons |

Proposed stack: a TypeScript web interface and Python batch processing. Frameworks and shared interfaces will be agreed before implementation.

## Join the project

1. Ask the repository owner for collaborator access; share your GitHub username.
2. Clone this repository with `git clone https://github.com/j-ctang/campus-flow.git`.
3. Pick one scoped task and create a branch such as `feature/crossing-counts`.
4. Open a pull request describing the change and how you checked it.

Use GitHub Issues for task ownership and pull requests for review. Keep changes focused so teammates can work independently.

## Data and credentials

Never commit API keys, billing details, source footage, dataset archives, or restricted derived assets. Use local environment variables for credentials. Public dataset access does not establish redistribution or cloud-upload permission. Check applicable terms before downloading or processing assets.

No deployed inference endpoint or automated API job is configured. The owner permits free promotional credits only, with a maximum total of $150. Coordinate all API runs with the owner; shared-team usage can exhaust the same credits. Do not launch unattended batches or automatic retries. Billing may lag by an hour.

## Hackathon contributions

Clearly identify code and features built during the event. Credit existing models, libraries, and data. Do not present supplied annotations as model-generated results. The repository is public for submission; it currently grants no additional license beyond applicable rights.
