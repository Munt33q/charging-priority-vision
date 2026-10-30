# Model audit and demo status

The completed Kaggle run (`mluyxp11/charging-priority-train-retry`, version 4)
reported the following held-out test metrics:

| Metric | Result |
|---|---:|
| Precision | 0.721 |
| Recall | 0.694 |
| mAP50 | 0.767 |
| mAP50–95 | 0.655 |

The model performed well on ambulance, car, pickup-truck, and semi-truck
examples. Police performance is not reliable: the test split contained only six
police instances and reported mAP50 0.022. Fire-truck evaluation is also small
(12 instances), so it should be treated as provisional.

The demo was verified on a snowy DAWN adverse-weather image. It detected a car
with confidence 0.843 and assigned priority tier 1. The annotated result is in
`reports/demo_snow_priority.jpg`.

## Important limitations

- The current demo performs per-frame detection and tier assignment.
- Tracking, temporal confirmation, charger-zone occupancy, and persistent queue
  scheduling are not implemented yet.
- Model checkpoints and datasets are intentionally excluded from Git. Download
  the trained `best.pt` artifact from the Kaggle run and pass it with
  `--weights`.
- Tier 0 means the detected vehicle type is an emergency vehicle; the model does
  not determine whether it is actively responding to an emergency.
