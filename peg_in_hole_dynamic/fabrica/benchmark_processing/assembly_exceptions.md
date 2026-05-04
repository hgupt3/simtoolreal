# Assembly Processing Exceptions

Per-assembly rules and overrides for the benchmark processing pipeline.

## Global Rules
- **Centering**: use bounding box center, not mass centroid (screws/bolts bias centroid toward dense end)
- **Tiebreaker**: when bbox extents are equal, prefer Y > X > Z as longest canonical axis

## car
- Parts 0, 3, 4, 5 (wheels): nearly cubic, need Y→X tiebreaker to align axle axis correctly

## cooling_manifold
(TBD)

## duct
(TBD)

## gamepad
(TBD)

## plumbers_block
(TBD)

## stool_circular
(TBD)
