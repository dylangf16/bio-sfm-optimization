
# Bio-SFM Optimization

## Overview

`bio.py` implements a bio-inspired algorithm for intelligent frame selection, designed to optimize the input for Structure from Motion (SFM) pipelines. The approach is based on swarm intelligence (bee colony behavior) and a compound eye model, simulating how bees collectively explore and evaluate visual information.

## What does it do?

The script receives a set of image frames (e.g., from a video or image sequence) intended for SFM. Instead of using all frames, which can be computationally expensive and redundant, it selects a subset of frames that maximize reconstruction quality while minimizing processing time.

## How does it work?

The algorithm models a swarm of artificial bees, each with a simulated compound eye. Bees explore the frame set, evaluate frame quality (sharpness, feature density, motion, spatial distribution), and communicate promising locations using a waggle dance mechanism. The collective decision process ensures that selected frames are both diverse and optimal for SFM.

### Input
- A directory containing image frames (JPG, PNG, BMP, etc.) to be used for SFM.

### Output
- A set of intelligently selected frame indices and filenames, saved to a summary JSON file and copied to an output directory. These frames are expected to yield high-quality SFM results with significantly reduced processing time.


## Note on COLMAP Usage

To evaluate processing times and reconstruction quality, COLMAP was used as the SFM pipeline. However, direct integration with COLMAP (via code or command-line) is not present in this repository. Instead, the COLMAP GUI was used manually to run experiments after frame selection. This approach was chosen for modularity and testing purposes.

## Authors

- **Dylan Gerardo Garbanzo Fallas** (Principal Author)
- **Luis Alberto Chavarría Zamora** (Co-author)
- **Pablo Soto-Quiros** (Co-author)

## Research Project

This work is part of a research project at Instituto Tecnológico de Costa Rica (Project No. 1440054).
