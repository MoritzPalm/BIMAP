### About The Project

Calcium imaging videos often suffer from motion artifacts caused by subject movement. 
Correcting this motion is a critical preprocessing step in neuroscience research.
We present a novel method for motion correction in the utilization of the 
general-purpose point tracking method CoTracker 
and compare it with established methods like ANTs, LDDMMs and NoRMCorre.
Additionally, we provide a framework to systematically evaluate and compare the 
performance of different motion correction algorithms across various datasets and parameter settings.

### Getting Started
Prerequisites:
- Python 3.11 or higher
- conda / mamba for the NoRMCorre installation
- uv as a dependency manager (for installation instructions see https://docs.astral.sh/uv/getting-started/installation/)

Installation:
Currently, the package is not available on PyPI.
To install the package, clone the repository and use uv to install the dependencies:
```bash
git clone https://github.com/MoritzPalm/BIMAP
cd BIMAP
uv install
```
#### NoRMCorre Installation
NoRMCorre needs to be installed separately using the instructions here: https://github.com/flatironinstitute/CaImAn
Make sure to install caimanmanager as well. 
Depending on the install location of caiman_data, you need to adapt the paths in src/bimap/mc_normcorre.py for both 'CAIMAN_TEMP' and the working directory in run().
Before running, move the mc_normcorre_callee.py file to the caiman_data/demos/notebooks folder.

### Usage
To run the motion correction algorithms and evaluate their performance, use the following command:
```bash
uv run orchestrator.py --config experiments.yaml --output ../../data/output
```
Input videos should be placed under:
```text
BIMAP/data/input/
├── low_movement/
└── strong_movement/
```
It will generate output files in the following structure: 
<details>
<summary><strong>Folder structure</strong> (click to expand)</summary>

```text
output/
└── <method>/                  # ants, normcorre, cotracker, lddmms
    └── <experiment>/          # template_index, gaussian_filtering, ...
        ├── low/               # low-motion recordings
        │   └── vX/
        │       └── run_<UUID>/
        │           ├── config.json
        │           ├── result.json
        │           └── artifacts/
        │               └── filename.(tif|mp4)
        └── strong/            # strong-motion recordings
            └── vX/
                └── run_<UUID>/
                    ├── config.json
                    ├── result.json
                    └── artifacts/
                        └── filename.(tif|mp4)
```
</details>

#### Other Scripts
```BIMAP/src/bimap/plotting``` contains all scripts to generate the plots used in the paper.

```BIMAP/src/notebooks``` contains different jupyter notebooks used in the early stages of development.


### License
Distributed under the MIT License. See `LICENSE.txt` for more information.




