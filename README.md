# rsDenoise
configurable resting-state denoising pipeline

cloned from HCP_MRI-behavior project

- `HCP_helpers.py` is the "clean" set of python functions published publicly

- `helpers.py` is the "development" version with a few more pipelines defined

- `Conte_[...].ipynb` are example uses

Currently the pipelines rely on the [HCP directory structure](https://wiki.humanconnectome.org/display/PublicData/How+To+Handle+Downloaded+HCP+Data "HCP directory structure"), i.e. paths are pre-defined in some of the functions. The goal here is to make the pipelines more flexible and allow other directory structures, and check/clean/document them further. 

It may be best to work from `HCP_helpers.py` which is the cleanest version, yet refer to `helpers.py` for preprocessing pipeline definitions.

Example output [here](https://caltech.box.com/s/t7zaw05wcnudl7b2jvn69wa2fvzlcc4j "outputs,preGLM0 pipeline")

