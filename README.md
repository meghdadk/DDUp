<h1 align="center">DDUp; Detect, Distill and Update: Learned DB Systems Facing Out of Distribution Data</h1>

[![License: Apache](https://img.shields.io/badge/License-Apache-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Introduction
The code includes instantiation of three learned database systems: DBEst++, Naru, and TVAE.

To develope DDUp, we have used the source codes published by the respective works at below:

[DBEst++](https://github.com/qingzma/DBEstClient )

[Naru](https://github.com/naru-project/naru)

[TVAE](https://github.com/sdv-dev/CTGAN)


## Setup

To increase reproducibility, we have created subdirectories for each dataset within each model. The latest codes for each case could be found inside the directory

For DBEst: The train/update procedures are located in the MDN.py. The evaluation procedures are located in benchmarking.py

For Naru: The train/update procedures are located in the incremental_train.py. The evaluation procedures are located in eval_model.py

For TVAE: The train/update procedures are located in the tvae_train.py. The evaluation procedures are located in benchmarking.py

The codes are tested for Python3.6 and Pytorch 1.9

## Datasets
The experiments in the paper are for six public datasets. For DBest++, we have used a query template with two columns and have added the modified datasets in the related folders. For TVAE, we have used a samller (1m) sample of DMV dataset, as it was too expensive to train TVAE on the full data.

The link to some of the datasets:

[Census](https://archive.ics.uci.edu/ml/datasets/census+income)

[Forest](https://archive.ics.uci.edu/ml/datasets/covertype)

[DMV](https://www.dropbox.com/s/akviv6e9xi0tl00/Vehicle__Snowmobile__and_Boat_Registrations.csv)

## References

Reference
If you find this repository useful in your work, please cite our SIGMOD23 paper:

@article{kurmanji2023detect,
  title={Detect, Distill and Update: Learned DB Systems Facing Out of Distribution Data},
  author={Kurmanji, Meghdad and Triantafillou, Peter},
  journal={Proceedings of the ACM on Management of Data},
  volume={1},
  number={1},
  pages={1--27},
  year={2023},
  publisher={ACM New York, NY, USA}
}