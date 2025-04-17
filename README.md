# MTLTranSyn

## Requirements

- python=3.7
- cuda=11.3
- pytorch=1.11.0
- scipy=1.7.3
- scikit-learn=1.0.2
- pandas=1.3.5

## Start

Run the AEprocess.py first to pre-train a drug encoder and a cell line encoder, and then run the run.py to train the model.

## Data

**drugs.csv**:  Information of 3118 drugs.

**cell_lines.csv**:  Information of 175 cell lines.

**drug_features.csv**:  Features of  3118 drugs, 1213-dimensional vector for each drug.

**cell_line_features.csv**:  Features of 175 cell lines, 5000-dimensional vector for each cell lines.

**oneil_summary_idx.csv**:  22 737 samples from O'Neil，each sample consists of two drugs id, a cell line id, synergy score of the drug combination on the cell line, respective sensitivity scores of the two drugs on the cell line.  

**casestudydata.csv**:269,544 samples from DrugComb, excluding drug pairs from the O'Neil dataset.  It includes 15,147 unique drug pairs , 2,036 drugs, and 152 cell lines.

## Training files

**AEtrain.py**: used to pre-train a drug encoder and a cell line encoder.

**main.py**: used to train MTLTranSyn.

**GBMtrain.py**: used to train Gradient Boosting Machine.

**RFtrain.py**: used to train Random Forest.

## Source code of the comparative methods

MTLTranSyn:https://github.com/TOJSSE-iData/MTLSynerg

PRODeepSyn: https://github.com/TOJSSE-iData/PRODeepSyn

TranSynergy: https://github.com/qiaoliuhub/drug_combination

DeepSynergy: https://github.com/KristinaPreuer/DeepSynergy