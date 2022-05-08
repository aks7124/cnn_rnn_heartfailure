# Reproducibility Project for CS598 DL4H

Author1 and Author2 - {vganes7, akhils5}@illinois.edu

Reproducing paper [Incorporating Medical Code Descriptions for Diagnosis Prediction in Healthcare](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6921390).

## Dependencies

* Python 3.7
* pytorch 1.10.2
* tqdm 4.63.0
* scikit-learn 0.23.2
* numpy 1.19.2 
* scipy 1.5.2 
* pandas 1.1.5
* pickle5 0.0.11
* nltk 3.4.5

## Data Processing

1. Download MIMIC-III full dataset from https://physionet.org/content/mimiciii/1.4
2. Copy `ADMISSIONS.csv` and `DIAGNOSES_ICD.csv` to `data` directory
3. Download ICD-9 code descriptions from https://www.cms.gov/Medicare/Coding/ICD9ProviderDiagnosticCodes/codes
4. Copy `CMS32_DESC_LONG_DX.txt` to `data` directory
5. Run `preprocessing.py` to process data, it will create `.pkl` files in `full`, `test`, `train` and `validate` folders. The models use files generated in `full` directory and split the dataset for training and testing.
6. The CNN model uses pretrained icd code vectors, these vectors are generated using [fastText](https://fasttext.cc), these pretrained vectors `icd_code_embeddings.vec` can be found [here](https://drive.google.com/drive/folders/165J3Wbk75tmRJZ_XfFgo1TtCZLteqCup?usp=sharing). 
7. Copy `icd_code_embeddings.vec` to `data` directory

## Training the models

1. To train both CNN and RNN models - run `code/train.py`
2. To train models indicudually - run `code/CNN.py` or run `code/RETAIN.py`

# References

1. [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882) (Yoon Kim, 2014)
2. MIT’s [MIMIC-III]( https://mimic.physionet.org/)
3. [fastText](https://fasttext.cc) GitHub [repo](https://github.com/facebookresearch/fastText)
4. UIUC CS-598 Deep Learning for Healthcare coursework