# Cross-Site Domain Generalisation Study using PolypGen Dataset

## Folder Structure
```
PolypGen-DG/
├── requirements.txt
├── .gitignore
├── data/
    ├── centre_A/
    │   ├── positive/
    │   └── negative/
    │       ├── seq1/
    │       ├── seq2/
    ├── centre_B/
    │   ├── positive/
    │   └── negative/
    ├── centre_C/
    │   ├── positive/
    │   └── negative/
│
├── results/
    ├── csv/
    │   ├── epoch2.csv
    │   ├── epoch5.csv
    │   └── epoch10.csv
    ├── figures/
    │   ├── epoch2_acc.png
    │   ├── epoch2_drop.png
    │   ├── epoch5_acc.png
    │   ├── epoch5_drop.png
    │   ├── epoch10_acc.png
    │   └── epoch10_drop.png
    ├── epoch2.txt   
    ├── epoch5.txt  
    └── epoch10.txt   
│
├── explore_data.py
├── dataset.py
├── train.py 
├── model.py
├── main.py
└── analyse_results.py
```

## About the project
This project investigates **domain generalisation** for polyp detection using the **PolypGen Dataset** (C1-C3 sites).

## About the dataset
- PolypGen dataset DOI: 10.7303/syn26376615  
- Download link: https://www.synapse.org/#!Synapse:syn45200214  
- Citation (mandatory if used):

1. Ali, S., Jha, D., Ghatwary, N. et al. Sci Data 10, 75 (2023).  
2. Ali, S., Ghatwary, N., Jha, D. et al. Sci Rep 14, 2032 (2024).  
3. Ali S, Dmitrieva M, Ghatwary N, Bano S, Polat G, et al. Med Image Anal, 2021.
