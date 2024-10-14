# EHR Data Preprocessing



Here lies source code for extracting and preprocessing EHR data from MIMIC-III and MIMIC-IV. The preprocessing codes are heavily borrowed from Fiddle pipeline (Tang et al., 2020).


## Datasets

The datasets can be acquired via the following links:

MIMIC-III: https://physionet.org/content/mimiciii/1.4/

MIMIC-IV: https://physionet.org/content/mimiciv/3.1/

Permission should be requested as the prerequisite of the sources.

### Preprocessing Steps

The initial step is to run 

```bash 
python combine.py
```

to combine the MIMIC-III and MIMIC-IV databases with necessary modifications. Then, modify the yaml configuration file to reflect data paths.


Subsequently, please stick to the first two steps of the following link:

https://github.com/MLD3/FIDDLE-experiments/tree/master/mimic3_experiments


Finally, the time invariant and variant data will be available at ./mimic_preprocessed/features/outcome=pretrain,T=48.0,dt=1.0/, represented by S.npz and X.npz, respectively.

Note that the whole preprocessing procedure demands a large amount of RAM (~300G).
