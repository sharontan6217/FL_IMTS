This is for "Federated Gated Bi-directional Recurrent Neural Network on Irregular Multivariate Time Series Forecasting".

Data:
1. PhysioNet EEG: 	Abo Alzahab, N., Di Iorio, A., Apollonio, L., Alshalak, M., Gravina, A., Antognoli, L., Baldi, M., Scalise, L., & Alchalabi, B. (2021). Auditory evoked potential EEG-Biometric dataset (version 1.0.0). PhysioNet. RRID:SCR_007345. https://doi.org/10.13026/ps31-fc50.
2. PhysioNet ECG: Nemcova, A., Smisek, R., Opravilová, K., Vitek, M., Smital, L., & Maršánová, L. (2020). Brno University of Technology ECG Quality Database (BUT QDB) (version 1.0.0). PhysioNet. RRID:SCR_007345. https://doi.org/10.13026/kah4-0w24.
3. PhysioNet MIMIC-ICU:  	Johnson, A., Pollard, T., & Mark, R. (2019). MIMIC-III Clinical Database Demo (version 1.4). PhysioNet. RRID:SCR_007345. https://doi.org/10.13026/C2HM2Q.
4. UCI HAR Dataset: Bulbul, Erhan, Aydin Cetin, and Ibrahim Alper Dogru. "Human activity recognition using smartphones." 2018 2nd international symposium on multidisciplinary studies and innovative technologies (ismsit). IEEE, 2018.
5. Climate 	Easterling, D. R. United States Historical Climatology Network daily temperature and precipitation data (1871-1997). No. ORNL/CDIAC-118. Oak Ridge National Lab.(ORNL), Oak Ridge, TN (United States), 2002.
6. Air quality:  Godahewa, R., Bergmeir, C., Webb, G., Hyndman, R., & Montero-Manso, P. (2020). KDD Cup Dataset (with Missing Values) (Version 4) [Dataset]. Zenodo. https://doi.org/10.5281/zenodo.4656719.


Installation:
Please run 'pip -r install requirements.txt'

Experiment:
The complete command is:
Example:
```shell
python run.py 
    --data_dir {data_dir} \
    --graph_dir {graph_dir} \
    --log_dir {log_dir} \
    --output {output_dir} \
```

- `data_dir`: type=str,default='./data/', 'directory of the original data.'  
- `graph_dir`,type=str,default='./graph/', 'directory of graphs.'
- `output_dir`,type=str,default='./output/', 'directory of outputs.'
- `log_dir`,type=str,default='./log/', 'directory of the transaction logs.'


The configuration file is in `./model/Config.py`, you can change the parameters there.

To get the forecasting result,
please run:
```shell
'python run.py '
```
or
```shell
'python run.py --data_dir ./data/eeg/eeg.csv'
```
