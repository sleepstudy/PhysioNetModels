1. The purpose we use Phyionet is for comparsion to existing know research, hence we apply the same data processing to produce for training data using: https://github.com/akaraspt/deepsleepnet , Prepare dataset session to prepare the dataset, but with different set of data
2. We download the data from https://physionet.org/physiobank/database/sleep-edfx/sleep-cassette/, only *PSG.edf and *Hypnogram.edf will be processed and use step 1. for processing, please notice SC4002E0-PSG.edf, SC4002EC-Hypnogram.edf SC4362F0-PSG.edf SC4362FC-Hypnogram.edf are broken source data, they can not be processed, others should be fine. After processing, put at least one data *.npz file into data\testing, data\training, data\vaildation respectively
3. For testing our codes and models, we put one subject npz file into data\testing, data\training and data\vaildation respectively. 
4. To control what models you want to run, you just make change train_EGG.py in MODEL_TYPE = 'RNN'  # TODO: Change this to 'MLP', 'CNN', or 'RNN' according to your task  
5. best_models_with_20_subjects have the models we ran for the paper and presentation with 12 training subjects, 4 vaildation subjects and 4 testing subjects
6. run following to the codes:
6.1 conda env create -f environment.yml
6.2 conda activate bd4hproject
6.3 cd code  (go into code folder)
6.4 the output results models will be under output\physionet, for example: output\physionet\MyRNN.pth
6.5 There are 3 output plots under code folers: accuracies_curves.png and losses_curves.png are for training and vaildation dataset, confusion_matrix.png is for testing dataset
