# HG-GNN
## Usage

### Data Processing
- The data preprocessing code is following: `https://github.com/twchen/lessr/blob/master/utils/data/preprocess.py` and `https://github.com/CRIPAC-DIG/SR-GNN/blob/master/datasets/preprocess.py`
- `data_processor/data_loader.py` is used for generating training/val/test data.

### Training & Testing
- Run `python main.py`
- You can modify the `basic.ini` configuration file to set the dataset details and modify the `config/model.ini` file to set the hyper-parameters of target model.

### Requirement
- `pip isntall -r requirements.txt `
