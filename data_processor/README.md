## Usage
- `data_helper.py`: 
  - Main function: process raw data (id mapping, filtering, modify format, etc.) and split the dataset following specific rules.
  - Key output file: `train.pkl`, `test.pkl`
  - Can be extended to other different format session-based datasets.
  - How to run: `cd HG-GNN && python ./data_processor/data_helper.py`  to generate the data file of `lastfm`, or import the class `Data_Process` and execute the runner function in the project.
- `data_loader.py`:
  - Main function: generate the session data (sequence) and label for model training or inference. ( for an input session `s = [v_{s,1}, v_{s,2}, ... , v{s,n}]`, generate a series of sequences and labels `([v_{s,1}], v_{s,2})`,`([v_{s,1}, v_{s,2}], v_{s,3})`, ... ,`([v_{s,1}, v_{s,2}, ... , v_{s,n−1}], v_{s,n})`)
  - Intermediate output file: `train_seq.pkl`, `test_seq.pkl`
- Similar with works like `https://github.com/twchen/lessr/blob/master/utils/data/preprocess.py` , `https://github.com/CRIPAC-DIG/SR-GNN/blob/master/datasets/preprocess.py`, etc.

## dataset
- lastfm-1K: 
  - (raw data) http://ocelma.net/MusicRecommendationDataset/lastfm-1K.html
- xing: 
  - (raw data) https://github.com/RecoHut-Datasets/xing/tree/v1
- reddit:
  - (raw data) https://www.kaggle.com/datasets/colemaclean/subreddit-interactions
- processed data:
  - https://drive.google.com/file/d/1edcrT_ExguRKZW3-YxPgOCtl4rTDrk_1/view?usp=share_link