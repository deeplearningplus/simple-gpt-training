# simple-gpt-training
A simple script to train GPT model


## 1. Extract text reports
```bash
cd data
python data.py # Not Run
```

## 2. Train tokenizer
```bash
cd tokenizer
python spm_train.py # Not Run
```

## 3. Train OPT model
```bash
bash train.sh
```

