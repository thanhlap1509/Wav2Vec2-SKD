# Fine-tuning Wav2Vec2 model for Vietnamese ASR

## Data preparation :
Data should be prepared similar to the sample provided. The content of the csv data file should have 2 columns:
 1. path: full path to audio file
 2. transcription: transcript of the audio file (the transcript should be normalized, such as removing all punctuation, converting to lowercase, etc.)

## Configuration:
Configure the information in the config file before running:
1. train_data: path to csv file containing the aforementioned path and transcript of training audio file
2. eval_data:  similar to train_data, with the use being for evaluation instead of training
3. save_dir: the directory that hold the weights and metrics folders containing the saved weights and training performance respectively
4. eval_model: used for evaluation, insert the path to .pth model

## Training:
```
python3 train.py
```

## Evaluating: 
```
python3 infer.py
```
## Citations :
```
@article{SKD-CTC,
  title={Guiding Frame-Level CTC Alignments Using Self-knowledge Distillation},
  author={Eungbeom Kim, Hantae Kim, Kyogu Lee},
  journal={INTERSPEECH 2024},
  year={2024},
}
```
