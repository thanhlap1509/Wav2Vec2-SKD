from typing import Dict, List, Optional, Union
import warnings

warnings.filterwarnings("ignore")

import numpy
from data_collator import DataCollatorCTCWithPadding
from datasets import load_dataset
import os
import evaluate
from transformers import Wav2Vec2Processor
import torch
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# from model_self_condition import Wav2Vec2ForCTC
import config
from transformers import set_seed
from transformers import Wav2Vec2ForCTC
from model import MyWav2Vec2ForCTC

# create folder metrics/weights inside save_dir
os.makedirs(os.path.join(config.save_dir, 'weights'), exist_ok=True)
os.makedirs(os.path.join(config.save_dir, 'metrics'), exist_ok=True)

# from model import Wav2Vec2ForCTCLabelSmoothing
set_seed(101)
# from model import Wav2Vec2ForCTCLabelSmoothing, Wav2Vec2ForOTC
train_dataset = load_dataset("csv", data_files=config.train_data, split="train", cache_dir='./cache')
eval_dataset = load_dataset("csv", data_files=config.eval_data, split="train", cache_dir='./cache')

processor = Wav2Vec2Processor.from_pretrained(config.pretrained_model)


def speech_file_to_array_fn(batch):
    speech_array, sampling_rate = torchaudio.load(batch["path"])
    speech_array = torchaudio.functional.resample(speech_array, orig_freq=sampling_rate, new_freq=16000)
    sampling_rate = 16000
    batch["speech"] = speech_array[0].numpy()
    batch["sampling_rate"] = sampling_rate
    batch["input_length"] = len(batch["speech"])
    batch["target_text"] = batch["transcription"] if batch["transcription"] is not None else '[UNK]'
    return batch


def prepare_dataset(batch):
    # check that all files have the correct sampling rate
    assert (
            len(set(batch["sampling_rate"])) == 1
    ), f"Make sure all inputs have the same sampling rate of {processor.feature_extractor.sampling_rate}."

    batch["input_values"] = processor(batch["speech"], sampling_rate=batch["sampling_rate"][0]).input_values
    with processor.as_target_processor():
        batch["labels"] = processor(batch["target_text"]).input_ids
    return batch


train_dataset = train_dataset.map(speech_file_to_array_fn, remove_columns=train_dataset.column_names)
train_dataset = train_dataset.map(prepare_dataset, remove_columns=['speech', 'sampling_rate', 'target_text'],
                                  batch_size=16, num_proc=16, batched=True)

eval_dataset = eval_dataset.map(speech_file_to_array_fn, remove_columns=eval_dataset.column_names)
eval_dataset = eval_dataset.map(prepare_dataset, remove_columns=eval_dataset.column_names, batch_size=16, num_proc=16,
                                batched=True)

data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

wer_metric = evaluate.load("wer")

model = MyWav2Vec2ForCTC.from_pretrained(
    config.pretrained_model,
    attention_dropout=0.1,
    hidden_dropout=0.1,
    feat_proj_dropout=0.1,
    final_dropout=0.1,
    # mask_time_prob=0.1,
    # mask_feature_prob=0.1,
    mask_time_prob=config.mask_time_prob,
    mask_feature_prob=config.mask_feature_prob,
    apply_spec_augment=config.apply_spec_augment,
    layerdrop=0.1,
    ctc_loss_reduction=config.ctc_loss_reduction,
    ctc_zero_infinity=True,
    pad_token_id=processor.tokenizer.pad_token_id,
    vocab_size=len(processor.tokenizer) - 2
)
model.to(device)


def scheduling_func(e, E=200, t=0.3):
    return min(max((e - 1) / (E - 1), t), 1 - t)


# setting training parameters
num_epoch = config.num_epochs
train_batch_size = config.per_device_train_batch_size
optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
warmup_step = config.warmup_ratio * num_epoch
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epoch - warmup_step)
min_wer = 100

# data loader
generator = torch.Generator()
generator.manual_seed(101)
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=train_batch_size,
    collate_fn=data_collator,
    shuffle=True,
    generator=generator
)
eval_loader = torch.utils.data.DataLoader(
    eval_dataset,
    batch_size=train_batch_size,
    collate_fn=data_collator,
    shuffle=True,
    generator=generator
)
with open(config.save_dir + '/metrics/eer_score.txt', 'w') as f:
    f.close()  # clear metric file

for epoch in range(num_epoch):
    # freeze model first 12.5% of the steps except linear
    if epoch < num_epoch // 8:
        model.freeze()
    else:
        model.unfreeze()
    model.train().to(device)
    running_loss = []
    for step, batch in tqdm(enumerate(train_loader), total=len(train_loader), desc=f'EPOCH {epoch}'):
        inputs = batch['input_values'].to(device)
        labels = batch['labels'].to(device)

        ctc_loss, ictc_loss, teacher_logits, student_logits = model(input_values=inputs, labels=labels)

        # skd
        teacher_logits_detached = teacher_logits.clone().detach()
        teacher_probs_detached = F.softmax(teacher_logits_detached, dim=-1, dtype=torch.float32).transpose(0, 1)
        student_log_probs = F.log_softmax(student_logits, dim=-1, dtype=torch.float32).transpose(0, 1)
        skd_loss = F.kl_div(student_log_probs, teacher_probs_detached)

        # total loss
        alpha = scheduling_func(e=epoch + 1, E=num_epoch, t=0.3)
        loss = (1 - alpha) * ctc_loss + alpha * (ictc_loss + skd_loss)

        running_loss.append(ictc_loss.item())
        loss.backward()
        optimizer.step()

        # linear warmup lr
        if epoch < warmup_step:
            lr = 3e-5 * (epoch + 1) / warmup_step
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        else:
            scheduler.step()

        optimizer.zero_grad()

    print(f"Training loss: {sum(running_loss) / len(running_loss)}")
    if sum(running_loss) / len(running_loss) <= 1:  # ensure decode fast
        with torch.no_grad():
            model.eval().to(device)
            losses = []
            predictions = []
            references = []

            for batch in tqdm(eval_loader, desc="Evaluating"):
                inputs = batch['input_values'].to(device)
                labels = batch['labels'].to(device)
                _, ictc_loss, _, student_logits = model(input_values=inputs, labels=labels)
                losses.append(ictc_loss)  # get loss at student level

                # Lấy dự đoán
                pred_ids = torch.argmax(student_logits, dim=-1)
                pred_str = processor.batch_decode(pred_ids)

                # Lấy nhãn thực
                label_ids = labels.cpu().numpy()
                label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
                label_str = processor.batch_decode(label_ids, group_tokens=False)

                predictions.extend(pred_str)
                references.extend(label_str)

            avg_loss = torch.mean(torch.tensor(losses, dtype=torch.float32))
            epoch_wer = wer_metric.compute(predictions=predictions, references=references)

            if (epoch_wer < min_wer):
                print("save_checkpoint...")
                min_wer = epoch_wer
                torch.save(model.state_dict(), config.save_dir + '/weights/epoch_' + str(epoch) + '.pth')
                torch.save(model.state_dict(), config.save_dir + '/weights/best.pth')

            with open(config.save_dir + '/metrics/eer_score.txt', 'a') as f:
                f.write("epoch " + str(epoch) + ": " + str(epoch_wer) + "\n")
            print("Evaluation loss: " + str(avg_loss.detach().numpy()))
            print("wer checkpoint " + str(epoch) + ": " + str(epoch_wer))
            print("min_wer: " + str(min_wer))





