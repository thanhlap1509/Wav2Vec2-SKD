import torch
import torchaudio
from datasets import load_dataset
from transformers import set_seed
from transformers import Wav2Vec2Processor
from model import MyWav2Vec2ForCTC
import evaluate
import config
from data_collator import DataCollatorCTCWithPadding
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
set_seed(101)
# Load the processor and model
processor = Wav2Vec2Processor.from_pretrained(config.pretrained_model)
model = MyWav2Vec2ForCTC.from_pretrained(config.pretrained_model)
model.load_state_dict(torch.load(config.eval_model, map_location=device))
model.to(device)
model.eval()

eval_dataset = load_dataset("csv", data_files=config.eval_data, split="train", cache_dir='./cache')


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


eval_dataset = eval_dataset.map(speech_file_to_array_fn, remove_columns=eval_dataset.column_names)
eval_dataset = eval_dataset.map(prepare_dataset, remove_columns=eval_dataset.column_names, batch_size=16, num_proc=16,
                                batched=True)

data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

# Initialize WER metric
wer_metric = evaluate.load("wer")

# Perform inference and calculate loss and WER
generator = torch.Generator()
generator.manual_seed(101)
eval_loader = torch.utils.data.DataLoader(
    eval_dataset,
    batch_size=config.per_device_train_batch_size,
    collate_fn=data_collator,
    shuffle=True,
    generator=generator
)
losses = []
predictions = []
references = []
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
    wer = wer_metric.compute(predictions=predictions, references=references)

    print(f"Average Loss: {avg_loss}")
    print(f"WER: {wer}")
