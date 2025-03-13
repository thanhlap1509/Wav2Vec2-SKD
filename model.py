from typing import Dict, List, Optional, Union
import warnings

warnings.filterwarnings("ignore")
from transformers import Wav2Vec2ForCTC
import torch.nn.functional as F
import torch
import config

distill_layer = config.distill_layer

# custom wav2vec2forctc class that implement ctc + ictc loss

class MyWav2Vec2ForCTC(Wav2Vec2ForCTC):
    def __init__(self, config):
        super().__init__(config, )
        self.student_layer = distill_layer

    def freeze(self):
        for name, param in self.named_parameters():
            if "lm_head" not in name:  # Keep lm_head trainable
                param.requires_grad = False

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True  # unfreeze all layer
        self.freeze_feature_extractor()  # but freeze feature_extractor

    def forward(
            self,
            input_values: Optional[torch.Tensor],
            attention_mask: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            labels: Optional[torch.Tensor] = None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if labels is not None and labels.max() >= self.config.vocab_size:
            raise ValueError(f"Label values must be <= vocab_size: {self.config.vocab_size}")

        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
        )

        teacher_hidden_states = outputs.hidden_states[-1]
        teacher_hidden_states = self.dropout(teacher_hidden_states)
        teacher_logits = self.lm_head(teacher_hidden_states)

        student_hidden_states = outputs.hidden_states[self.student_layer - 1]  # since layer encoder index start at 0
        student_hidden_states = self.dropout(student_hidden_states)
        student_logits = self.lm_head(student_hidden_states)

        loss = None
        if labels is not None:
            # retrieve loss input_lengths from attention_mask
            attention_mask = (
                attention_mask if attention_mask is not None else torch.ones_like(input_values, dtype=torch.float32)
            )
            input_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1)).to(torch.long)

            # assuming that padded tokens are filled with -100
            # when not being attended to
            labels_mask = labels >= 0
            target_lengths = labels_mask.sum(-1)
            flattened_targets = labels.masked_select(labels_mask)

            # ctc_loss doesn't support fp16
            teacher_log_probs = F.log_softmax(teacher_logits, dim=-1, dtype=torch.float32).transpose(0, 1)
            student_log_probs = F.log_softmax(student_logits, dim=-1, dtype=torch.float32).transpose(0, 1)

            with torch.backends.cudnn.flags(enabled=False):
                ctc_loss = F.ctc_loss(
                    teacher_log_probs,
                    flattened_targets,
                    input_lengths,
                    target_lengths,
                    blank=self.config.pad_token_id,
                    reduction=self.config.ctc_loss_reduction,
                    zero_infinity=self.config.ctc_zero_infinity,
                )
                ictc_loss = F.ctc_loss(
                    student_log_probs,
                    flattened_targets,
                    input_lengths,
                    target_lengths,
                    blank=self.config.pad_token_id,
                    reduction=self.config.ctc_loss_reduction,
                    zero_infinity=self.config.ctc_zero_infinity,
                )
        return ctc_loss, ictc_loss, teacher_logits, student_logits
