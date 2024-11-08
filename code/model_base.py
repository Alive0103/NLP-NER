# -*- coding: utf-8 -*-
# Copyright (c) 2022 Heiheiyoyo. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
from dataclasses import dataclass
from transformers import BertModel, BertPreTrainedModel, PretrainedConfig
from transformers.utils import ModelOutput
from typing import Optional, Tuple
from torch.nn import functional as F
from transformers import BertTokenizerFast
@dataclass
class UIEModelOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    start_prob: torch.FloatTensor = None
    end_prob: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

class UIE(BertPreTrainedModel):

    def __init__(self, config: PretrainedConfig):
        super(UIE, self).__init__(config)
        self.encoder = BertModel(config)
        self.config = config
        hidden_size = self.config.hidden_size

        self.linear_start = nn.Linear(hidden_size, 1)
        self.linear_end = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.1)
        # self.loss_weight = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        # self.loss_weight.data.fill_(-0.2)

        self.total_weight = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.total_weight.data.fill_(-0.2)


        if hasattr(config, 'use_task_id') and config.use_task_id:
            # Add task type embedding to BERT
            task_type_embeddings = nn.Embedding(
                config.task_type_vocab_size, config.hidden_size)
            self.encoder.embeddings.task_type_embeddings = task_type_embeddings

            def hook(module, input, output):
                input = input[0]
                return output+task_type_embeddings(torch.zeros(input.size(), dtype=torch.int64, device=input.device))
            self.encoder.embeddings.word_embeddings.register_forward_hook(hook)

        self.post_init()




    def forward(self, input_ids: Optional[torch.Tensor] = None,
                token_type_ids: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                head_mask: Optional[torch.Tensor] = None,
                inputs_embeds: Optional[torch.Tensor] = None,
                start_positions: Optional[torch.Tensor] = None,
                end_positions: Optional[torch.Tensor] = None,
                fake_: Optional[torch.Tensor]=None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
                comput_kl: Optional[bool] = None
                ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.encoder(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        sequence_output = outputs[0]
        #sequence_output = self.dropout(sequence_output)
        start_logits = self.linear_start(sequence_output)
        start_logits = torch.squeeze(start_logits, -1)
        start_prob = self.sigmoid(start_logits)
        end_logits = self.linear_end(sequence_output)
        end_logits = torch.squeeze(end_logits, -1)
        end_prob = self.sigmoid(end_logits)
        if start_positions is not None:
            start_positions = start_positions.float()
        if end_positions is not None:
            end_positions = end_positions.float()
        if comput_kl:
            loss_rate = torch.sigmoid(self.total_weight)
        #print(fake_)
        total_loss = None
        if start_positions is not None and end_positions is not None:
            if fake_ is None:
                loss_fct = nn.BCELoss()
                start_loss = loss_fct(start_prob, start_positions)
                end_loss = loss_fct(end_prob, end_positions)
                total_loss = (start_loss + end_loss) / 2.0
            else:
                rate = torch.sigmoid(self.loss_weight)
                total_nums = input_ids.shape[0]
                fake_num = fake_.sum().item()

                loss_fct = nn.BCELoss(reduction="none")
                start_loss = loss_fct(start_prob, start_positions).mean(dim=-1)
                end_loss = loss_fct(end_prob, end_positions).mean(dim=-1)
                if fake_num ==0:
                    total_loss = (start_loss.sum() + end_loss.sum()) / 2.0
                elif fake_num == total_nums:
                    start_loss = (rate*fake_*start_loss).sum()/fake_num
                    end_loss = (rate*fake_*end_loss).sum()/fake_num
                    total_loss = (start_loss + end_loss) / 2.0
                else:
                    #end_loss = loss_fct(end_prob, end_positions)
                    #print(start_loss.shape)
                    #print(fake_.shape)
                    #print((rate*fake_*start_loss).shape)
                    start_loss = (rate*fake_*start_loss).sum()/fake_num+((1-rate)*(1-fake_)*start_loss).sum()/(total_nums-fake_num)
                    end_loss = (rate * fake_ * end_loss).sum() / fake_num + ((1 - rate) * (
                                1 - fake_) * end_loss).sum() / (total_nums - fake_num)
                    total_loss = (start_loss + end_loss) / 2.0
        if comput_kl:
            #loss = (1 - lambda_) * loss_SL + lambda_ * T * T * loss_KD
            kl_loss = use_kl(input_ids,token_type_ids,attention_mask, start_prob, end_prob)
            total_loss = loss_rate*kl_loss +(1-loss_rate)*total_loss

        if not return_dict:
            output = (start_prob, end_prob) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return UIEModelOutput(
            loss=total_loss,
            start_prob=start_prob,
            end_prob=end_prob,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
#model = UIE.from_pretrained("./saved_model/model1/model_best")
#model.to("cuda")
def use_kl(input_ids,token_type_ids,attention_mask, start_prob, end_prob):
    model=None
    output = model(input_ids=input_ids,
                   token_type_ids=token_type_ids,
                   attention_mask=attention_mask)
    start_, end_ = output[0], output[1]
    #kl_loss = torch.nn.MSELoss(reduction="mean")
    kl_loss = nn.KLDivLoss(reduction="mean")
    start_loss = kl_loss(start_prob, start_)
    end_loss = kl_loss(end_prob, end_)
    return (start_loss+end_loss)/2.0

class focal_loss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, num_classes=3, size_average=True):
        super(focal_loss, self).__init__()
        self.size_average = size_average
        if isinstance(alpha, list):
            assert len(alpha) == num_classes
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha < 1
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1 - alpha)

        self.gamma = gamma

    def forward(self, preds, labels):
        # assert preds.dim()==2 and labels.dim()==1
        preds = preds.view(-1, preds.size(-1))
        self.alpha = self.alpha.to(preds.device)
        preds_softmax = F.softmax(preds, dim=1)
        preds_logsoft = torch.log(preds_softmax)

        # focal_loss func, Loss = -α(1-yi)**γ *ce_loss(xi,yi)
        preds_softmax = preds_softmax.gather(1, labels.view(-1, 1))
        preds_logsoft = preds_logsoft.gather(1, labels.view(-1, 1))
        self.alpha = self.alpha.gather(0, labels.view(-1))
        # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ
        loss = -torch.mul(torch.pow((1 - preds_softmax), self.gamma), preds_logsoft)

        loss = torch.mul(self.alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


