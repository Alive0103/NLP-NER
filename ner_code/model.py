import torch
import torch.nn as nn
from dataclasses import dataclass
from transformers import BertModel, BertPreTrainedModel, PretrainedConfig
from transformers.utils import ModelOutput
from typing import Optional, Tuple
from torch.nn import functional as F

@dataclass
class UIEModelOutput(ModelOutput):  # 定义模型输出
    '''
    loss：表示模型的损失值。默认为 None。
    start_prob：表示模型预测的起始位置概率。默认为 None。
    end_prob：表示模型预测的结束位置概率。默认为 None。
    hidden_states：表示模型的隐藏状态。默认为 None。
    attentions：表示模型的注意力权重。默认为 None。
    '''
    loss: Optional[torch.FloatTensor] = None
    start_prob: torch.FloatTensor = None
    end_prob: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class UIE(BertPreTrainedModel):  # 基于 BERT 预训练模型的自定义模型

    def __init__(self, config: PretrainedConfig):
        super(UIE, self).__init__(config)
        self.encoder = BertModel(config)  # config 参数，这个参数包含了 BERT 模型的配置信息，如隐藏层大小、层数等。
        self.config = config
        hidden_size = self.config.hidden_size

        self.linear_start = nn.Linear(hidden_size, 1)  # 表示起始位置的概率
        self.linear_end = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.1)
        # 创建了一个 Dropout 层 dropout，丢弃概率为 0.1。Dropout 层在训练过程中以一定概率丢弃输入的部分特征，有助于减少过拟合现象，提高模型的泛化能力
        # self.loss_weight = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        # self.loss_weight.data.fill_(-0.2)

        self.total_weight = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.total_weight.data.fill_(-0.2)

        if hasattr(config, 'use_task_id') and config.use_task_id:  # 判断是否需要为 BERT 模型添加任务类型嵌入
            # Add task type embedding to BERT
            task_type_embeddings = nn.Embedding(  # 任务类型嵌入层
                config.task_type_vocab_size, config.hidden_size)
            # 嵌入层的大小为 config.task_type_vocab_size（任务类型词汇表的大小）乘以 config.hidden_size（BERT 模型的隐藏层大小）
            # 每个任务类型都会被嵌入到 config.hidden_size 维度的空间中
            self.encoder.embeddings.task_type_embeddings = task_type_embeddings

            def hook(module, input, output):  # 将钩子函数注册到 BERT 模型的词嵌入层上
                input = input[0]
                return output + task_type_embeddings(torch.zeros(input.size(), dtype=torch.int64, device=input.device))

            self.encoder.embeddings.word_embeddings.register_forward_hook(hook)

        self.post_init()

    def forward(self, model_path:Optional[str]=None,
                input_ids: Optional[torch.Tensor] = None,
                token_type_ids: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                head_mask: Optional[torch.Tensor] = None,
                inputs_embeds: Optional[torch.Tensor] = None,
                start_positions: Optional[torch.Tensor] = None,
                end_positions: Optional[torch.Tensor] = None,
                fake_: Optional[torch.Tensor] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
                comput_kl: Optional[bool] = None
                ):
        '''
        input_ids：输入文本的token IDs。
        token_type_ids：用于区分文本片段的token类型 IDs。
        position_ids：位置 IDs，用于标识每个token的位置。
        attention_mask：注意力掩码，用于指示哪些token是padding的。
        head_mask：用于遮蔽模型中的注意力头部。
        inputs_embeds：可选的输入嵌入。
        start_positions：开始位置的标签，用于训练序列标注任务。
        end_positions：结束位置的标签，用于训练序列标注任务。
        fake_：一个可选的张量，用于指示哪些token是假的。
        output_attentions：是否返回注意力权重。
        output_hidden_states：是否返回隐藏状态。
        return_dict：是否返回输出结果的字典形式。
        comput_kl：是否计算KL散度。
        '''
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
        # sequence_output = self.dropout(sequence_output)
        start_logits = self.linear_start(sequence_output)
        start_logits = torch.squeeze(start_logits, -1)
        start_prob = self.sigmoid(start_logits)
        end_logits = self.linear_end(sequence_output)
        end_logits = torch.squeeze(end_logits, -1)
        end_prob = self.sigmoid(end_logits)
        # 得到起始位置和结束位置的logits，并通过 Sigmoid 函数得到概率
        if start_positions is not None:
            start_positions = start_positions.float()
        if end_positions is not None:
            end_positions = end_positions.float()
        if comput_kl:
            loss_rate = torch.sigmoid(self.total_weight)  # 计算交叉熵损失
        # print(fake_)
        total_loss = None
        if start_positions is not None and end_positions is not None:
            if fake_ is None:
                loss_fct = nn.BCELoss()  # 二元交叉熵损失函数
                start_loss = loss_fct(start_prob, start_positions)
                end_loss = loss_fct(end_prob, end_positions)
                total_loss = (start_loss + end_loss) / 2.0
            else:
                rate = torch.sigmoid(self.loss_weight)  # 损失的加权系数
                total_nums = input_ids.shape[0]  # 获取输入文本的批次大小
                fake_num = fake_.sum().item()
                # 使用了 PyTorch 的 sum() 方法统计了 fake_ 中非零元素的数量，并使用 item() 方法将张量的值转换为 Python 中的标量值

                loss_fct = nn.BCELoss(reduction="none")  # reduction="none" 表示不对每个样本的损失进行平均，而是返回每个样本的损失值
                start_loss = loss_fct(start_prob, start_positions).mean(dim=-1)
                end_loss = loss_fct(end_prob, end_positions).mean(dim=-1)  # 通过 mean(dim=-1) 对每个样本的损失进行了平均，得到一个标量值
                if fake_num == 0:
                    total_loss = (start_loss.sum() + end_loss.sum()) / 2.0
                elif fake_num == total_nums:
                    start_loss = (rate * fake_ * start_loss).sum() / fake_num
                    end_loss = (rate * fake_ * end_loss).sum() / fake_num
                    total_loss = (start_loss + end_loss) / 2.0
                else:
                    # end_loss = loss_fct(end_prob, end_positions)
                    # print(start_loss.shape)
                    # print(fake_.shape)
                    # print((rate*fake_*start_loss).shape)
                    start_loss = (rate * fake_ * start_loss).sum() / fake_num + (
                                (1 - rate) * (1 - fake_) * start_loss).sum() / (total_nums - fake_num)
                    end_loss = (rate * fake_ * end_loss).sum() / fake_num + ((1 - rate) * (
                            1 - fake_) * end_loss).sum() / (total_nums - fake_num)
                    total_loss = (start_loss + end_loss) / 2.0
                '''
                如果假样本的数量为 0，则直接对起始位置和结束位置的损失进行求和并取平均作为总体损失。
                如果所有样本都是假样本（假样本数量等于总样本数量），则对假样本的损失进行加权平均，并取得到的总体损失作为最终结果。
                如果存在真实样本和假样本，则分别对真实样本和假样本的损失进行加权平均，并将两者的加权平均作为总体损失。
                '''
        if comput_kl:
            # loss = (1 - lambda_) * loss_SL + lambda_ * T * T * loss_KD
            kl_loss = use_kl(input_ids, token_type_ids, attention_mask, start_prob, end_prob)
            total_loss = loss_rate * kl_loss + (1 - loss_rate) * total_loss

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


# model = UIE.from_pretrained("../data/user_data/saved_model/model1/model_best")
# model.to("cuda")


def use_kl(input_ids, token_type_ids, attention_mask, start_prob, end_prob):
    output = model(input_ids=input_ids,
                   token_type_ids=token_type_ids,
                   attention_mask=attention_mask)
    start_, end_ = output[0], output[1]
    # kl_loss = torch.nn.MSELoss(reduction="mean")
    kl_loss = nn.KLDivLoss(reduction="mean")  # reduction="mean" 表示计算平均损失
    start_loss = kl_loss(start_prob, start_)
    end_loss = kl_loss(end_prob, end_)
    return (start_loss + end_loss) / 2.0


class focal_loss(nn.Module):
    # Focal Loss 是一种用于解决类别不平衡问题的损失函数，通过降低易分类样本的权重，从而提高难分类样本的重要性
    def __init__(self, alpha=0.25, gamma=2, num_classes=3, size_average=True):
        '''
        alpha：平衡因子，用于调节易分类和难分类样本的权重。如果是一个列表，则表示每个类别的权重，如果是一个标量，则表示所有类别的权重相同，可根据实际情况进行调节，默认为0.25。
        gamma：调节因子，用于控制易分类样本的权重减小的速度，通常取较大的值以增强 Focal Loss 的效果，默认为2。
        num_classes：类别数量，默认为3。
        size_average：是否对损失进行平均，如果为 True，则返回的损失是一个标量值，表示平均损失；如果为 False，则返回的损失是一个总和，表示总损失。
        '''
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
        #  preds 的维度为 [样本数, 类别数]
        self.alpha = self.alpha.to(preds.device)
        preds_softmax = F.softmax(preds, dim=1)
        preds_logsoft = torch.log(preds_softmax)

        # focal_loss func, Loss = -α(1-yi)**γ *ce_loss(xi,yi)
        # 根据标签 labels 获取对应的预测概率和 log_softmax，并获取相应类别的平衡因子 alpha
        preds_softmax = preds_softmax.gather(1, labels.view(-1, 1))
        preds_logsoft = preds_logsoft.gather(1, labels.view(-1, 1))
        self.alpha = self.alpha.gather(0, labels.view(-1))
        # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ
        loss = -torch.mul(torch.pow((1 - preds_softmax), self.gamma), preds_logsoft)
        # 根据 Focal Loss 的公式 -alpha*(1-p)^gamma*log(p) 计算出原始的 Focal Loss

        loss = torch.mul(self.alpha, loss.t())
        # 将原始的 Focal Loss 乘以 self.alpha，实现了对不同类别的损失进行加权处理
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss
