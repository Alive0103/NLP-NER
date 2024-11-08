import argparse
import shutil
import sys
import time
import os
import torch
import torch.nn as nn
from dataclasses import dataclass
from transformers.utils import ModelOutput
from typing import Optional, Tuple
from torch.nn import functional as F
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast, BertModel, BertPreTrainedModel, PretrainedConfig
from torch.utils.data import Dataset
from util.utils import logger, tqdm, IEDatasetFake, set_seed, SpanEvaluator, EarlyStopping, logging_redirect_tqdm, IEDataset
from sklearn.model_selection import KFold
from util.evaluate import evaluate

# 定义UIEModelOutput数据类，用于保存模型的输出
@dataclass
class UIEModelOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    start_prob: torch.FloatTensor = None
    end_prob: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

# 定义UIE模型类，继承自BertPreTrainedModel
class UIE(BertPreTrainedModel):

    def __init__(self, config: PretrainedConfig):
        super(UIE, self).__init__(config)
        self.encoder = BertModel(config)  # 使用BERT作为编码器
        self.config = config
        hidden_size = self.config.hidden_size

        self.linear_start = nn.Linear(hidden_size, 1)  # 定义线性层，用于起始位置预测
        self.linear_end = nn.Linear(hidden_size, 1)    # 定义线性层，用于结束位置预测
        self.sigmoid = nn.Sigmoid()                   # 定义Sigmoid激活函数
        self.dropout = nn.Dropout(0.1)                # 定义Dropout层

        self.total_weight = nn.Parameter(torch.FloatTensor(1), requires_grad=True)  # 定义可训练参数
        self.total_weight.data.fill_(-0.2)  # 初始化参数

        # 如果配置中使用任务ID，则添加任务类型嵌入
        if hasattr(config, 'use_task_id') and config.use_task_id:
            task_type_embeddings = nn.Embedding(config.task_type_vocab_size, config.hidden_size)
            self.encoder.embeddings.task_type_embeddings = task_type_embeddings

            def hook(module, input, output):
                input = input[0]
                return output + task_type_embeddings(torch.zeros(input.size(), dtype=torch.int64, device=input.device))
            self.encoder.embeddings.word_embeddings.register_forward_hook(hook)

        self.post_init()

    # 定义前向传播函数
    def forward(self, model_path: Optional[str] = None, input_ids: Optional[torch.Tensor] = None,
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
                if fake_num == 0:
                    total_loss = (start_loss.sum() + end_loss.sum()) / 2.0
                elif fake_num == total_nums:
                    start_loss = (rate * fake_ * start_loss).sum() / fake_num
                    end_loss = (rate * fake_ * end_loss).sum() / fake_num
                    total_loss = (start_loss + end_loss) / 2.0
                else:
                    start_loss = (rate * fake_ * start_loss).sum() / fake_num + ((1 - rate) * (1 - fake_) * start_loss).sum() / (total_nums - fake_num)
                    end_loss = (rate * fake_ * end_loss).sum() / fake_num + ((1 - rate) * (1 - fake_) * end_loss).sum() / (total_nums - fake_num)
                    total_loss = (start_loss + end_loss) / 2.0
        if comput_kl:
            model = UIE.from_pretrained(model_path)
            model.to("cuda")
            kl_loss = use_kl(input_ids, token_type_ids, attention_mask, start_prob, end_prob, model)
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

# 定义KL损失计算函数
def use_kl(input_ids, token_type_ids, attention_mask, start_prob, end_prob, model):
    output = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
    start_, end_ = output[0], output[1]
    kl_loss = nn.KLDivLoss(reduction="mean")
    start_loss = kl_loss(start_prob, start_)
    end_loss = kl_loss(end_prob, end_)
    return (start_loss + end_loss) / 2.0

# 定义Focal Loss
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
        preds = preds.view(-1, preds.size(-1))
        self.alpha = self.alpha.to(preds.device)
        preds_softmax = F.softmax(preds, dim=1)
        preds_logsoft = torch.log(preds_softmax)

        preds_softmax = preds_softmax.gather(1, labels.view(-1, 1))
        preds_logsoft = preds_logsoft.gather(1, labels.view(-1, 1))
        self.alpha = self.alpha.gather(0, labels.view(-1))

        loss = -torch.mul(torch.pow((1 - preds_softmax), self.gamma), preds_logsoft)

        loss = torch.mul(self.alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss

# 定义EMA类
class EMA():
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

# 定义数据加载类
class Data_Loader(Dataset):
    def __init__(self, dataset):
        self.dataset = list(dataset)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        return self.dataset[item]

# 定义训练函数
def do_train_fake(train_data_loader, dev_data_loader, model, comput_kl, save_direction, kl_model_path):
    set_seed(args.seed)
    show_bar = True
    ema = EMA(model, 0.999)
    ema.register()
    fgm = FGM(model)
    tokenizer = BertTokenizerFast.from_pretrained(args.model)
    optimizer = torch.optim.AdamW(lr=args.learning_rate, params=model.parameters())

    criterion = torch.nn.functional.binary_cross_entropy
    metric = SpanEvaluator()

    if args.early_stopping:
        early_stopping_save_dir = os.path.join(save_direction, "early_stopping")
        if not os.path.exists(early_stopping_save_dir):
            os.makedirs(early_stopping_save_dir)
        if show_bar:
            def trace_func(*args, **kwargs):
                with logging_redirect_tqdm([logger.logger]):
                    logger.info(*args, **kwargs)
        else:
            trace_func = logger.info
        early_stopping = EarlyStopping(patience=7, verbose=True, trace_func=trace_func, save_dir=early_stopping_save_dir)

    loss_list = []
    loss_sum = 0
    loss_num = 0
    global_step = 0
    best_f1 = 0
    tic_train = time.time()
    epoch_iterator = range(1, args.num_epochs + 1)
    if show_bar:
        train_postfix_info = {'loss': 'unknown'}
        epoch_iterator = tqdm(epoch_iterator, desc='Training', unit='epoch')
    for epoch in epoch_iterator:
        train_data_iterator = train_data_loader
        if show_bar:
            train_data_iterator = tqdm(train_data_iterator, desc=f'Training Epoch {epoch}', unit='batch')
            train_data_iterator.set_postfix(train_postfix_info)
        for batch in train_data_iterator:
            if show_bar:
                epoch_iterator.refresh()
            input_ids, token_type_ids, att_mask, start_ids, end_ids = batch

            if args.device == 'gpu':
                input_ids = input_ids.cuda()
                token_type_ids = token_type_ids.cuda()
                att_mask = att_mask.cuda()
                start_ids = start_ids.cuda()
                end_ids = end_ids.cuda()
            outputs = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=att_mask, start_positions=start_ids, end_positions=end_ids, comput_kl=comput_kl, model_path=kl_model_path)

            loss, start_prob, end_prob = outputs[0], outputs[1], outputs[2]

            loss.backward()
            fgm.attack()
            outputs = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=att_mask, start_positions=start_ids, end_positions=end_ids, comput_kl=comput_kl, model_path=kl_model_path)

            loss_adv, start_prob, end_prob = outputs[0], outputs[1], outputs[2]
            loss_adv.backward()
            fgm.restore()

            ema.update()
            optimizer.step()
            optimizer.zero_grad()
            loss_list.append(float(loss))
            loss_sum += float(loss)
            loss_num += 1

            if show_bar:
                loss_avg = loss_sum / loss_num
                train_postfix_info.update({
                    'loss': f'{loss_avg:.5f}'
                })
                train_data_iterator.set_postfix(train_postfix_info)

            global_step += 1
            if global_step % args.logging_steps == 0:
                time_diff = time.time() - tic_train
                loss_avg = loss_sum / loss_num

                if show_bar:
                    with logging_redirect_tqdm([logger.logger]):
                        logger.info(
                            "global step %d, epoch: %d, loss: %.5f, speed: %.2f step/s"
                            % (global_step, epoch, loss_avg, args.logging_steps / time_diff))
                else:
                    logger.info(
                        "global step %d, epoch: %d, loss: %.5f, speed: %.2f step/s"
                        % (global_step, epoch, loss_avg, args.logging_steps / time_diff))
                tic_train = time.time()

            if global_step % args.valid_steps == 0:
                save_dir = os.path.join(save_direction, "model_%d" % global_step)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                model_to_save = model
                model_to_save.save_pretrained(save_dir)
                tokenizer.save_pretrained(save_dir)
                if args.max_model_num:
                    model_to_delete = global_step - args.max_model_num * args.valid_steps
                    model_to_delete_path = os.path.join(save_direction, "model_%d" % model_to_delete)
                    if model_to_delete > 0 and os.path.exists(model_to_delete_path):
                        shutil.rmtree(model_to_delete_path)
                ema.apply_shadow()
                dev_loss_avg, precision, recall, f1 = evaluate(model, metric, data_loader=dev_data_loader, device=args.device, loss_fn=criterion)
                ema.restore()
                if show_bar:
                    train_postfix_info.update({
                        'F1': f'{f1:.3f}',
                        'dev loss': f'{dev_loss_avg:.5f}'
                    })
                    train_data_iterator.set_postfix(train_postfix_info)
                    with logging_redirect_tqdm([logger.logger]):
                        logger.info("Evaluation precision: %.5f, recall: %.5f, F1: %.5f, dev loss: %.5f"
                                    % (precision, recall, f1, dev_loss_avg))
                else:
                    logger.info("Evaluation precision: %.5f, recall: %.5f, F1: %.5f, dev loss: %.5f"
                                % (precision, recall, f1, dev_loss_avg))
                if f1 > best_f1:
                    if show_bar:
                        with logging_redirect_tqdm([logger.logger]):
                            logger.info(
                                f"best F1 performence has been updated: {best_f1:.5f} --> {f1:.5f}"
                            )
                    else:
                        logger.info(
                            f"best F1 performence has been updated: {best_f1:.5f} --> {f1:.5f}"
                        )
                    best_f1 = f1
                    save_dir = os.path.join(save_direction, "model_best")
                    model_to_save = model
                    model_to_save.save_pretrained(save_dir)
                    tokenizer.save_pretrained(save_dir)
                tic_train = time.time()

        if args.early_stopping:
            dev_loss_avg, precision, recall, f1 = evaluate(model, metric, data_loader=dev_data_loader, device=args.device, loss_fn=criterion)

            if show_bar:
                train_postfix_info.update({
                    'F1': f'{f1:.3f}',
                    'dev loss': f'{dev_loss_avg:.5f}'
                })
                train_data_iterator.set_postfix(train_postfix_info)
                with logging_redirect_tqdm([logger.logger]):
                    logger.info("Evaluation precision: %.5f, recall: %.5f, F1: %.5f, dev loss: %.5f"
                                % (precision, recall, f1, dev_loss_avg))
            else:
                logger.info("Evaluation precision: %.5f, recall: %.5f, F1: %.5f, dev loss: %.5f"
                            % (precision, recall, f1, dev_loss_avg))
            early_stopping(dev_loss_avg, model)
            if early_stopping.early_stop:
                if show_bar:
                    with logging_redirect_tqdm([logger.logger]):
                        logger.info("Early stopping")
                else:
                    logger.info("Early stopping")
                tokenizer.save_pretrained(early_stopping_save_dir)
                sys.exit(0)

# 定义FGM对抗训练类
class FGM():
    """用于对抗训练，提升鲁棒性，
    使用方式
    fgm.attack()
    loss_adv = loss_func(x, lable)
    loss_adv.backward(retain_graph=True)
    fgm.restore()
    """

    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., emb_name='word_embedding.'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='word_embedding.'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

# 定义K折交叉验证函数
def KFoldCrossValidator(comput_kl, train_path, save_direction, kl_model_path, i):
    model = UIE.from_pretrained(args.model)
    if args.device == 'gpu':
        model = model.cuda()
    kf = KFold(5, shuffle=True, random_state=789)
    tokenizer = BertTokenizerFast.from_pretrained(args.model)
    stack_raw_examples = IEDatasetFake(train_path, tokenizer, 512)
    for i, (train_ids, dev_ids) in enumerate(kf.split(stack_raw_examples)):
        train_raw_examples = [stack_raw_examples[_idx] for _idx in train_ids]
        dev_raw_examples = [stack_raw_examples[_idx] for _idx in dev_ids]
        train_raws = Data_Loader(train_raw_examples)
        dev_raws = Data_Loader(dev_raw_examples)
        train_data_loader = DataLoader(train_raws, batch_size=args.batch_size, shuffle=True)
        dev_data_loader = DataLoader(dev_raws, batch_size=args.batch_size, shuffle=True)
        do_train_fake(train_data_loader, dev_data_loader, model, comput_kl, save_direction, kl_model_path)

# 定义评估函数
def do_eval(model_path, test_path, i):
    tokenizer = BertTokenizerFast.from_pretrained(model_path)
    model = UIE.from_pretrained(model_path)
    if args.device == 'gpu':
        model = model.cuda()

    test_ds = IEDataset(test_path, tokenizer=tokenizer, max_seq_len=args.max_seq_len)
    test_data_loader = DataLoader(test_ds, batch_size=40, shuffle=False)
    metric = SpanEvaluator()
    precision, recall, f1 = evaluate(model, metric, test_data_loader, args.device)
    logger.info(f"第{i}个任务Evaluation precision: %.5f, recall: %.5f, F1: %.5f" %
                (precision, recall, f1))
    print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-b", "--batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--learning_rate", default=1e-5,
                        type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--max_seq_len", default=512, type=int, help="The maximum input sequence length. "
                        "Sequences longer than this will be split automatically.")
    parser.add_argument("--num_epochs", default=2, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--seed", default=1000, type=int,
                        help="Random seed for initialization")
    parser.add_argument("--logging_steps", default=100,
                        type=int, help="The interval steps to logging.")
    parser.add_argument("--valid_steps", default=100, type=int,
                        help="The interval steps to evaluate model performance.")
    parser.add_argument("-D", '--device', choices=['cpu', 'gpu'], default="gpu",
                        help="Select which device to train model, defaults to gpu.")
    parser.add_argument("-m", "--model", default="uie_base_pytorch", type=str,
                        help="Select the pretrained model for few-shot learning.")
    parser.add_argument("--max_model_num", default=5, type=int,
                        help="Max number of saved model. Best model and earlystopping model is not included.")
    parser.add_argument("--early_stopping", action='store_true', default=True,
                        help="Use early stopping while training")

    args = parser.parse_args()

    k = 1
    do_eval("models/model1/model_best", f"data/test{k}.txt", k)
    numbers = [2, 3, 4, 5, 6]
    my_list = [k]
    for i in numbers:
        KFoldCrossValidator(True, f"data/trains{i}.txt", f"models/model{i}", f"models/model{my_list[-1]}/model_best", i)
        my_list.append(i)
        for j in my_list:
            do_eval(f"models/model{j}/model_best", f"data/test{j}.txt", i)
