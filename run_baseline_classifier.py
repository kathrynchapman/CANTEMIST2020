import pandas as pd
import sys

sys.path.insert(1, '../CLEF_Datasets_ICD/processed_data/')
from process_data import *
import torch
import io
import numpy as np
import os
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import MultiLabelBinarizer
import pickle
import logging
import random
import json
import argparse
from loss import BalancedBCEWithLogitsLoss, RankingLoss

from torch.utils.data import Dataset, RandomSampler, DataLoader, SequentialSampler
from utils import *
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss

from transformers.modeling_bert import BertConfig, BertModel, BertPreTrainedModel
from transformers.modeling_xlm_roberta import XLMRobertaModel

from tqdm import tqdm, trange

from transformers import (WEIGHTS_NAME, BertConfig,
                          BertForSequenceClassification, BertTokenizer,
                          XLMRobertaConfig, XLMRobertaForSequenceClassification,
                          XLMRobertaTokenizer, AdamW, get_linear_schedule_with_warmup)

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)

ALL_MODELS = ['Bert', 'XLMRoberta']


class ICDDataloader(Dataset):
    def __init__(self, data_path):
        self.data = pickle_load(data_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
        # return self.data.iloc[idx,]


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds, average='micro')
    prec = precision_score(y_true=labels, y_pred=preds, average="micro")
    recall = recall_score(y_true=labels, y_pred=preds, average="micro")
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
        "precision": prec,
        "recall": recall,
    }


#
# class BertForMultiLabelSequenceClassification(BertPreTrainedModel):
#     """BERT model for classification.
#     This module is composed of the BERT model with a linear layer on top of
#     the pooled output.
#     Params:
#         `config`: a BertConfig class instance with the configuration to build a new model.
#         `num_labels`: the number of classes for the classifier. Default = 2.
#     Inputs:
#         `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
#             with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
#             `extract_features.py`, `run_classifier.py` and `run_squad.py`)
#         `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
#             types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
#             a `sentence B` token (see BERT paper for more details).
#         `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
#             selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
#             input sequence length in the current batch. It's the mask that we typically use for attention when
#             a batch has varying length sentences.
#         `labels`: labels for the classification output: torch.LongTensor of shape [batch_size, num_labels]
#             with indices selected in [0, ..., num_labels].
#     Outputs:
#         if `labels` is not `None`:
#             Outputs the CrossEntropy classification loss of the output with the labels.
#         if `labels` is `None`:
#             Outputs the classification logits of shape [batch_size, num_labels].
#     """
#
#     def __init__(self, config, num_labels=2, loss_fct=''):
#         super(BertForMultiLabelSequenceClassification, self).__init__(config)
#         self.num_labels = num_labels
#         self.loss_fct = loss_fct
#         self.model = BertModel(config)
#         self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
#         self.classifier = torch.nn.Linear(config.hidden_size, num_labels)
#         # self.apply(self.init_weights)
#         self.init_weights()
#
#     def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, doc_ids=None):
#
#
#
#         outputs = self.model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
#         sequence_output, pooled_output = outputs[0], outputs[1]
#         pooled_output = self.dropout(pooled_output)
#         logits = self.classifier(pooled_output)
#
#         outputs = (logits,) + outputs[2:]
#         if labels is not None:
#             loss_fct = BCEWithLogitsLoss()
#             labels = labels.float()
#             loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
#             outputs = (loss,) + outputs
#         return outputs

class BertForMultiLabelSequenceClassification(BertPreTrainedModel):
    def __init__(self, config, args='', loss_fct='', class_weights=None):
        super().__init__(config)
        self.args = args
        self.num_labels = args.num_labels
        self.bert = BertModel(config)
        if not class_weights:
            self.class_weights = torch.ones((self.num_labels,))
        else:
            self.class_weights = class_weights
        self.iteration = 1
        self.loss_fct = loss_fct
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.classifier = torch.nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    # @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING.format("(batch_size, sequence_length)"))
    def forward(
            self,
            doc_input_ids=None,
            doc_attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            ranks=None,
            output_attentions=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`label` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    Examples::
        from transformers import BertTokenizer, BertForSequenceClassification
        import torch
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, logits = outputs[:2]
        """

        outputs = self.bert(
            doc_input_ids,
            attention_mask=doc_attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            # output_attentions=output_attentions,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        logits = logits.view(-1, self.num_labels)

        if self.args.doc_batching:
            if self.args.logit_aggregation == 'max':
                logits = torch.max(logits, axis=0)[0]
            elif self.args.logit_aggregation == 'avg':
                logits = torch.mean(logits, axis=0)


        # temp = logits.view(-1, self.num_labels) - labels.view(-1, self.num_labels) + 1
        temp = torch.mean(torch.abs(logits.view(-1, self.num_labels)-labels.view(-1, self.num_labels)).float()+1, axis=0)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        # #############
        # # get the sequence-level document representations
        # doc_seq_output = doc_outputs[0]
        # # print("DOC SEQ OUTPUT SHAPE:", doc_seq_output.shape)
        # batch_size = doc_seq_output.shape[0]
        # # print("BATCH SIZE:", batch_size)
        #
        # # get the sequence-level label description representations
        # label_seq_output = label_outputs[0]
        #
        # label_seq_output = label_seq_output.reshape(self.num_labels * self.args.label_max_seq_length, self.hidden_size)
        # temp = torch.matmul(doc_seq_output, label_seq_output.T)
        # temp = temp.permute(0, 2, 1)
        #
        # temp = self.w1(temp)
        #
        # temp = temp.reshape(batch_size, self.num_labels, self.args.label_max_seq_length)
        #
        # temp = self.w2(temp)
        #
        # logits = temp.view(-1, self.num_labels)
        #
        # ############
        if labels is not None:
            if self.args.do_iterative_class_weights:
                # temp = logits.view(-1, self.num_labels) - labels.view(-1, self.num_labels) + 1
                temp = logits.detach()
                temp = torch.nn.Sigmoid()(temp)
                temp = (temp > self.args.prediction_threshold).float()
                temp = torch.mean(
                    torch.abs(temp.view(-1, self.num_labels) - labels.view(-1, self.num_labels)).float() + 1, axis=0)
                try:
                    self.class_weights = torch.Tensor(self.class_weights).cuda()
                except:
                    pass
                self.class_weights *= self.iteration
                self.class_weights += temp
                self.class_weights /= (self.iteration + 1)
                class_weights = self.class_weights.detach()
            elif self.args.do_normal_class_weights:
                class_weights = torch.Tensor(self.class_weights).cuda()
            else:
                class_weights = None
            labels = labels.float()

            if self.loss_fct == 'bce':
                loss_fct = BCEWithLogitsLoss(pos_weight=class_weights)
            elif self.loss_fct == 'bbce':
                loss_fct = BalancedBCEWithLogitsLoss(grad_clip=True, weights=class_weights)
            elif self.loss_fct == 'cel':
                loss_fct = CrossEntropyLoss()

            if self.loss_fct != 'none':
                loss = loss_fct(logits.view(-1, self.num_labels), ranks.float().view(-1, self.num_labels))
            else:
                loss = 0

            if self.args.do_ranking_loss or self.args.do_weighted_ranking_loss:
                if not self.args.do_weighted_ranking_loss:
                    loss_fct = RankingLoss(self.args.doc_batching, weights=None)
                else:
                    loss_fct = RankingLoss(self.args.doc_batching, weights=class_weights)
                loss += loss_fct(logits, ranks)
            outputs = (loss,) + outputs



            # loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            # outputs = (loss,) + outputs
        self.iteration += 1
        return outputs  # (loss), logits, (hidden_states), (attentions)


MODEL_CLASSES = {
    "xlmroberta": (XLMRobertaConfig, XLMRobertaForSequenceClassification, XLMRobertaTokenizer),
    "bert": (BertConfig, BertForMultiLabelSequenceClassification, BertTokenizer),
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset, model, tokenizer, class_weights):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    if args.doc_batching:
        train_dataloader = DataLoader(train_dataset, sampler=None, batch_size=args.n_gpu, collate_fn=my_collate)
        train_dataloader = list(train_dataloader)
    else:
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    num_warmup_steps = int(len(train_dataloader) * args.warmup_proportion) * args.num_train_epochs
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon, correct_bias=False)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
            os.path.join(args.model_name_or_path, "scheduler.pt")
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        if args.doc_batching:
            model = MyDataParallel(model)
        else:
            model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True,
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num labels = %d", args.num_labels)
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path):
        # set global_step to gobal_step of last saved checkpoint from model path
        global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
        epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
        steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

        logger.info("  Continuing training from checkpoint, will skip to saved global_step")
        logger.info("  Continuing training from epoch %d", epochs_trained)
        logger.info("  Continuing training from global step %d", global_step)
        logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0],
    )
    set_seed(args)  # Added here for reproductibility
    for _ in train_iterator:
        if args.doc_batching:
            random.shuffle(train_dataloader)
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            if args.doc_batching:
                batch = tuple(tuple(ti.to(args.device) for ti in t) for t in batch)
            else:
                batch = tuple(t.to(args.device) for t in batch)

            inputs = {"doc_input_ids": batch[0], "doc_attention_mask": batch[1], "labels": batch[3], "ranks": batch[-1]}
            # inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logs = {}
                    if (
                            args.local_rank == -1 and args.evaluate_during_training
                    ):  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer)
                        for key, value in results.items():
                            eval_key = "eval_{}".format(key)
                            logs[eval_key] = value

                    loss_scalar = (tr_loss - logging_loss) / args.logging_steps
                    learning_rate_scalar = scheduler.get_lr()[0]
                    logs["learning_rate"] = learning_rate_scalar
                    logs["loss"] = loss_scalar
                    logging_loss = tr_loss

                    for key, value in logs.items():
                        tb_writer.add_scalar(key, value, global_step)
                    print(json.dumps({**logs, **{"step": global_step}}))

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, prefix=""):
    eval_output_dir = args.output_dir

    results = {}

    eval_dataset, idx2id = load_and_cache_examples(args, tokenizer, evaluate=True)

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    if args.doc_batching:
        eval_dataloader = DataLoader(eval_dataset, sampler=None, batch_size=1, collate_fn=my_collate)
    else:
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    ids = []
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()

        if args.doc_batching:
            batch = tuple(tuple(ti.to(args.device) for ti in t) for t in batch)
        else:
            batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():


            #############################
            if args.doc_batching:
                input_ids = batch[0][0]
                attn_mask = batch[1][0]
                labels = batch[3][0]
                ranks = batch[-1][0]
            else:
                input_ids = batch[0]  # may need to fix this!
                attn_mask = batch[1]  # may need to fix this!
                labels = batch[3]
                ranks = batch[-1]
            inputs = {"doc_input_ids": input_ids, "doc_attention_mask": attn_mask, "labels": labels, "ranks": ranks}

            #############################


            # if args.doc_batching:
            #     # print(batch[3])
            #     # labs = batch[3][0][0,:]
            #     labs = batch[3][0]
            #     # rnks = batch[-1][0][0, :]
            #     rnks = batch[-1][0]
            #
            #     # inputs = {"doc_input_ids": batch[0][0], "doc_attention_mask": batch[1][0], "labels": batch[3][0], "ranks": batch[-1][0]}
            #     inputs = {"doc_input_ids": batch[0][0], "doc_attention_mask": batch[1][0], "labels": labs, "ranks": rnks}
            # else:
            #     labs = batch[3][0]
            #     rnks = batch[-1][0]
            #     inputs = {"doc_input_ids": batch[0], "doc_attention_mask": batch[1], "labels": labs, "ranks": rnks}
            # inputs = {'input_ids': batch[0],
            #           'attention_mask': batch[1],
            #           'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,
            #           # XLM and RoBERTa don't use segment_ids
            #           'labels': batch[3]}

            #             inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}

            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1

        # if preds is None:
        #     preds = logits.detach().cpu().numpy()
        #     out_label_ids = batch[3].detach().cpu().numpy()
        # else:
        #     preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
        #     # print(len(preds))
        #     out_label_ids = np.append(out_label_ids, batch[3].detach().cpu().numpy(), axis=0)
        # if len(ids) == 0:
        #     ids.append(batch[4].detach().cpu().numpy())
        # else:
        #     ids[0] = np.append(
        #         ids[0], batch[4].detach().cpu().numpy(), axis=0)
        if preds is None:
            # preds = logits.detach().cpu().numpy()
            preds = logits.detach().cpu().numpy()
            if args.doc_batching:
                out_label_ids = batch[3][0].detach().cpu().numpy()
            else:
                out_label_ids = batch[3].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            # preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            # print(len(preds))
            if args.doc_batching:
                out_label_ids = np.append(out_label_ids, batch[3][0].detach().cpu().numpy(), axis=0)
            else:
                out_label_ids = np.append(out_label_ids, batch[3].detach().cpu().numpy(), axis=0)

        if len(ids) == 0:
            if args.doc_batching:
                ids.append(batch[4][0].detach().cpu().numpy().item())
            else:
                ids.append(batch[4].detach().cpu().numpy())

        else:
            if args.doc_batching:
                ids.append(batch[4][0].detach().cpu().numpy().item())
            else:
                ids[0] = np.append(
                    ids[0], batch[4].detach().cpu().numpy(), axis=0)


    eval_loss = eval_loss / nb_eval_steps

    preds = preds.reshape((len(eval_dataset), args.num_labels)) ### added
    out_label_ids = out_label_ids.reshape((len(eval_dataset), args.num_labels)) ### added

    preds = sigmoid(preds)



    # preds = (preds > args.prediction_threshold).astype(int)

    preds[preds<args.prediction_threshold] = 0
    sorted_preds_idx = np.flip(np.argsort(preds))
    preds = (preds > args.prediction_threshold)

    result = acc_and_f1(preds, out_label_ids)
    results.update(result)

    # n_labels = np.sum(preds, axis=1)
    # # n_labels = np.sum(preds)
    # preds = np.array([sorted_preds_idx[i,:n] for i,n in enumerate(n_labels)])
    # # preds = np.array(sorted_preds_idx[:n_labels])

    n_labels = np.sum(preds, axis=1)
    preds = np.array([sorted_preds_idx[i,:n] for i,n in enumerate(n_labels)])

    if not args.doc_batching:
        ids = ids[0]
    # ids = np.array([i for i in range(ids[-1]+1)])





    with open(os.path.join(args.data_dir, "mlb_{}_{}.p".format(args.label_threshold, args.ignore_labelless_docs)),
              "rb") as rf:
        mlb = pickle.load(rf)
    # preds = [mlb.classes_[preds[i, :].astype(bool)].tolist() for i in range(preds.shape[0])]

    preds = [mlb.classes_[preds[i][:]].tolist() for i in range(preds.shape[0])]

    id2preds = {val: preds[i] for i, val in enumerate(ids)}
    preds = [id2preds[val] if val in id2preds else [] for i, val in enumerate(ids)]

    with open(os.path.join(args.output_dir, "preds_development.tsv"), "w") as wf:
        wf.write("file\tcode\n")
        for idx, doc_id in enumerate(ids):
            for p in preds[idx]:
                line = str(idx2id[doc_id]) + "\t" + p + "\n"
                wf.write(line)

    eval_cmd = 'python cantemist-evaluation-library/src/main.py -g ' \
               'cantemist/dev-set1-to-publish/cantemist-coding/dev1-coding.tsv -p {}/preds_development.tsv ' \
               '-c cantemist-evaluation-library/valid-codes.tsv -s coding'.format(args.output_dir)
    # eval_cmd = eval_cmd.format(
    #     "../CLEF_Datasets_ICD/2019_German/nts-icd_train/ids_development.txt",
    #     "../CLEF_Datasets_ICD/2019_German/nts-icd_train/anns_train_dev.txt",
    #     "{}/preds_development.txt".format(args.output_dir),
    #     "challenge_eval_output.txt"
    # )

    output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results {} *****".format(prefix))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))
        eval_results = os.popen(eval_cmd).read()
        print("*** Eval results with challenge script: *** ")
        print(eval_results)
        writer.write(eval_results)

    return results


def generate_test_preds(args, model, tokenizer, prefix=""):
    test_output_dir = args.output_dir

    results = {}

    test_dataset, idx2id = load_and_cache_examples(args, tokenizer, test=True)

    if not os.path.exists(test_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(test_output_dir)

    args.test_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    test_sampler = SequentialSampler(test_dataset) if args.local_rank == -1 else DistributedSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.test_batch_size)

    # Predict!
    logger.info("***** Running test {} *****".format(prefix))
    logger.info("  Num examples = %d", len(test_dataset))
    logger.info("  Batch size = %d", args.test_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    ids = []

    for batch in tqdm(test_dataloader, desc="Evaluating"):

        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}

            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1

        if preds is None:
            preds = logits.detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
        if len(ids) == 0:
            ids.append(batch[4].detach().cpu().numpy())
        else:
            ids[0] = np.append(
                ids[0], batch[4].detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps

    preds = sigmoid(preds)

    preds = (preds > args.prediction_threshold).astype(int)
    ids = ids[0]

    with open(os.path.join(args.data_dir, "mlb_{}_{}.p".format(args.label_threshold, args.ignore_labelless_docs)),
              "rb") as rf:
        mlb = pickle.load(rf)
    preds = [mlb.classes_[preds[i, :].astype(bool)].tolist() for i in range(preds.shape[0])]
    id2preds = {val: preds[i] for i, val in enumerate(ids)}
    preds = [id2preds[val] if val in id2preds else [] for i, val in enumerate(ids)]

    with open(os.path.join(args.output_dir, "preds_test.txt"), "w") as wf:
        for idx, doc_id in enumerate(ids):
            line = str(idx2id[doc_id]) + "\t" + "|".join(preds[idx]) + "\n"
            wf.write(line)


def evaluate_test_preds(args):
    results = {}
    with open(os.path.join(args.output_dir, "preds_test.txt"), "r") as rf:
        preds = [line.split('\t') for line in rf.read().split('\n') if line]
        pred_ids, preds = [l[0] for l in preds], [l[1].split('|') for l in preds]

    with open(os.path.join(args.data_dir, "test_gold_{}.tsv".format(args.ignore_labelless_docs)), "r") as rf:
        gold = [line.split('\t') for line in rf.read().split('\n')[1:] if line]
        gold_ids, gold = [l[0] for l in gold], [l[1].split('|') for l in gold]

    assert pred_ids == gold_ids, print("Preds/Gold IDs mismatch")
    mlb_test = MultiLabelBinarizer()
    mlb_test.fit(gold + preds)
    num_uniq_pred_labels = len(set([item for sublist in preds for item in sublist]))
    num_uniq_gold_labels = len(set([item for sublist in gold for item in sublist]))
    gold = mlb_test.transform(gold)
    preds = mlb_test.transform(preds)

    result = acc_and_f1(preds, gold)
    results.update(result)

    output_test_file = os.path.join(args.output_dir, "test_results.txt")
    with open(output_test_file, "w") as writer:
        logger.info("***** Test Results Normal *****")
        logger.info("   Num Uniq Total Labels {} ".format(str(len(mlb_test.classes_))))
        logger.info("   Num Uniq Pred Labels {} ".format(str(num_uniq_pred_labels)))
        logger.info("   Num Uniq Gold Labels {} ".format(str(num_uniq_gold_labels)))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

    eval_cmd = 'python evaluation.py --ids_file="{}" --anns_file="{}" --dev_file="{}" --out_file="{}"'

    eval_cmd = eval_cmd.format(
        "../CLEF_Datasets_ICD/2019_German/nts_icd_test/ids_test.txt",
        "../CLEF_Datasets_ICD/2019_German/nts_icd_test/anns_test.txt",
        "{}/preds_test.txt".format(args.output_dir),
        "test_output.txt"
    )

    eval_results = os.popen(eval_cmd).read()
    print("*** Eval results with challenge script: *** ")
    print(eval_results)



def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS),
    )

    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    # Other parameters
    parser.add_argument(
        "--prediction_threshold",
        default=0.5,
        type=float,
        help="Threshold at which to decide between 0 and 1 for labels.",
    )
    parser.add_argument(
        "--loss_fct", default="bce", type=str, help="The loss function to use.",
    )
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--doc_max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
             "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument("--label_threshold", type=int, default=0, help="Exclude labels which occur <= threshold")
    parser.add_argument("--logit_aggregation", type=str, default='max', help="Whether to aggregate logits by max value "
                                                                             "or average value. Options:"
                                                                             "'--max', '--avg'")
    parser.add_argument("--preprocess", action="store_true", help="Whether to do the initial processing of the data.")
    parser.add_argument("--ignore_labelless_docs", action="store_true",
                        help="Whether to ignore the documents which have no labels.")
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true', help="Whether to run testing.")
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Run evaluation during training at each logging step.",
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model.",
    )

    parser.add_argument(
        "--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument('--do_iterative_class_weights', action='store_true', help="Whether to use iteratively "
                                                                                  "calculated class weights")
    parser.add_argument('--do_normal_class_weights', action='store_true', help="Whether to use normally "
                                                                                  "calculated class weights")
    parser.add_argument('--do_ranking_loss', action='store_true', help="Whether to use the ranking loss component.")
    parser.add_argument('--do_weighted_ranking_loss', action='store_true', help="Whether to use the weighted ranking loss component.")
    parser.add_argument('--doc_batching', action='store_true', help="Whether to fit one document into a batch during")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform.",
    )
    parser.add_argument('--make_plots', action='store_true', help="Whether to make plots on data.")
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_proportion", default=0.1, type=float, help="Linear warmup over warmup proportion.")

    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
             "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
    args = parser.parse_args()
    if args.doc_batching:
        args.per_gpu_train_batch_size = 10
        args.per_gpu_eval_batch_size = 10

    if (
            os.path.exists(args.output_dir)
            and os.listdir(args.output_dir)
            and args.do_train
            and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)

    if args.preprocess:
        cantemist_reader = CantemistReader(args)
        # spanish_reader.process_data()
        # german_reader = GermanReader(args)
        # german_reader.process_data()
        args.overwrite_cache = True

    # Prepare task
    try:
        processor = MyProcessor(args)
    except:
        cantemist_reader = CantemistReader(args)
        cantemist_reader.process_data()
        # spanish_reader = SpanishReader(args)
        # spanish_reader.process_data()
        # german_reader = GermanReader(args)
        # german_reader.process_data()
        processor = MyProcessor(args)

    class_weights = processor.get_class_weights()
    label_list = processor.get_labels()
    num_labels = len(label_list)
    args.num_labels = num_labels

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task="offenseval",
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    # model = model_class.from_pretrained(
    #     args.model_name_or_path,
    #     from_tf=bool(".ckpt" in args.model_name_or_path),
    #     config=config,
    #     cache_dir=args.cache_dir if args.cache_dir else None,
    #     num_labels=num_labels,
    #     loss_fct=args.loss_fct
    # )
    model = model_class.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        loss_fct=args.loss_fct,
        args=args,
        class_weights=class_weights,
    )
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name)

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        train_dataset, idx2id = load_and_cache_examples(args, tokenizer, evaluate=False)
        global_step, tr_loss = train(args, train_dataset, model, tokenizer, class_weights)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

        # Load a trained model and vocabulary that you have fine-tuned
        model = model_class.from_pretrained(args.output_dir, args=args, loss_fct=args.loss_fct)
        tokenizer = tokenizer_class.from_pretrained(args.output_dir)
        model.to(args.device)

    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""

            model = model_class.from_pretrained(args.output_dir, loss_fct=args.loss_fct, args=args)
            # model = model_class.from_pretrained(
            #     checkpoint,
            #     from_tf=bool(".ckpt" in args.model_name_or_path),
            #     config=config,
            #     cache_dir=args.cache_dir if args.cache_dir else None,
            #     num_labels=num_labels,
            #     loss_fct=args.loss_fct
            # )
            model.to(args.device)
            result = evaluate(args, model, tokenizer, prefix=prefix)
            result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
            results.update(result)

    # Make predictions on test set
    results = {}
    if args.do_test and args.local_rank in [-1, 0]:
        if not os.path.exists(os.path.join(args.output_dir, "preds_test.txt")):
            tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
            checkpoints = [args.output_dir]
            if args.eval_all_checkpoints:
                checkpoints = list(
                    os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
                )
                logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
            logger.info("Evaluate the following checkpoints: %s", checkpoints)
            for checkpoint in checkpoints:
                global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
                prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""
                model = model_class.from_pretrained(checkpoint, args=args, loss_fct=args.loss_fct)
                model.to(args.device)
                predictions = generate_test_preds(args, model, tokenizer, prefix=global_step)
        evaluate_test_preds(args)


    return results


if __name__ == '__main__':
    main()
