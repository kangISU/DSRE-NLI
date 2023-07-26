import argparse
import json
import os

import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from utils import set_seed, collate_fn
from prepro import MultiClassProcessor
from evaluation import get_f1, get_indiv_f1, generate_predicted_file
from model import REModel
from torch.cuda.amp import GradScaler
import wandb
import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def train(args, model, train_features, benchmarks):
    train_dataloader = DataLoader(train_features, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn,
                                  drop_last=True)
    total_steps = int(len(train_dataloader) * args.num_train_epochs // args.gradient_accumulation_steps)
    warmup_steps = int(total_steps * args.warmup_ratio)

    scaler = GradScaler()
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=total_steps)
    logger.info('Total steps: {}'.format(total_steps))
    logger.info('Warmup steps: {}'.format(warmup_steps))

    pos_loss = []

    num_steps = 0
    for epoch in range(int(args.num_train_epochs)):
        model.zero_grad()
        for step, batch in enumerate(tqdm(train_dataloader)):
            model.train()

            inputs = {'stage': 'train',
                      'input_ids': batch[1].to(args.device),
                      'attention_mask': batch[2].to(args.device),
                      'labels': batch[3].to(args.device),
                      'ss': batch[4].to(args.device),
                      'os': batch[5].to(args.device),
                      }

            outputs = model(**inputs)
            loss = outputs[0] / args.gradient_accumulation_steps
            scaler.scale(loss).backward()
            if step % args.gradient_accumulation_steps == 0:
                num_steps += 1
                if args.max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                model.zero_grad()
                wandb.log({'loss': loss.item()}, step=num_steps)
                logger.info('loss: {} step: {}'.format(loss.item(), num_steps))

            if num_steps % args.evaluation_steps == 0 and step % args.gradient_accumulation_steps == 0:
                logger.info('===== evaluation in training =====')
                for tag, features in benchmarks:
                    if tag != 'train':
                        f1, output, indiv_prf = evaluate(args, model, features, tag=tag)
                        wandb.log(output, step=num_steps)

    logger.info('===== evaluation in testing =====')
    for tag, features in benchmarks:
        f1, output, indiv_prf = evaluate(args, model, features, tag=tag)
        wandb.log(output, step=num_steps)
        id2rel = {args.rel2id[k]: k for k in args.rel2id.keys()}
        print(f'==== {tag} ====')
        for r in indiv_prf.keys():
            print(id2rel[r] + ':')
            print(indiv_prf[r])
            print('\n')

    return pos_loss


def evaluate(args, model, features, tag='test'):
    dataloader = DataLoader(features, batch_size=args.test_batch_size, collate_fn=collate_fn, drop_last=False)
    keys, preds = [], []
    for i_b, batch in enumerate(dataloader):
        model.eval()

        inputs = {'input_ids': batch[1].to(args.device),
                  'attention_mask': batch[2].to(args.device),
                  'ss': batch[4].to(args.device),
                  'os': batch[5].to(args.device),
                  }

        with torch.no_grad():
            logit = model(**inputs)[0]
            pred = torch.argmax(logit, dim=-1)
        preds += pred.tolist()
        keys += batch[3].tolist()

    keys = np.array(keys, dtype=np.int64)
    preds = np.array(preds, dtype=np.int64)
    prec, reca, f1, correct, guessed, gold = get_f1(keys, preds)
    indiv_prf = get_indiv_f1(args.num_class, keys, preds)

    output = {
        tag: {"f1": f1 * 100, "precision": prec * 100, "recall": reca * 100, "correct": correct, "guessed": guessed, "gold": gold}
    }
    logger.info(output)
    logger.info(indiv_prf)

    return f1, output, indiv_prf


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--project_name", type=str, default='DSRE-NLI')
    parser.add_argument("--run_name", type=str, default='nyt1')

    parser.add_argument("--data_dir", type=str, default='nyt1')
    parser.add_argument("--train_file", type=str, default='train_genr_patt_npin.json')
    parser.add_argument("--model_name_or_path", default="bert-base-cased", type=str)
    parser.add_argument("--input_format", default="entity_mask", type=str,
                        help="in [entity_mask, entity_marker, entity_marker_punct, typed_entity_marker, typed_entity_marker_punct]")

    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated.")

    parser.add_argument("--train_batch_size", default=64, type=int,
                        help="Batch size for training.")
    parser.add_argument("--test_batch_size", default=64, type=int,
                        help="Batch size for testing.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int,
                        help="Number of updates steps to accumulate the gradients for, before performing a backward/update pass.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--warmup_ratio", default=0.1, type=float,
                        help="Warm up ratio for Adam.")
    parser.add_argument("--num_train_epochs", default=2.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--seed", type=int, default=78,
                        help="random seed for initialization")
    parser.add_argument("--num_class", type=int, default=11)
    parser.add_argument("--gamma", type=int, default=1,
                        help="a hyperparameter to cope with class imbalance")
    parser.add_argument("--evaluation_steps", type=int, default=300,
                        help="Number of steps to evaluate the model")
    parser.add_argument("--dropout_prob", type=float, default=0.1)

    args = parser.parse_args()

    args.data_dir = os.path.join('../dataset', args.data_dir)

    rel2id_file = os.path.join(args.data_dir, "rel2id.json")
    with open(rel2id_file, 'r', encoding='utf-8') as rf:
        rel2id = json.loads(rf.read())

    args.num_class = len(rel2id)

    logger.info('Arguments:')
    for arg in vars(args):
        logger.info('    {}: {}'.format(arg, getattr(args, arg)))

    args.rel2id = rel2id

    run = wandb.init(project=args.project_name, name=args.run_name, reinit=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    if args.seed > 0:
        set_seed(args)

    config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        num_labels=args.num_class
    )
    config.gradient_checkpointing = True

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    train_file = os.path.join(args.data_dir, args.train_file)
    # dev_test_file = os.path.join(args.data_dir, 'dev_test.json')
    dev_file = os.path.join(args.data_dir, 'dev.json')
    test_file = os.path.join(args.data_dir, 'test.json')

    processor = MultiClassProcessor(args, tokenizer, rel2id)
    train_features = processor.read(train_file)
    # dev_test_features = processor.read(dev_test_file)
    dev_features = processor.read(dev_file)
    test_features = processor.read(test_file)

    args.prior, args.weight = processor.cal_prior_weight(train_file)
    args.prior = torch.tensor(args.prior, dtype=torch.float)
    args.weight = torch.tensor(args.weight, dtype=torch.float)

    print('prior: ', args.prior)
    print('weight: ', args.weight)

    model = REModel(args, config)
    model.to(0)

    if len(processor.new_tokens) > 0:
        model.encoder.resize_token_embeddings(len(tokenizer))

    benchmarks = (
        ("train", train_features),
        # ("dev_test", dev_test_features),
        ("dev", dev_features),
        ("test", test_features),
    )

    train(args, model, train_features, benchmarks)

    run.finish()


if __name__ == "__main__":
    main()
