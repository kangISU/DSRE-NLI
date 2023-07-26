import argparse
import logging
from dataclasses import dataclass
import string
import re
import os
import json
import sys
import gc
import numpy as np
import torch
from numpy import median
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoConfig,
)

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def median_freq(patts):
    freqs = []
    for p in patts:
        if p['freq'] > 0:
            freqs.append(p['freq'])

    return median(list(set(freqs)))


@dataclass
class Feature:
    sent1: str
    sent2: str


class Classifier(object):
    """Abstact classifier class."""

    def __init__(
            self, pretrained_model: str = "roberta-large-mnli", use_cuda=True, half=False, verbose=True
    ):
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
        self.use_cuda = use_cuda
        self.half = half
        self.verbose = verbose

        # Supress stdout printing for model downloads
        if not verbose:
            sys.stdout = open(os.devnull, "w")
            self._initialize(pretrained_model)
            sys.stdout = sys.__stdout__
        else:
            self._initialize(pretrained_model)

        self.model = self.model.to(self.device)
        self.model = self.model.eval()
        if self.use_cuda and self.half and torch.cuda.is_available():
            self.model = self.model.half()

    def _initialize(self, pretrained_model):
        raise NotImplementedError

    def __call__(self, context):
        raise NotImplementedError

    def clear_gpu_memory(self):
        self.model.cpu()
        del self.model
        gc.collect()
        torch.cuda.empty_cache()


class TextEntailment(Classifier):
    def __init__(
            self,
            pretrained_model: str = "roberta-large-mnli",
            use_cuda=True,
            verbose=True,
            half=False
    ):
        super().__init__(
            pretrained_model=pretrained_model,
            use_cuda=use_cuda,
            verbose=verbose,
            half=half
        )

    def _initialize(self, pretrained_model):
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.model = AutoModelForSequenceClassification.from_pretrained(pretrained_model)
        self.config = AutoConfig.from_pretrained(pretrained_model)
        self.ent_pos = self.config.label2id.get("ENTAILMENT", self.config.label2id.get("entailment", None))
        if self.ent_pos is None:
            raise ValueError("The model config must contain ENTAILMENT label in the label2id dict.")
        else:
            self.ent_pos = int(self.ent_pos)

    def _run_batch(self, batch):
        with torch.no_grad():
            input_ids = self.tokenizer.batch_encode_plus(batch, padding=True, truncation=True)
            input_ids = torch.tensor(input_ids["input_ids"]).to(self.device)
            output = self.model(input_ids)[0].detach().cpu().numpy()
            output = np.exp(output) / np.exp(output).sum(
                -1, keepdims=True
            )
            output = output[..., self.ent_pos].reshape(1, -1)

        return output[0]

    def __call__(
            self,
            feature: Feature
    ):
        sentences = [
            f"{feature.sent1} {self.tokenizer.sep_token} {feature.sent2}."
        ]

        output = self._run_batch(sentences)

        return output


def intial_patterns(train_data_file, rel_file, neg_tag):
    with open(rel_file, 'r', encoding='utf-8') as rf:
        rels = json.loads(rf.read())
        rels = list(rels.keys())
        rels.remove(neg_tag)

    non_sense = []
    with open('stop_words.txt', 'r', encoding='utf-8') as fp:
        stop_words = eval(fp.read())
    non_sense.extend(stop_words)
    punctuations = [p for p in string.punctuation] + ['--', '\'\'', '', '-LRB-', '-RRB-']
    non_sense.extend(punctuations)

    rel_patts = {}
    with open(train_data_file, 'r', encoding='utf-8') as tf:
        data = json.load(tf)
        for inst in data:
            rel = inst['relation']
            if rel in rels:
                if rel not in rel_patts:
                    rel_patts[rel] = {}
                if inst['subj_end'] < inst['obj_start']:
                    between_tokens = ' '.join(inst['token'][inst['subj_end'] + 1: inst['obj_start']])
                    patt = ' '.join(['{subj}', between_tokens, '{obj}'])
                else:
                    between_tokens = ' '.join(inst['token'][inst['obj_end'] + 1: inst['subj_start']])
                    patt = ' '.join(['{obj}', between_tokens, '{subj}'])

                if between_tokens not in non_sense and not all([token in non_sense for token in between_tokens.split(' ')]):
                    if patt in rel_patts[rel]:
                        rel_patts[rel][patt] += 1
                    else:
                        rel_patts[rel][patt] = 1
    return rel_patts


def save_patterns(rel_patts, save_path):
    for rel in rel_patts.keys():
        patts = rel_patts[rel]
        patts = {k: v for k, v in sorted(patts.items(), key=lambda item: item[1], reverse=True)}
        patts_list = []
        for p, f in patts.items():
            patts_list.append({'freq': f, 'patt': p})

        rel_name = re.sub('[:, /]', '.', rel)
        if rel_name.startswith('.'):
            rel_name = rel_name[1:]
        file_name = rel_name + '.json'
        file_path = os.path.join(save_path, file_name)
        with open(file_path, 'w', encoding='utf-8') as fp:
            fp.write(
                '[' +
                ',\n'.join(json.dumps(p) for p in patts_list) +
                ']\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default='../dataset/nyt1')
    parser.add_argument("--pretrained_model", type=str, default="microsoft/deberta-v2-xlarge-mnli")
    parser.add_argument("--entail_threshold", type=float, default=0.95)
    parser.add_argument("--neg_tag", type=str, default='None')

    args = parser.parse_args()

    logger.info('Arguments:')
    for arg in vars(args):
        logger.info('    {}: {}'.format(arg, getattr(args, arg)))

    clf = TextEntailment(
        pretrained_model=args.pretrained_model,
    )

    """
    step1: generate initial patterns
    """
    initial_patts_dir = os.path.join(args.dataset_dir, 'initial_patterns')
    if not os.path.exists(initial_patts_dir):
        os.mkdir(initial_patts_dir)

    train_file = os.path.join(args.dataset_dir, 'train.json')
    rel_file = os.path.join(args.dataset_dir, 'rel2id.json')

    rel_patterns = intial_patterns(train_file, rel_file, args.neg_tag)
    save_patterns(rel_patterns, initial_patts_dir)

    """
    step2: pattern screening (discard patterns that entail the seeds)
    """
    if 'nyt1' in args.dataset_dir:
        genr_templates = {
            "/location/location/contains": [
                "{obj} is located in {subj}"
            ],
            "/people/person/nationality": [
                "{obj} is the nationality of {subj}"
            ],
            "/location/country/capital": [
                "{obj} is the capital of {subj}"
            ],
            "/people/person/place_lived": [
                "{subj} lives in {obj}"
            ],
            "/business/person/company": [
                "{subj} works in {obj}"
            ],
            "/location/neighborhood/neighborhood_of": [
                "{subj} is in the neighborhood of {obj}"
            ],
            "/people/person/place_of_birth": [
                "{subj} was born in {obj}"
            ],
            "/people/deceased_person/place_of_death": [
                "{subj} died in {obj}"
            ],
            "/business/company/founders": [
                "{subj} was founded by {obj}"
            ],
            "/people/person/children": [
                "{subj} is the parent of {obj}"
            ]
        }
    elif 'nyt2' in args.dataset_dir:
        genr_templates = {
            "/location/location/contains": [
                "{obj} is located in {subj}"
            ],
            "/people/person/nationality": [
                "{obj} is the nationality of {subj}"
            ],
            "/location/country/capital": [
                "{obj} is the capital of {subj}"
            ],
            "/people/person/place_lived": [
                "{subj} lives in {obj}"
            ],
            "/business/person/company": [
                "{subj} works in {obj}"
            ],
            "/location/neighborhood/neighborhood_of": [
                "{subj} is in the neighborhood of {obj}"
            ],
            "/people/person/place_of_birth": [
                "{subj} was born in {obj}"
            ],
            "/people/deceased_person/place_of_death": [
                "{subj} died in {obj}"
            ],
            "/business/company/founders": [
                "{subj} was founded by {obj}"
            ],
            "/people/person/children": [
                "{subj} is the parent of {obj}"
            ],
            "/business/company/place_founded": [
                "{subj} was founded in {obj}"
            ]
        }
    elif 'tacrev-s' in args.dataset_dir:
        genr_templates = {
            "per:alternate_names": ["{subj} is also known as {obj}"],
            "per:date_of_birth": ["{subj} was born in {obj}"],
            "per:age": ["{subj} is {obj} years old"],
            "per:country_of_birth": ["{subj} was born in {obj}"],
            "per:stateorprovince_of_birth": ["{subj} was born in {obj}"],
            "per:city_of_birth": ["{subj} was born in {obj}"],
            "per:origin": ["{obj} is the nationality of {subj}"],
            "per:date_of_death": ["{subj} died in {obj}"],
            "per:country_of_death": ["{subj} died in {obj}"],
            "per:stateorprovince_of_death": ["{subj} died in {obj}"],
            "per:city_of_death": ["{subj} died in {obj}"],
            "per:cause_of_death": ["{obj} is the cause of {subj}’s death"],
            "per:countries_of_residence": ["{subj} lives in {obj}"],
            "per:stateorprovinces_of_residence": ["{subj} lives in {obj}"],
            "per:cities_of_residence": ["{subj} lives in {obj}"],
            "per:schools_attended": ["{subj} studied in {obj}"],
            "per:title": ["{subj} is a {obj}"],
            "per:employee_of": ["{subj} is an employee of {obj}"],
            "per:religion": ["{obj} is the religion of {subj}"],
            "per:parents": ["{obj} is the parent of {subj}"],
            "per:spouse": ["{subj} is the spouse of {obj}"],
            "per:children": ["{subj} is the parent of {obj}"],
            "per:siblings": ["{subj} and {obj} are siblings"],
            "per:other_family": ["{subj} and {obj} are family"],
            "per:charges": ["{obj} are the charges of {subj}"],
            "org:alternate_names": ["{subj} is also known as {obj}"],
            "org:political/religious_affiliation": ["{subj} has an affiliation with {obj}"],
            "org:top_members/employees": ["{obj} is a high level member of {subj}"],
            "org:number_of_employees/members": ["{subj} has about {obj} employees"],
            "org:members": ["{obj} is member of {subj}"],
            "org:member_of": ["{subj} is member of {obj}"],
            "org:subsidiaries": ["{obj} is a subsidiary of {subj}"],
            "org:parents": ["{subj} is a subsidiary of {obj}"],
            "org:founded_by": ["{subj} was founded by {obj}"],
            "org:founded": ["{subj} was founded in {obj}"],
            "org:dissolved": ["{subj} dissolved in {obj}"],
            "org:country_of_headquarters": ["{subj} has its headquarters in {obj}"],
            "org:stateorprovince_of_headquarters": ["{subj} has its headquarters in {obj}"],
            "org:city_of_headquarters": ["{subj} has its headquarters in {obj}"],
            "org:shareholders": ["{obj} holds shares in {subj}"],
            "org:website": ["{obj} is the website of {subj}"],
        }
    else:
        raise Exception('Check dataset name')

    screened_patts_dir = os.path.join(args.dataset_dir, 'screened_patterns')
    if not os.path.exists(screened_patts_dir):
        os.mkdir(screened_patts_dir)

    for rel in tqdm(genr_templates.keys()):
        rel_name = re.sub('[:, /]', '.', rel)
        if rel_name.startswith('.'):
            rel_name = rel_name[1:]
        with open(os.path.join(initial_patts_dir, rel_name + '.json'), 'r', encoding='utf-8') as fp:
            freq_patt = json.load(fp)
            topk = int(len(freq_patt) * 0.1)
            freq_patt = freq_patt[: topk]
            freq_patt = [p for p in freq_patt if len(p['patt'].split(' ')) < 10]
            if len(freq_patt) > 50:
                freq_patt = freq_patt[: 50]

        for i, p in enumerate(freq_patt):
            feature = Feature(
                sent1=p['patt'],
                sent2=genr_templates[rel][0]
            )

            output = clf(feature)
            if output[0] > args.entail_threshold:
                freq_patt[i]['freq'] = 0

        temp = [{'freq': p['freq'], 'patt': p['patt']} for p in freq_patt]
        temp.sort(key=lambda x: x['freq'], reverse=True)

        rel_name = re.sub('[:, /]', '.', rel)
        if rel_name.startswith('.'):
            rel_name = rel_name[1:]
        f_name = os.path.join(screened_patts_dir, rel_name + '.json')
        with open(f_name, 'w', encoding='utf-8') as fp:
            fp.write(
                '[' +
                ',\n'.join(json.dumps(p) for p in temp) +
                ']\n')

    """
    step3: pattern grouping
    """
    grouped_patts_dir = os.path.join(args.dataset_dir, 'grouped_patterns')
    if not os.path.exists(grouped_patts_dir):
        os.mkdir(grouped_patts_dir)

    for rel in tqdm(genr_templates.keys()):
        rel_name = re.sub('[:, /]', '.', rel)
        if rel_name.startswith('.'):
            rel_name = rel_name[1:]
        with open(os.path.join(screened_patts_dir, rel_name + '.json'), 'r', encoding='utf-8') as fp:
            screened_patts = json.load(fp)

        patts = [p for p in screened_patts if p['freq'] > 0]

        def get_key(ele):
            return len(ele['patt'].split(' '))

        patts.sort(key=get_key, reverse=True)

        for p in patts:
            p['new_freq'] = p['freq']

        for i, p in enumerate(patts):
            entail = False
            cands = patts[i + 1:]
            if len(cands) == 0:
                break

            for j, cand in enumerate(cands):
                feature = Feature(
                    sent1=p['patt'],
                    sent2=cand['patt']
                )

                output = clf(feature)
                if output[0] > args.entail_threshold:
                    entail = True
                    patts[j + (i + 1)]['new_freq'] += p['freq']

            if entail:
                p['new_freq'] = 0

        temp = [{'freq': p['new_freq'], 'patt': p['patt']} for p in patts]
        temp.sort(key=lambda x: x['freq'], reverse=True)

        rel_name = re.sub('[:, /]', '.', rel)
        if rel_name.startswith('.'):
            rel_name = rel_name[1:]
        f_name = os.path.join(grouped_patts_dir, rel_name + '.json')
        with open(f_name, 'w', encoding='utf-8') as fp:
            fp.write(
                '[' +
                ',\n'.join(json.dumps(p) for p in temp) +
                ']\n')


if __name__ == '__main__':
    main()
