from collections import defaultdict
from typing import Dict, List
from dataclasses import dataclass

import os
import sys
import gc
import numpy as np
import torch
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoConfig,
)


@dataclass
class REInputFeatures:
    subj: str
    obj: str
    context: str
    pair_type: str = None
    label: str = None


class Classifier(object):
    """Abstact classifier class."""

    def __init__(
            self, labels: List[str], pretrained_model: str = "roberta-large-mnli", use_cuda=True, half=False, verbose=True
    ):
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
        self.labels = labels
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

    def __call__(self, context, batch_size=1):
        raise NotImplementedError

    def clear_gpu_memory(self):
        self.model.cpu()
        del self.model
        gc.collect()
        torch.cuda.empty_cache()


class _NLIRelationClassifier(Classifier):
    def __init__(
            self,
            labels: List[str],
            *args,
            pretrained_model: str = "roberta-large-mnli",
            use_cuda=True,
            half=False,
            verbose=True,
            # negative_threshold=0.95,
            negative_idx=0,
            max_activations=np.inf,
            valid_conditions=None,
            **kwargs,
    ):
        super().__init__(
            labels,
            pretrained_model=pretrained_model,
            use_cuda=use_cuda,
            verbose=verbose,
            half=half,
        )
        # self.ent_pos = entailment_position
        # self.cont_pos = -1 if self.ent_pos == 0 else 0
        self.negative_threshold = kwargs["negative_threshold"]
        self.negative_idx = negative_idx
        self.max_activations = max_activations
        self.n_rel = len(labels)
        # for label in labels:
        #     assert '{subj}' in label and '{obj}' in label

        if valid_conditions:
            self.valid_conditions = {}
            rel2id = {r: i for i, r in enumerate(labels)}
            self.n_rel = len(rel2id)
            for relation, conditions in valid_conditions.items():
                if relation not in rel2id:
                    continue
                for condition in conditions:
                    if condition not in self.valid_conditions:
                        self.valid_conditions[condition] = np.zeros(self.n_rel)
                        self.valid_conditions[condition][rel2id[kwargs["negative_tag"]]] = 1.0  # TODO (modify)
                    self.valid_conditions[condition][rel2id[relation]] = 1.0

        else:
            self.valid_conditions = None

        def idx2label(idx):
            return self.labels[idx]

        self.idx2label = np.vectorize(idx2label)

    def _initialize(self, pretrained_model):
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.model = AutoModelForSequenceClassification.from_pretrained(pretrained_model)
        self.config = AutoConfig.from_pretrained(pretrained_model)
        self.ent_pos = self.config.label2id.get("ENTAILMENT", self.config.label2id.get("entailment", None))
        if self.ent_pos is None:
            raise ValueError("The model config must contain ENTAILMENT label in the label2id dict.")
        else:
            self.ent_pos = int(self.ent_pos)

    def _run_batch(self, batch, multiclass=False):
        with torch.no_grad():
            input_ids = self.tokenizer.batch_encode_plus(batch, padding=True, truncation=True)
            input_ids = torch.tensor(input_ids["input_ids"]).to(self.device)
            output = self.model(input_ids)[0].detach().cpu().numpy()
            if multiclass:
                output = np.exp(output) / np.exp(output).sum(
                    -1, keepdims=True
                )  # np.exp(output[..., [self.cont_pos, self.ent_pos]]).sum(-1, keepdims=True)
            output = output[..., self.ent_pos].reshape(input_ids.shape[0] // len(self.labels), -1)

        return output

    def __call__(
            self,
            features: List[REInputFeatures],
            batch_size: int = 1,
            multiclass=False,
    ):
        if not isinstance(features, list):
            features = [features]

        batch, outputs = [], []
        for i, feature in tqdm(enumerate(features), total=len(features)):
            sentences = [
                f"{feature.context} {self.tokenizer.sep_token} {label_template.format(subj=feature.subj, obj=feature.obj)}."
                for label_template in self.labels
            ]
            batch.extend(sentences)

            if (i + 1) % batch_size == 0:
                output = self._run_batch(batch, multiclass=multiclass)
                outputs.append(output)
                batch = []

        if len(batch) > 0:
            output = self._run_batch(batch, multiclass=multiclass)
            outputs.append(output)

        outputs = np.vstack(outputs)

        return outputs

    def _apply_negative_threshold(self, probs):
        activations = (probs >= self.negative_threshold).sum(-1).astype(int)
        idx = np.logical_or(
            activations == 0, activations >= self.max_activations
        )  # If there are no activations then is a negative example, if there are too many, then is a noisy example
        probs[idx, self.negative_idx] = 1.00
        return probs

    def _apply_valid_conditions(self, probs, features: List[REInputFeatures]):
        mask_matrix = np.stack(
            [self.valid_conditions.get(feature.pair_type, np.zeros(self.n_rel)) for feature in features],
            axis=0,
        )
        probs = probs * mask_matrix

        return probs

    def predict(
            self,
            contexts: List[REInputFeatures],
            batch_size: int = 1,
            return_labels: bool = True,
            return_confidences: bool = False,
            topk: int = 1,
    ):
        output = self(contexts, batch_size)
        topics = np.argsort(output, -1)[:, ::-1][:, :topk]
        if return_labels:
            topics = self.idx2label(topics)
        if return_confidences:
            topics = np.stack((topics, np.sort(output, -1)[:, ::-1][:, :topk]), -1).tolist()
            topics = [
                [(int(label), float(conf)) if not return_labels else (label, float(conf)) for label, conf in row]
                for row in topics
            ]
        else:
            topics = topics.tolist()
        if topk == 1:
            topics = [row[0] for row in topics]

        return topics


class NLIRelationClassifierWithMappingHead(_NLIRelationClassifier):
    def __init__(
            self,
            labels: List[str],
            template_mapping: Dict[str, list],
            pretrained_model: str = "roberta-large-mnli",
            valid_conditions: Dict[str, list] = None,
            *args,
            **kwargs,
    ):

        self.template_mapping_reverse = defaultdict(list)
        for key, value in template_mapping.items():
            for v in value:
                self.template_mapping_reverse[v].append(key)
        self.new_topics = list(self.template_mapping_reverse.keys())

        self.target_labels = labels
        self.new_labels2id = {t: i for i, t in enumerate(self.new_topics)}
        self.mapping = defaultdict(list)
        for key, value in template_mapping.items():
            self.mapping[key].extend([self.new_labels2id[v] for v in value])

        super().__init__(
            self.new_topics,
            *args,
            pretrained_model=pretrained_model,
            valid_conditions=None,
            **kwargs,
        )

        if valid_conditions:
            self.valid_conditions = {}
            rel2id = {r: i for i, r in enumerate(labels)}
            self.n_rel = len(rel2id)
            for relation, conditions in valid_conditions.items():
                if relation not in rel2id:
                    continue
                for condition in conditions:
                    if condition not in self.valid_conditions:
                        self.valid_conditions[condition] = np.zeros(self.n_rel)
                        self.valid_conditions[condition][rel2id[kwargs["negative_tag"]]] = 1.0  # TODO (modify)
                    self.valid_conditions[condition][rel2id[relation]] = 1.0

        else:
            self.valid_conditions = None

        def idx2label(idx):
            return self.target_labels[idx]

        self.idx2label = np.vectorize(idx2label)

    def __call__(self, features: List[REInputFeatures], batch_size=1, multiclass=True):
        outputs = super().__call__(features, batch_size, multiclass)
        outputs = np.hstack(
            [
                np.max(outputs[:, self.mapping[label]], axis=-1, keepdims=True)
                if label in self.mapping
                else np.zeros((outputs.shape[0], 1))
                for label in self.target_labels
            ]
        )

        outputs = np_softmax(outputs) if not multiclass else outputs

        if self.valid_conditions:
            outputs = self._apply_valid_conditions(outputs, features)

        outputs = self._apply_negative_threshold(outputs)

        return outputs


def np_softmax(x, dim=-1):
    e = np.exp(x)
    return e / np.sum(e, axis=dim, keepdims=True)


def main():
    pass


if __name__ == "__main__":
    main()
