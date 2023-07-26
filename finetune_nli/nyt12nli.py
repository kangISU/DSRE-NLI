from argparse import ArgumentParser
from dataclasses import dataclass
from collections import defaultdict, Counter
from typing import Dict
import numpy as np
import json
import os
from pprint import pprint
import random

LABELS = [
    "None",
    "/location/location/contains",
    "/people/person/nationality",
    "/location/country/capital",
    "/people/person/place_lived",
    "/business/person/company",
    "/location/neighborhood/neighborhood_of",
    "/people/person/place_of_birth",
    "/people/deceased_person/place_of_death",
    "/business/company/founders",
    "/people/person/children"
]

LABEL_TEMPLATES = {
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


@dataclass
class REInputFeatures:
    subj: str
    obj: str
    context: str
    pair_type: str = None
    label: str = None


@dataclass
class MNLIInputFeatures:
    premise: str
    hypothesis: str
    label: int


def nyt12nli(
        instance: REInputFeatures,
        positive_templates,
        negative_templates,
        templates,
        labels2id,
        negn=1,
        posn=1,
        negt='None'
):
    if instance.label == negt:
        template = random.choices(templates, k=negn)
        return [
            MNLIInputFeatures(
                premise=instance.context,
                hypothesis=f"{t.format(subj=instance.subj, obj=instance.obj)}.",
                label=labels2id["contradiction"],
            )
            for t in template
        ]

    # Generate the positive examples
    mnli_instances = []
    # positive_template = random.choices(positive_templates[instance.label], k=posn)
    positive_template = positive_templates[instance.label]
    mnli_instances.extend(
        [
            MNLIInputFeatures(
                premise=instance.context,
                hypothesis=f"{t.format(subj=instance.subj, obj=instance.obj)}.",
                label=labels2id["entailment"],
            )
            for t in positive_template
        ]
    )

    # Generate the negative templates
    negative_template = random.choices(negative_templates[instance.label], k=negn)
    mnli_instances.extend(
        [
            MNLIInputFeatures(
                premise=instance.context,
                hypothesis=f"{t.format(subj=instance.subj, obj=instance.obj)}.",
                label=labels2id["neutral"],
            )
            for t in negative_template
        ]
    )

    return mnli_instances


def nyt12nli_with_negative_pattern(
        instance: REInputFeatures,
        positive_templates,
        negative_templates,
        templates,
        labels2id,
        negn=1,
        posn=1,
        negt='None'
):
    mnli_instances = []
    # Generate the positive examples
    positive_template = random.choices(positive_templates[instance.label], k=posn)
    mnli_instances.extend(
        [
            MNLIInputFeatures(
                premise=instance.context,
                hypothesis=f"{t.format(subj=instance.subj, obj=instance.obj)}.",
                label=labels2id["entailment"],
            )
            for t in positive_template
        ]
    )

    # Generate the negative templates
    negative_template = random.choices(negative_templates[instance.label], k=negn)
    mnli_instances.extend(
        [
            MNLIInputFeatures(
                premise=instance.context,
                hypothesis=f"{t.format(subj=instance.subj, obj=instance.obj)}.",
                label=labels2id["neutral"] if instance.label != negt else labels2id["contradiction"],
            )
            for t in negative_template
        ]
    )

    # Add the contradiction regarding the no_relation pattern if the relation is not no_relation
    if instance.label != negt:
        mnli_instances.append(
            MNLIInputFeatures(
                premise=instance.context,
                hypothesis="{subj} and {obj} are not related.".format(subj=instance.subj, obj=instance.obj),
                label=labels2id["contradiction"],
            )
        )

    return mnli_instances


def main():
    parser = ArgumentParser()

    parser.add_argument("--dataset", type=str, default="nyt1")
    parser.add_argument("--input", type=str, default="train.json")
    parser.add_argument("--output", type=str, default="train_reformulated.json")
    parser.add_argument("--negative_pattern", action="store_true", default=False)
    parser.add_argument("--negative_tag", type=str, default='None')
    parser.add_argument("--negn", type=int, default=1)

    args = parser.parse_args()

    random.seed(0)
    np.random.seed(0)

    input_file = os.path.join('../dataset', args.dataset, args.input)

    output_path = os.path.join('../data_for_finetune', args.dataset)
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    output_file = os.path.join(output_path, args.output)

    templates = [
        "{subj} and {obj} are not related",
        "{obj} is located in {subj}",
        "{obj} is the nationality of {subj}",
        "{obj} is the capital of {subj}",
        "{subj} lives in {obj}",
        "{subj} works in {obj}",
        "{subj} is in the neighborhood of {obj}",
        "{subj} was born in {obj}",
        "{subj} died in {obj}",
        "{subj} was founded by {obj}",
        "{subj} is the parent of {obj}"
    ]

    labels2id = {"entailment": 2, "neutral": 1, "contradiction": 0}

    positive_templates: Dict[str, list] = defaultdict(list)
    negative_templates: Dict[str, list] = defaultdict(list)

    if not args.negative_pattern:
        templates = templates[1:]

    for label in LABELS:
        if not args.negative_pattern and label == args.negative_tag:
            continue
        for template in templates:
            if label != args.negative_tag and template == "{subj} and {obj} are not related":
                continue
            if template in LABEL_TEMPLATES[label]:
                positive_templates[label].append(template)
            else:
                negative_templates[label].append(template)

    nyt12nli_ = nyt12nli_with_negative_pattern if args.negative_pattern else nyt12nli

    with open(input_file, "rt", encoding='utf-8') as f:
        mnli_data = []
        stats = []
        for line in json.load(f):
            mnli_instance = nyt12nli_(
                REInputFeatures(
                    subj=" ".join(line["token"][line["subj_start"]: line["subj_end"] + 1])
                        .replace("-LRB-", "(")
                        .replace("-RRB-", ")")
                        .replace("-LSB-", "[")
                        .replace("-RSB-", "]"),
                    obj=" ".join(line["token"][line["obj_start"]: line["obj_end"] + 1])
                        .replace("-LRB-", "(")
                        .replace("-RRB-", ")")
                        .replace("-LSB-", "[")
                        .replace("-RSB-", "]"),
                    pair_type=f"{line['subj_type']}:{line['obj_type']}",
                    context=" ".join(line["token"])
                        .replace("-LRB-", "(")
                        .replace("-RRB-", ")")
                        .replace("-LSB-", "[")
                        .replace("-RSB-", "]"),
                    label=line["relation"],
                ),
                positive_templates,
                negative_templates,
                templates,
                labels2id,
                negn=args.negn,
                negt=args.negative_tag,
            )
            mnli_data.extend(mnli_instance)
            stats.append(line["relation"] != args.negative_tag)

    with open(output_file, "wt", encoding='utf-8') as f:
        for data in mnli_data:
            f.write(f"{json.dumps(data.__dict__)}\n")

    count = Counter([data.label for data in mnli_data])
    pprint(count)
    count = Counter(stats)
    pprint(count)


if __name__ == '__main__':
    main()
