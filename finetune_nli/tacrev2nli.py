from argparse import ArgumentParser
from dataclasses import dataclass
from collections import defaultdict, Counter
from typing import Dict
import numpy as np
import json
import os
from pprint import pprint
import random

TACRED_LABELS = [
    "no_relation",
    "org:alternate_names",
    "org:city_of_headquarters",
    "org:country_of_headquarters",
    "org:dissolved",
    "org:founded",
    "org:founded_by",
    "org:member_of",
    "org:members",
    "org:number_of_employees/members",
    "org:parents",
    "org:political/religious_affiliation",
    "org:shareholders",
    "org:stateorprovince_of_headquarters",
    "org:subsidiaries",
    "org:top_members/employees",
    "org:website",
    "per:age",
    "per:alternate_names",
    "per:cause_of_death",
    "per:charges",
    "per:children",
    "per:cities_of_residence",
    "per:city_of_birth",
    "per:city_of_death",
    "per:countries_of_residence",
    "per:country_of_birth",
    "per:country_of_death",
    "per:date_of_birth",
    "per:date_of_death",
    "per:employee_of",
    "per:origin",
    "per:other_family",
    "per:parents",
    "per:religion",
    "per:schools_attended",
    "per:siblings",
    "per:spouse",
    "per:stateorprovince_of_birth",
    "per:stateorprovince_of_death",
    "per:stateorprovinces_of_residence",
    "per:title",
]

TACRED_LABEL_TEMPLATES = {
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


def tacrev2nli(
        instance: REInputFeatures,
        positive_templates,
        negative_templates,
        templates,
        labels2id,
        negn=1,
        posn=1,
        negt='no_relation'
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


def tacrev2nli_with_negative_pattern(
        instance: REInputFeatures,
        positive_templates,
        negative_templates,
        templates,
        labels2id,
        negn=1,
        posn=1,
        negt='no_relation'
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

    parser.add_argument("--dataset", type=str, default="tacrev-s")
    parser.add_argument("--input", type=str, default="train.json")
    parser.add_argument("--output", type=str, default="train_reformulated.json")
    parser.add_argument("--negative_pattern", action="store_true", default=False)
    parser.add_argument("--negative_tag", type=str, default='no_relation')
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
        "{subj} is also known as {obj}",
        "{subj} was born in {obj}",
        "{subj} is {obj} years old",
        "{obj} is the nationality of {subj}",
        "{subj} died in {obj}",
        "{obj} is the cause of {subj}’s death",
        "{subj} lives in {obj}",
        "{subj} studied in {obj}",
        "{subj} is a {obj}",
        "{subj} is an employee of {obj}",
        "{subj} believe in {obj}",
        "{subj} is the spouse of {obj}",
        "{subj} is the parent of {obj}",
        "{obj} is the parent of {subj}",
        "{subj} and {obj} are siblings",
        "{subj} and {obj} are family",
        "{subj} was convicted of {obj}",
        "{subj} has political affiliation with {obj}",
        "{obj} is a high level member of {subj}",
        "{subj} has about {obj} employees",
        "{obj} is member of {subj}",
        "{subj} is member of {obj}",
        "{obj} is a branch of {subj}",
        "{subj} is a branch of {obj}",
        "{subj} was founded by {obj}",
        "{subj} was founded in {obj}",
        "{subj} existed until {obj}",
        "{subj} has its headquarters in {obj}",
        "{obj} holds shares in {subj}",
        "{obj} is the website of {subj}",
    ]

    labels2id = {"entailment": 2, "neutral": 1, "contradiction": 0}

    positive_templates: Dict[str, list] = defaultdict(list)
    negative_templates: Dict[str, list] = defaultdict(list)

    if not args.negative_pattern:
        templates = templates[1:]

    for label in TACRED_LABELS:
        if not args.negative_pattern and label == args.negative_tag:
            continue
        for template in templates:
            if label != args.negative_tag and template == "{subj} and {obj} are not related":
                continue
            if template in TACRED_LABEL_TEMPLATES[label]:
                positive_templates[label].append(template)
            else:
                negative_templates[label].append(template)

    tacrev2nli_ = tacrev2nli_with_negative_pattern if args.negative_pattern else tacrev2nli

    with open(input_file, "rt", encoding='utf-8') as f:
        mnli_data = []
        stats = []
        for line in json.load(f):
            mnli_instance = tacrev2nli_(
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
