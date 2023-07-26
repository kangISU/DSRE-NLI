import argparse
import json
import os

import numpy as np

from sklearn.metrics import precision_recall_fscore_support

import nli_base
import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default='tacrev-s')
    parser.add_argument("--input_file", type=str, default='../dataset/tacrev-s/test_rev.json')

    parser.add_argument("--input_type", type=str, default='test', help='train or test, if train, then confidence will be appended')
    parser.add_argument("--output_dir", type=str, default='../dataset/tacrev-s', help='useful only if input_type is train')
    parser.add_argument("--template_type", type=str, default='genr_patt', help='genr/genr_patt')

    parser.add_argument("--pretrained_model", type=str, default="microsoft/deberta-v2-xlarge-mnli")
    # parser.add_argument("--pretrained_model", type=str, default="../tmp/nli/tacrev-s", help='path to saved fine-tuned model')
    parser.add_argument("--negative_threshold", type=float, default=0.95)
    parser.add_argument("--negative_tag", type=str, default="no_relation")
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()

    logger.info('Arguments:')
    for arg in vars(args):
        logger.info('    {}: {}'.format(arg, getattr(args, arg)))

    labels = [
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

    if args.template_type == 'genr':
        template_mapping = {
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
    elif args.template_type == 'genr_patt':
        template_mapping = {
            "per:alternate_names": [
                "{subj} is also known as {obj}",
                "{obj} , whose real name is {subj}",
            ],
            "per:date_of_birth": [
                "{subj} was born in {obj}",
                "{subj} was born on {obj}",
            ],
            "per:age": [
                "{subj} is {obj} years old",
                "{subj} dies at {obj}",
            ],
            "per:country_of_birth": [
                "{subj} was born in {obj}",
                "{obj} Undersecretary of State {subj}",
            ],
            "per:stateorprovince_of_birth": ["{subj} was born in {obj}"],
            "per:city_of_birth": ["{subj} was born in {obj}"],
            "per:origin": [
                "{obj} is the nationality of {subj}",
                "{obj} Prime Minister {subj}",
                "{obj} actress {subj}",
                "{obj} president {subj}",
                "{obj} Rep. {subj}",
                "{obj} Undersecretary of State {subj}",
            ],
            "per:date_of_death": [
                "{subj} died in {obj}",
                "{subj} passed away on {obj}",
            ],
            "per:country_of_death": ["{subj} died in {obj}"],
            "per:stateorprovince_of_death": ["{subj} died in {obj}"],
            "per:city_of_death": ["{subj} died in {obj}"],
            "per:cause_of_death": [
                "{obj} is the cause of {subj} 's death",
                "{subj} dies after fight with {obj}",
            ],
            "per:countries_of_residence": ["{subj} lives in {obj}"],
            "per:stateorprovinces_of_residence": [
                "{subj} lives in {obj}",
                "{obj} Attorney General {subj}",
                "{obj} Sen. {subj}",
                "{obj} Rep. {subj}",
            ],
            "per:cities_of_residence": [
                "{subj} lives in {obj}",
                "{obj} district attorney {subj}",
            ],
            "per:schools_attended": [
                "{subj} studied in {obj}",
                "{subj} attended {obj}",
                "{subj} graduated from {obj}",
            ],
            "per:title": [
                "{subj} is a {obj}",
                "{subj} has served as {obj}",
            ],
            "per:employee_of": [
                "{subj} is an employee of {obj}",
                "{obj} spokesman {subj}",
                "{obj} president {subj}",
                "{obj} founder {subj}",
                "{obj} leader {subj}",
                "{subj} , president , {obj}",
                "{obj} chief executive {subj}",
                "{subj} , founder of the {obj}",
                "{obj} Director {subj}",
            ],
            "per:religion": ["{obj} is the religion of {subj}"],
            "per:parents": [
                "{obj} is the parent of {subj}"],
            "per:spouse": [
                "{subj} is the spouse of {obj}",
                "{subj} 's wife , {obj}",
                "{subj} 's husband , {obj}",
            ],
            "per:children": [
                "{subj} is the parent of {obj}",
                "{subj} sons {obj}",
            ],
            "per:siblings": ["{subj} and {obj} are siblings"],
            "per:other_family": ["{subj} and {obj} are family"],
            "per:charges": [
                "{obj} are the charges of {subj}",
                "{subj} convicted of {obj}",
            ],
            "org:alternate_names": ["{subj} is also known as {obj}"],
            "org:political/religious_affiliation": [
                "{subj} has an affiliation with {obj}",
                "{obj} group {subj}",
            ],
            "org:top_members/employees": [
                "{obj} is a high level member of {subj}",
                "{subj} head {obj}",
                "{subj} spokeswoman {obj}",
                "{obj} , executive director of {subj}",
                "{subj} president {obj}",
                "{obj} , director of {subj}",
                "{obj} , spokesman for the {subj}",
                "{subj} Vice Chairman {obj}",
                "{obj} , chief economist at {subj}",
                "{subj} manager {obj}",
            ],
            "org:number_of_employees/members": [
                "{subj} has about {obj} employees",
                "{subj} carrying {obj}",
                "{obj} members of the {subj}",
            ],
            "org:members": ["{obj} is member of {subj}"],
            "org:member_of": ["{subj} is member of {obj}"],
            "org:subsidiaries": [
                "{obj} is a subsidiary of {subj}",
                "{obj} , owned by {subj}",
                "{subj} agrees to buy {obj}",
            ],
            "org:parents": [
                "{subj} is a subsidiary of {obj}",
                "{obj} , which controls {subj}",
                "{obj} 's state-run {subj}",
                "{obj} moves to buy {subj}",
                "{subj} unit of {obj}",
            ],
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
        raise Exception('wrong template type')

    valid_conditions = {
        "per:alternate_names": ["PERSON:PERSON", "PERSON:MISC"],
        "per:date_of_birth": ["PERSON:DATE"],
        "per:age": ["PERSON:NUMBER", "PERSON:DURATION"],
        "per:country_of_birth": ["PERSON:COUNTRY"],
        "per:stateorprovince_of_birth": ["PERSON:STATE_OR_PROVINCE"],
        "per:city_of_birth": ["PERSON:CITY"],
        "per:origin": [
            "PERSON:NATIONALITY",
            "PERSON:COUNTRY",
            "PERSON:LOCATION",
        ],
        "per:date_of_death": ["PERSON:DATE"],
        "per:country_of_death": ["PERSON:COUNTRY"],
        "per:stateorprovince_of_death": ["PERSON:STATE_OR_PROVICE"],
        "per:city_of_death": ["PERSON:CITY", "PERSON:LOCATION"],
        "per:cause_of_death": ["PERSON:CAUSE_OF_DEATH"],
        "per:countries_of_residence": ["PERSON:COUNTRY", "PERSON:NATIONALITY"],
        "per:stateorprovinces_of_residence": ["PERSON:STATE_OR_PROVINCE"],
        "per:cities_of_residence": ["PERSON:CITY", "PERSON:LOCATION"],
        "per:schools_attended": ["PERSON:ORGANIZATION"],
        "per:title": ["PERSON:TITLE"],
        "per:employee_of": ["PERSON:ORGANIZATION"],
        "per:religion": ["PERSON:RELIGION"],
        "per:spouse": ["PERSON:PERSON"],
        "per:children": ["PERSON:PERSON"],
        "per:siblings": ["PERSON:PERSON"],
        "per:other_family": ["PERSON:PERSON"],
        "per:charges": ["PERSON:CRIMINAL_CHARGE"],
        "org:alternate_names": [
            "ORGANIZATION:ORGANIZATION",
            "ORGANIZATION:MISC",
        ],
        "org:political/religious_affiliation": [
            "ORGANIZATION:RELIGION",
            "ORGANIZATION:IDEOLOGY",
        ],
        "org:top_members/employees": ["ORGANIZATION:PERSON"],
        "org:number_of_employees/members": ["ORGANIZATION:NUMBER"],
        "org:members": ["ORGANIZATION:ORGANIZATION", "ORGANIZATION:COUNTRY"],
        "org:member_of": [
            "ORGANIZATION:ORGANIZATION",
            "ORGANIZATION:COUNTRY",
            "ORGANIZATION:LOCATION",
            "ORGANIZATION:STATE_OR_PROVINCE",
        ],
        "org:subsidiaries": [
            "ORGANIZATION:ORGANIZATION",
            "ORGANIZATION:LOCATION",
        ],
        "org:parents": ["ORGANIZATION:ORGANIZATION", "ORGANIZATION:COUNTRY"],
        "org:founded_by": ["ORGANIZATION:PERSON"],
        "org:founded": ["ORGANIZATION:DATE"],
        "org:dissolved": ["ORGANIZATION:DATE"],
        "org:country_of_headquarters": ["ORGANIZATION:COUNTRY"],
        "org:stateorprovince_of_headquarters": ["ORGANIZATION:STATE_OR_PROVINCE"],
        "org:city_of_headquarters": [
            "ORGANIZATION:CITY",
            "ORGANIZATION:LOCATION",
        ],
        "org:shareholders": [
            "ORGANIZATION:PERSON",
            "ORGANIZATION:ORGANIZATION",
        ],
        "org:website": ["ORGANIZATION:URL"],
    }

    clf = nli_base.NLIRelationClassifierWithMappingHead(
        labels=labels,
        template_mapping=template_mapping,
        pretrained_model=args.pretrained_model,
        valid_conditions=valid_conditions,
        negative_threshold=args.negative_threshold,
        negative_tag=args.negative_tag,
    )

    label2id = {label: i for i, label in enumerate(labels)}

    with open(args.input_file, "rt", encoding='utf-8') as f:
        features, labels_ = [], []
        for i, line in enumerate(json.load(f)):
            features.append(
                nli_base.REInputFeatures(
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
                )
            )
            labels_.append(label2id[line["relation"]])

    labels_ = np.array(labels_)
    output = clf(features, batch_size=args.batch_size)

    if args.input_type == 'test':
        output_ = output.argmax(-1)
        pre, rec, f1, _ = precision_recall_fscore_support(labels_, output_, average="micro", labels=list(range(1, len(labels))))
        print('Precision: {:.2f}, Recall: {:.2f}, F1: {:.2f}'.format(pre * 100, rec * 100, f1 * 100))

    elif args.input_type == 'train':
        with open(args.input_file, "r", encoding='utf-8') as f:
            data = json.load(f)

        output_file = os.path.join(args.output_dir, 'train_' + args.template_type + '.json')
        new_data = []
        with open(output_file, 'w', encoding='utf-8') as f:
            for prob, inst in zip(output, data):
                prob_label = sorted(list(zip(prob, labels)), reverse=True)[:1][0]
                inst['mnli_conf'] = round(prob_label[0], 4)
                inst['mnli_pred_relation'] = prob_label[1]
                new_data.append(inst)
            json.dump(new_data, f)

    else:
        raise Exception('Wrong input type')


if __name__ == '__main__':
    main()
