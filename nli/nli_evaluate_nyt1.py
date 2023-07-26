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
    parser.add_argument("--run_name", type=str, default='nyt1')
    parser.add_argument("--input_file", type=str, default='../dataset/nyt1/dev_test.json')

    parser.add_argument("--input_type", type=str, default='test', help='train or test, if train, then confidence will be appended')
    parser.add_argument("--output_dir", type=str, default='../dataset/nyt1', help='useful only if input_type is train')
    parser.add_argument("--template_type", type=str, default='genr_patt', help='genr/genr_patt')

    parser.add_argument("--pretrained_model", type=str, default="microsoft/deberta-v2-xlarge-mnli")
    # parser.add_argument("--pretrained_model", type=str, default="../tmp/nli/nyt1", help='path to saved fine-tuned model')
    parser.add_argument("--negative_threshold", type=float, default=0.95)
    parser.add_argument("--negative_tag", type=str, default="None")
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()

    logger.info('Arguments:')
    for arg in vars(args):
        logger.info('    {}: {}'.format(arg, getattr(args, arg)))

    labels = [
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

    if args.template_type == 'genr':
        template_mapping = {
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
    elif args.template_type == 'genr_patt':
        template_mapping = {
            "/location/location/contains": [
                "{obj} is located in {subj}",
                "{subj} , including {obj}",
            ],
            "/people/person/nationality": [
                "{obj} is the nationality of {subj}",
                "{subj} , president of {obj}",
                "{obj} , Prime Minister {subj}",
                "{obj} 's foreign minister , {subj}",
                "{obj} 's former prime minister , {subj}",
                "{obj} President {subj}",
                "{obj} 's acting prime minister , {subj}",
            ],
            "/location/country/capital": [
                "{obj} is the capital of {subj}",
            ],
            "/people/person/place_lived": [
                "{subj} lives in {obj}",
                "{subj} , Republican from {obj}",
                "{subj} , Democrat of {obj}",
                "{obj} Gov. {subj}",
                "{obj} mayor , {subj}",
                "{subj} , the former senator from {obj}",
                "{subj} , who was mayor when {obj}",
                "{obj} Coach {subj}",
                "{obj} State Coach {subj}",
            ],
            "/business/person/company": [
                "{subj} works in {obj}",
                "{subj} , the head of {obj}",
                "{obj} 's chairman , {subj}",
                "{obj} 's president , {subj}",
                "{subj} chief executive , {obj}",
                "{obj} secretary general , {subj}",
                "{subj} , a professor at {obj}",
                "{subj} , founder of {obj}",
                "{obj} newsman {subj}",
                "{subj} , the co-founder of {obj}",
            ],
            "/location/neighborhood/neighborhood_of": [
                "{subj} is in the neighborhood of {obj}",
            ],
            "/people/person/place_of_birth": [
                "{subj} was born in {obj}",
            ],
            "/people/deceased_person/place_of_death": [
                "{subj} died in {obj}",
            ],
            "/business/company/founders": [
                "{subj} was founded by {obj}",
                "{subj} co-founder {obj}",
            ],
            "/people/person/children": [
                "{subj} is the parent of {obj}",
            ]
        }
    else:
        raise Exception('wrong template type')

    valid_conditions = {
        "/location/location/contains": [
            "LOCATION:LOCATION",
            "LOCATION:ORGANIZATION"
        ],
        "/people/person/nationality": [
            "PERSON:LOCATION"
        ],
        "/location/country/capital": [
            "LOCATION:LOCATION"
        ],
        "/people/person/place_lived": [
            "PERSON:LOCATION"
        ],
        "/business/person/company": [
            "PERSON:ORGANIZATION"
        ],
        "/location/neighborhood/neighborhood_of": [
            "LOCATION:LOCATION"
        ],
        "/people/person/place_of_birth": [
            "PERSON:LOCATION"
        ],
        "/people/deceased_person/place_of_death": [
            "PERSON:LOCATION"
        ],
        "/business/company/founders": [
            "ORGANIZATION:PERSON"
        ],
        "/people/person/children": [
            "PERSON:PERSON"
        ]
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
    print(label2id)

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
        pre, rec, f1, supt = precision_recall_fscore_support(labels_, output_, labels=list(range(1, len(labels))))
        print('precision: ', str(pre))
        print('recall: ', str(rec))
        print('f1: ', str(f1))
        print('supt: ', str(supt))

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
