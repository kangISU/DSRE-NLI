import json
import os


def NPIN(input_file, output_file, neg_tag):
    with open(input_file, 'r', encoding='utf-8') as fp:
        data = json.load(fp)

    pos = 0
    neg = 0
    new_data = []
    for d in data:
        if d['mnli_pred_relation'] != neg_tag:
            d['revised_relation'] = d['mnli_pred_relation']
            new_data.append(d)
            pos += 1
        elif d['mnli_pred_relation'] == neg_tag and d['relation'] == neg_tag:
            d['revised_relation'] = neg_tag
            new_data.append(d)
            neg += 1

    print('pos: ', pos)
    print('neg: ', neg)

    with open(output_file, 'w', encoding='utf-8') as fp:
        json.dump(new_data, fp)


def IPIN(input_file, output_file, neg_tag):
    with open(input_file, 'r', encoding='utf-8') as fp:
        data = json.load(fp)

    pos = 0
    neg = 0
    new_data = []
    for d in data:
        if d['relation'] == d['mnli_pred_relation'] and d['relation'] == neg_tag:
            d['revised_relation'] = neg_tag
            new_data.append(d)
            neg += 1
        elif d['relation'] == d['mnli_pred_relation']:
            d['revised_relation'] = d['relation']
            new_data.append(d)
            pos += 1

    print('pos: ', pos)
    print('neg: ', neg)

    with open(output_file, 'w', encoding='utf-8') as fp:
        json.dump(new_data, fp)


def main():
    dataset_dir = '../dataset'
    dataset = 'nyt1'
    input_file = os.path.join(dataset_dir, dataset, 'train_genr_patt.json')
    neg_tag = 'None'

    output_file = os.path.join(dataset_dir, dataset, 'train_genr_patt_ipin.json')
    IPIN(input_file, output_file, neg_tag)

    output_file = os.path.join(dataset_dir, dataset, 'train_genr_patt_npin.json')
    NPIN(input_file, output_file, neg_tag)


if __name__ == '__main__':
    main()
