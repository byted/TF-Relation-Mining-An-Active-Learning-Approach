import csv, json, re, sys
from collections import Counter
from random import choice
from string import ascii_uppercase
 
class CorpusParser():
    def __init__(self, filepath):
        '''Parse the result CSV from Thomas et al. 2014 paper'''
        with open(filepath, 'r') as csv_file:
            reader = csv.DictReader(csv_file)
            self.data = [{k.lower(): v for k, v in line.items()} for line in reader]
        if 'class' in self.data[0]:
            self.__simplify_classes()
        else:
            print('WARNING: dataset contains no class. Use Supplement2.csv for generation of simple labels.')

    def __simplify_classes(self):
        '''Simplify labels to 0/1 according to the description in the Thomas et al. 2014 paper'''
        for instance in self.data:
            if instance['class'] == 'TP':
                instance['simple_class'] = 1
            elif instance['class'] == 'FP':
                instance['simple_class'] = 0
            elif instance['class'] == 'NN':
                if instance['details'] == 'cooperation/competition in transcription':
                    instance['simple_class'] = 1
                else:
                    instance['simple_class'] = 0
            else:
                print('WARNING: unknown class in the dataset: {}'.format(instance['class']))                    

    def __describe(self, column, dataset):
        c = Counter()
        for line in dataset:
            c.update({line[column]})
        return c.most_common()

    def describe_labels(self, dataset=None):
        dataset = self.data if dataset is None else dataset
        return self.__describe('class', dataset)

    def describe_details(self, dataset=None):
        dataset = self.data if dataset is None else dataset
        return self.__describe('details', dataset)

    def filter(self, column, value):
        return [line for line in self.data if line[column] == value]

    def describe_filter(self, column, value):
        filtered_data = self.filter(column, value)
        return self.describe_labels(dataset=filtered_data)

    def get_sentences_with_annotations(self):
        def create_annotation_dict(gene, start_idx, end_idx, is_source=False):
            assert(len(gene) == end_idx - start_idx)
            location = {'char_idx': start_idx, 'length': len(gene), 'sentence_idx': None}
            return {'id': ''.join(choice(ascii_uppercase) for i in range(8)), 'location': location, 'type': 'Gene', 'text': gene, 'is_source': is_source}
        
        def create_annotation(gene_start, gene_end, sentence, **kwargs):
            gene_start, gene_end = int(gene_start), int(gene_end)
            gene_text = sentence[gene_start:gene_end]
            annos = [create_annotation_dict(gene_text.replace(' ', '_'), gene_start, gene_end, **kwargs)]
            return gene_text, annos

        sents = []
        for line_idx, line in enumerate(self.data):
            sent_dict = {'id': line_idx, 'sentence': line['sentence'], 'relations': []}
            gene1, gene2 = line['gene1'], line['gene2']

            # find gene in text
            gene1, annos1 = create_annotation(line['gene1_char_start'], line['gene1_char_end'], line['sentence'])
            gene2, annos2 = create_annotation(line['gene2_char_start'], line['gene2_char_end'], line['sentence'], is_source=True)
            sent_dict['annotations'] = sorted(annos1 + annos2, key=lambda i: i['location']['char_idx'])
            anns_to_remove = []
            for idx, ann in enumerate(sent_dict['annotations'][:-1]):
                # get rid of nested annotations
                next_ann = sent_dict['annotations'][idx+1]
                ann_start, ann_len = ann['location']['char_idx'],ann['location']['length']
                next_ann_start, next_ann_len = next_ann['location']['char_idx'], next_ann['location']['length']
                if ann_start in range(next_ann_start, next_ann_start+next_ann_len) and ann_len < next_ann_len:
                    anns_to_remove.append(ann['id'])
                elif next_ann_start in range(ann_start, ann_start+ann_len) and next_ann_len < ann_len:
                    anns_to_remove.append(next_ann['id'])
            sent_dict['annotations'] = [ann for ann in sent_dict['annotations'] if ann['id'] not in anns_to_remove]
            sent_dict['removed_annotations'] = [ann for ann in sent_dict['annotations'] if ann['id'] in anns_to_remove]
            sent_dict['simple_class'] = line['simple_class']

            for g in [gene1, gene2]:
                # if g not in sent_dict['sentence']:
                #     import ipdb; ipdb.set_trace()
                sent_dict['sentence'] = sent_dict['sentence'].replace(g, g.replace(' ', '_'))

            sents.append(sent_dict)
        return sents

    def get_all_sentences(self):
        return self.data


if __name__ == '__main__':
    try:
        filepath = '../resources/tf_extraction_paper_thomas/Supplement2.csv'
        corpus = CorpusParser(filepath)
    except FileNotFoundError:
        sys.exit()
    sents = corpus.get_sentences_with_annotations()

    for s in sents:
        for a in s['annotations']:
            start, length = a['location']['char_idx'], a['location']['length']
            assert s['sentence'][start:start+length] == a['text']

    assert len(sents) == 2500

    print('no errors')
