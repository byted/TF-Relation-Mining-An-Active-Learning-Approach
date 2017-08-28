import logging, json, sys, re
from tqdm import tqdm
from copy import deepcopy
from collections import Counter
from unidecode import unidecode

import GeneRegParser
import Top2500Parser
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from bllipparser import RerankingParser

def untokenize(tokens):
    string = ''
    for t in tokens:
        string += t if t in ',.;:?!' or t.startswith('-') or t.startswith('_') else ' ' + t
    return string.strip()

def replace_all(string, translation_dict):
    for _from, _to in translation_dict.items():
        string = string.replace(_from, _to)
    return string

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN  # might be a bad solution

def to_token_dict(token_tuple):
    return [{'text': text, 'pos': pos} for text, pos in token_tuple]


def match_tokens_to_original_sentence(sentence, tokens):
    '''
    Annotate tokens with their starting position in the original sentence.
    A new list of tokens will be produced so it's not inplace!
    '''

    special_chars = {'-LRB-': '(', '-RRB-': ')', '-LSB-': '[', '-RSB-': ']', '``': '"', "''": '"'}
    cur_char_idx, cur_token_idx = 0, 0
    new_tokens = []
    while cur_token_idx < len(tokens):
        cur_token_text = replace_all(tokens[cur_token_idx]['text'], special_chars)
        if sentence[cur_char_idx:].startswith(cur_token_text):
            # consume our token
            new_token = {k:v for k,v in tokens[cur_token_idx].items()}
            new_token['text'] = cur_token_text
            new_token['start'] = cur_char_idx
            new_tokens.append(new_token)
            cur_char_idx += len(cur_token_text)
            cur_token_idx += 1
        elif sentence[cur_char_idx:cur_char_idx+1] in ' \t':
            # we have a whitespace that got lost during tokenization
            cur_char_idx += 1
        else:
            pass

    return new_tokens

def tag_tokens_with_annotations(tokens_with_pos, annotations):
    '''
    Given annotations, tag and modify tokens based on their position in the original sentence.
    tokens_with_pos should be a dict of the form {'start': start pos in sentence, 'text': token's text}
    '''

    def get_tokens_touched_by_annotation(ann_start, ann_length, tokens_with_pos):
        token_candidates = []
        for idx, t in enumerate(tokens_with_pos):
            positions_of_token = set(range(t['start'], t['start']+len(t['text'])))
            positions_of_entity = set(range(ann_start, ann_start+ann_length))
            if len(positions_of_token & positions_of_entity) > 0:
                token_candidates.append((idx, t))
        return token_candidates

    for ann in annotations:
        ann_start, ann_length = ann['location']['char_idx'], ann['location']['length']
        token_candidates = get_tokens_touched_by_annotation(ann_start, ann_length, tokens_with_pos)
        token_idxs = token_candidates[0][0], token_candidates[-1][0]  # indizes of tokens touched
        # merge if multiple tokens are touched
        # merged_token_text = ''.join([t['text'] for _, t in token_candidates])
        merged_token = {
            'start': token_candidates[0][1]['start'],
            'text': untokenize([t['text'] for _, t in token_candidates]),
            'pos': token_candidates[0][1]['pos'],  # heuristic: take the first one
            'entity_text': ann['text'],
            'entity_type': ann['type'].upper(),
            'entity_id': ann['id']
        }

        # handle special cases where an entity does not align with the token
        prefix_token, suffix_token = None, None

        prefix_diff = ann_start - merged_token['start']
        if prefix_diff > 0:
            # cut off prefix and put in new token
            prefix_token = {'start': merged_token['start'],
                            'text': merged_token['text'][:prefix_diff],
                            'pos': merged_token['pos']}
            merged_token['text'] = merged_token['text'][prefix_diff:]
            merged_token['start'] = merged_token['start'] + prefix_diff

        suffix_diff = (merged_token['start']+len(merged_token['text'])) - (ann_start+ann_length)
        if suffix_diff > 0 :
            # cut off suffix and put in new token
            suffix_token = {'start': ann_start+ann_length,
                            'text': merged_token['text'][-suffix_diff:],
                            'pos': merged_token['pos']}
            merged_token['text'] = merged_token['text'][:-suffix_diff]
        new_tokens = [t for t in [prefix_token, merged_token, suffix_token] if t is not None]
        
        # build modified token list
        tokens_with_pos = tokens_with_pos[:token_idxs[0]] + new_tokens + tokens_with_pos[token_idxs[1]+1:]

    return tokens_with_pos

def serialize_tuple(tup):
    # id&&token&&lemma&&pos&&entity&&type
    return '{}&&{}&&{}&&{}&&{}&&{}'.format(*tup)

def serialize_tuples(tups):
    return ' '.join([serialize_tuple(t) for t in tups])

def serialize_line(label, _id, tups):
    # label \t id \t BODY \n
    return '{}\t{}\t{}\n'.format(label, _id, serialize_tuples(tups))

def get_annotation_range(ann):
    return range(ann['location']['char_idx'], ann['location']['char_idx'] + ann['location']['length'])

def does_overlap(range1, range2):
    return len(set(range1).intersection(set(range2))) > 0

if __name__ == '__main__':
    TAGGED_TOKEN_CACHE_FILE_PREFIX = 'tagged_token_cache'

    modes = ['genereg', 'top2500']
    if len(sys.argv) < 3 or sys.argv[1] not in modes:
        print('mode not specified! Usage: python3 {} [{}] src_file'.format(sys.argv[0], '|'.join(modes)))
        sys.exit()
    mode = sys.argv[1]
    src_file = sys.argv[2]
    bidirectional = False
    annotate_all_entities = True

    # TODO refactor cmdl parameter handling

    cache_file = '{}-{}-{}'.format(TAGGED_TOKEN_CACHE_FILE_PREFIX, mode, src_file.split('/')[-1])
    print('{} mode selected'.format(mode))
    print('parse "{}"'.format(src_file))

    # setup tools and data
    if mode == 'genereg':
        print('Load Genereg corpus...')
        corpus = GeneRegParser.CorpusParser(src_file)
        sents_with_annotations = corpus.get_sentences_with_annotations(genes_only=True)
    elif mode == 'top2500':
        print('Load Top2500 corpus...')
        corpus = Top2500Parser.CorpusParser(src_file)
        sents_with_annotations = corpus.get_sentences_with_annotations()

    print('...done')
    print('Load lemmatizer...')
    lemmatizer = WordNetLemmatizer()
    print('...done')

    debugf = open('debug', 'w')

    assert len(sents_with_annotations) == len(corpus.get_all_sentences())
    # load tokens from cache file or run a new tagging pass
    try:
        with open(cache_file, 'r') as ttcf:
            tokens = json.loads(ttcf.read())
        print('use tagger caching file "{}"'.format(cache_file))
    except FileNotFoundError:
        print('Load RerankingParser...')
        rrp = RerankingParser.from_unified_model_dir('../resources/McClosky-2009/biomodel')
        print('...done')
        tokens = [to_token_dict(rrp.tag(item['sentence'])) for item in tqdm(sents_with_annotations)]
        with open(cache_file, 'w') as ttcf:
            ttcf.write(json.dumps(tokens, indent=2))
            print('wrote tagging result to "{}"'.format(cache_file))
    
    try:
        assert len(sents_with_annotations) == len(tokens)
    except AssertionError:
        print('The number of instances does not match the number cached token instances - \
            most likely the tagged token cache file is outdated. Remove it to re-tag')
        sys.exit()

    print('Align tokens & tag entities...')
    for i in tqdm(range(len(tokens))):
        metadata = sents_with_annotations[i]
        tokens[i] = match_tokens_to_original_sentence(metadata['sentence'], tokens[i])
        # tokens[i] = tag_tokens_with_annotations(tokens[i], metadata['annotations'])

    print('Lemmatizing...')
    for token in tqdm(sum(tokens, [])):
        token['lemma'] = lemmatizer.lemmatize(token['text'], get_wordnet_pos(token['pos']))

    assert len(sents_with_annotations) == len(tokens)
    assert all(['lemma' in t for t in sum(tokens, [])])

    print('expand sentences to instances...')
    instances = []
    pairs_cnt = 0
    not_found_annos = []
    affected_instances = 0
    ignore_counter = 0
    less_than_2_anns_counter = 0
    for tokens, metadata in tqdm(zip(tokens, sents_with_annotations), total=len(tokens)):
        anns = metadata['annotations']
        rels = [(rel['nodes']['cause'], rel['nodes']['theme']) for rel in metadata['relations']]
        if len(anns) < 2:
            less_than_2_anns_counter +=1
            print(json.dumps(metadata, indent=2))

        pairs = []
        for i in range(len(anns)-1):
            for j in range(i+1, len(anns)):
                pairs.append((anns[i], anns[j]))

        pairs_cnt += len(pairs)

        for idx, (src, target) in enumerate(pairs):
            src_range = get_annotation_range(src)
            target_range = get_annotation_range(target)
            if does_overlap(src_range, target_range):
                # src and target overlap... ignore
                ignore_counter += 1
                continue
            
            # handle overlapping annotations - focus on the ones in question
            instance_anns = [src, target]
            if annotate_all_entities:
                for a in anns:
                    a_range = get_annotation_range(a)
                    if does_overlap(a_range, src_range) or does_overlap(a_range, target_range):
                        continue
                    instance_anns.append(a)

            instance_tokens = tag_tokens_with_annotations(deepcopy(tokens), instance_anns)

            instance_id = '{}-{}'.format(metadata['id'], idx)
            src_token, target_token = None, None
            for token in instance_tokens:
                if 'entity_id' in token:
                    if token['entity_id'] == src['id']:
                        src_token = token
                    elif token['entity_id'] == target['id']:
                        target_token = token
                    else:
                        token['entity_label'] = 'O'

            if not src_token:
                not_found_annos.append(src)
            if not target_token:
                not_found_annos.append(target)
            if not src_token or not target_token:
                affected_instances += 1
                continue

            if mode == 'top2500':
                label = metadata['simple_class']
            elif mode == 'genereg': 
                label = 1 if (src['id'], target['id']) in rels or (target['id'], src['id']) in rels else 0

            src_token['entity_label'] = 'A'
            target_token['entity_label'] = 'T'

            # if bidirectional:
            #     set_directed_entity_labels(src_token, target_token)

            #     if label == 1:
            #         src_token['entity_label'] = 'A' if src['is_source'] else 'T'
            #         target_token['entity_label'] = 'A' if target['is_source'] else 'T'
            #         assert(src_token['entity_label'] != target_token['entity_label'])
            #         ## TODO: add reversed thingies as label 0
            #     else:
            #         # src_token['entity_label'], target_token['entity_label'] = 'T', 'A'
            #         # instances.append((label, instance_id+'-reverse', deepcopy(instance_tokens)))
            #         src_token['entity_label'], target_token['entity_label'] = 'A', 'T'

            # elif mode == 'genereg': 
            #     if (src['id'], target['id']) in rels else 0

            #     :
            #         src_token['entity_label'], target_token['entity_label'] = 'A', 'T'
            #         label = 1
            #     elif (target['id'], src['id']) in rels:
            #         src_token['entity_label'], target_token['entity_label'] = 'T', 'A'
            #         label = 1
            #     else:
            #         # src_token['entity_label'], target_token['entity_label'] = 'T', 'A'
            #         label = 0
            #         # instances.append((label, instance_id+'-reverse', deepcopy(instance_tokens)))
            #         src_token['entity_label'], target_token['entity_label'] = 'A', 'T'
            # else:
            #     print('can\'t find a label for ' + json.dumps(metadata, indent=2))
            #     sys.exit()
            instances.append((label, instance_id, instance_tokens))


    print('convert instances to jSRE format & write to file...')
    outf = open('{}.preprocess_output'.format(src_file), 'w', encoding='latin-1')
    for label, instance_id, tokens in tqdm(instances):
        for idx, token in enumerate(tokens):
            if 'entity_label' not in token:
                token['entity_label'] = 'O'
            if 'entity_type' not in token:
                token['entity_type'] = 'O'
        tokens = [(
                idx,
                token['text'].replace(' ', '_'),
                token['lemma'] if 'lemma' in token else token['text'].replace(' ', '_'),
                token['pos'],
                token['entity_type'],
                token['entity_label']
            ) for idx, token in enumerate(tokens)]
        outf.write(unidecode(serialize_line(label, instance_id, tokens)))
        debugf.write(json.dumps(tokens, indent=2))
    print('ignore_counter:', ignore_counter)
    print('affected_instances:', affected_instances)
    print('less_than_2_anns_counter:', less_than_2_anns_counter)
    outf.close()
    debugf.close()
    if mode == 'genereg':
        (_, neg), (_, pos) = Counter([lbl for lbl, _, _ in instances]).most_common()
        print('wrote {} instances ({} negative, {} positive)'.format(len(instances), neg, pos))
        print(affected_instances)
        print(not_found_annos)