import xml.etree.ElementTree as ET
from nltk import sent_tokenize

class Annotation():
    def __init__(self, annotation_xml_object):
        self.id = annotation_xml_object.attrib['id']
        infons = {a.attrib['key']: a.text for a in annotation_xml_object.findall('infon')}
        self.file, self.type = infons['file'], infons['type']
        self.location = {i[0]: int(i[1]) for i in annotation_xml_object.find('location').attrib.items()}
        self.text = annotation_xml_object.find('text').text
        self.encoded_text = None
    
    def __str__(self):
        return 'ID: {}\nLocation: {}\nText: {}'.format(self.id, self.location, self.text)
    
    def as_dict(self):
        return {'id': self.id, 'type': self.type, 'location': self.location, 'text': self.text}

class Relation():    
    def __init__(self, rel_xml_object):
        self.id = rel_xml_object.attrib['id']
        infons = {a.attrib['key']: a.text for a in rel_xml_object.findall('infon')}
        self.file, self.type = infons['file'], infons['type']
        try:
            self.relation_type = infons['relation type']
        except KeyError:
            self.relation_type = None
        self.nodes = {n.attrib['role'].lower(): n.attrib['refid'] for n in rel_xml_object.findall('node')}
    
    def __str__(self):
        return 'Relation {}\nType {}\nFrom {} to {}'.format(self.id, self.relation_type, *self.nodes.values())
    
    def as_dict(self):
        return {'id': self.id, 'type': self.type, 'nodes': self.nodes}

class Document():
    def __init__(self, document_xml_object, ignore_text=True):
        self.id = document_xml_object[0].text
        
        self.text = document_xml_object[1][1].text if not ignore_text else ''
        self.sentences = sent_tokenize(self.text)
        if self.id == '14731280':  # hack to fix a bad sentence split
            self.sentences = self.sentences[:3] + [self.sentences[3] + self.sentences[4]] + self.sentences[5:]
        
        self.annotations = {a.id: a for a in [Annotation(a) for a in document_xml_object[1].findall('annotation')]}
        self.__generate_sentence_based_locations_for_annotations()
        
        self.relations = {r.id: r for r in [Relation(r) for r in document_xml_object[1].findall('relation')]}
        self.__create_entity_tokens()
        self.__extract_annotation_ids_for_genes()
        self.__extract_gene_relations()
        
    def __str__(self):
        return 'ID: {}\nAnnotations: {}\nRelations: {}'.format(
            self._id, len(self.annotations), len(self.relations))
    
    def __extract_annotation_ids_for_genes(self):
        self.gene_ids = set([a.id for a in self.annotations.values() if a.type == 'Gene'])
    
    def __extract_gene_relations(self):
        self.gene_relations = {r.id: r for r in self.relations.values() if
            all([_id in self.gene_ids for _id in r.nodes.values()])}
    
    def __generate_sentence_based_locations_for_annotations(self):
        for ann in self.annotations.values():
            char_offset = ann.location['offset']
            for i, sent in enumerate(self.sentences):
                if char_offset - len(sent) < 0:
                    ann.location = {'sentence_idx': i, 'char_idx': char_offset, 'length': ann.location['length']}
                    break
                else:
                    char_offset -= len(sent) + 1  # subtract one for the space between 2 sentences
    
    def __create_entity_tokens(self):
        for ann in self.annotations.values():
            # ann.text = ann.text.replace(' ', '_')
            sent_id, start, length = ann.location['sentence_idx'], ann.location['char_idx'], ann.location['length']
            sent = self.sentences[sent_id]
            self.sentences[sent_id] = sent[:start] + sent[start:start+length] + sent[start+length:]
    
    def __rel2sentence(self, rels):
        sentences = []
        for rel in rels:
            src = self.annotations[rel.nodes['cause']]
            target = self.annotations[rel.nodes['theme']]
            src_sentence_id, target_sentence_id = [it.location['sentence_idx'] for it in [src, target]]
            
            try:
                assert src_sentence_id == target_sentence_id
            except AssertionError:
                pass
#                 print(self.id)
#                 print(src_sentence_id, target_sentence_id)
#                 print((src.id, src.text), (target.id, target.text))
#                 print(self.sentences[src_sentence_id])
#                 print()
#                 print(self.sentences[target_sentence_id])
#                 print('---')
            sentences.append({
                'sentence': self.sentences[src_sentence_id],
                'src': src.as_dict(), 'target': target.as_dict()
            })
        return sentences
    
    def get_sentences_with_gene_relation(self):
        return self.__rel2sentence(self.gene_relations.values())
    
    def get_sentences_without_gene_relation(self):
        non_gene_rels = [rel for rel_id, rel in self.relations.items() if rel_id not in self.gene_relations]
        return self.__rel2sentence(non_gene_rels)
    
    def _test_annotation_location_types(self):
        for ann in self.annotations.values():
            full_text_based = self.text[ann.location['offset']:ann.location['offset']+ann.location['length']]
            start, length = ann.location['char_idx'], ann.location['length']
            sent_based = self.sentences[ann.location['sentence_idx']][start:start+length]
            if not full_text_based == sent_based:
                print('-----')
                print('DocID: ' + self.id)
                print('AnnotationID: ' + ann.id)
                print('Should:\n' + ann.text)
                print('Full text:\n' + full_text_based)
                print('Sent based:\n' + sent_based)
                print('----')
    
class CorpusParser():
    def __init__(self, path):
        self.tree = ET.parse(path)
        self.root = self.tree.getroot()
        self.documents = [Document(d, ignore_text=False) for d in self.root if d.tag == 'document']
    
    def get_sentences_with_annotations(self, genes_only=False):
        sentences = []
        for d in self.documents:
            for sent_idx, sent in enumerate(d.sentences):
                sent_dict = {'sentence': sent, 'id': '{}-{}'.format(d.id, sent_idx)}
                sent_dict['annotations'] = [ann.as_dict() for ann in d.annotations.values()
                                            if ann.location['sentence_idx'] == sent_idx 
                                            and (not genes_only or ann.type == 'Gene')]
                sent_dict['annotations'] = sorted(sent_dict['annotations'], key=lambda a: a['location']['char_idx'])
                ann_ids = set([ann['id'] for ann in sent_dict['annotations']])
                sent_dict['relations'] = [rel.as_dict() for rel in d.gene_relations.values()
                                          if rel.nodes['cause'] in ann_ids]
                sentences.append(sent_dict)    
        return sentences
    
    def get_all_sentences_with_gene_relation(self):
        sentences = []
        for d in self.documents:
            sents = d.get_sentences_with_gene_relation()
            if sents is not None:
                sentences += sents
        return sentences

    def get_all_sentences_without_gene_relation(self):
        sentences = []
        for d in self.documents:
            sents = d.get_sentences_without_gene_relation()
            if sents is not None:
                sentences += sents
        return sentences

    def get_all_sentences(self):
        return sum([d.sentences for d in self.documents], [])
    
    def get_all_relations(self):
        return sum([list(d.relations.values()) for d in self.documents], [])
    
    def get_all_gene_relations(self):
        return sum([list(d.gene_relations.values()) for d in self.documents], [])