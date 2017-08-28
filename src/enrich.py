import sys, csv, tqdm
import xml.etree.ElementTree as ET
import gene_find

def build_ulf_xml_dict(ulf_xml_path):
    """Read ULF XML and create a dict of (sentence, gene1, gene2) -> (gene1 offset, gene2 offset)"""
    root = ET.parse(ulf_xml_path).getroot()
    offset_dict = {}
    counter = 0
    for docs in root.iter('document'):
        for sent in docs.iter('sentence'):
            entities = {el.get('id'): (el.get('text'), el.get('charOffset')) for el in sent.findall('entity')}
            for pair in sent.findall('pair'):
                offset_dict[(
                    sent.get('text').strip(),
                    entities[pair.get('e1')][0],
                    entities[pair.get('e2')][0]
                )] = (entities[pair.get('e1')][1], entities[pair.get('e2')][1])
                if entities[pair.get('e1')][1] is None:
                    print(sent.get('text').strip())
    return offset_dict

def enrich(rows, ulf):
    ulf = build_ulf_xml_dict(ulf)
    count = 0
    for row in rows:
        row['gene1_char_start'], row['gene1_char_end'], row['gene2_char_start'], row['gene2_char_end'] = None, None, None, None
        gene1s = gene_find.find(row['gene1'], row['sentence'])
        gene2s = gene_find.find(row['gene2'], row['sentence'])

        if len(gene1s) == 1 and len(gene2s) == 1:
            row['gene1_char_start'], row['gene1_char_end'] = [int(i) for i in gene1s[0].span()]
            row['gene2_char_start'], row['gene2_char_end'] = [int(i) for i in gene2s[0].span()]
        else:  # lookup the triple (sentence, gene1, gene2) in ULF XML
            triple = (row['sentence'], row['gene1'], row['gene2'])
            try:
                [gene1_char_start, gene1_char_end], [gene2_char_start, gene2_char_end] = \
                    [pos.split('-') for pos in ulf[triple]]
                row['gene1_char_start'], row['gene1_char_end'] = int(gene1_char_start), int(gene1_char_end)+1
                row['gene2_char_start'], row['gene2_char_end'] = int(gene2_char_start), int(gene2_char_end)+1
            except KeyError:
                count += 1
    # print(count)
    return rows


supplement_fp = sys.argv[1]
ulf_fp = sys.argv[2]


with open(supplement_fp) as f:
    reader = csv.DictReader(f)
    d = list(reader)


enriched = enrich(d, ulf=ulf_fp)
with open(supplement_fp + '.enriched', 'w') as f:
    writer = csv.DictWriter(f, fieldnames=enriched[0].keys())
    writer.writeheader()
    for row in enriched:
        writer.writerow(row)

with open(supplement_fp + '.enriched.filtered', 'w') as f:
    writer = csv.DictWriter(f, fieldnames=enriched[0].keys())
    writer.writeheader()
    for row in (r for r in enriched if r['gene1_char_start'] is not None and r['gene2_char_start'] is not None):
        writer.writerow(row)
