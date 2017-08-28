"""Given a two strings, heuristically determine if string 1 is in string 2"""

import re
def find(gene, sent):
    trans = {
        'Peroxisome proliferator-activated receptor-gamma/NFkB': 'Peroxisome proliferator-activated receptor-gamma',
        'NEUROG3/E47': 'NEUROG3',
        'Oct4/Gata-6': 'Oct4-Gata-6',
        'Beta2/NeuoD1': 'Beta2/NeuroD1',
        'TCF/Smad4': 'Smad4',
        'Smad3/c-myc?': 'Smad3',
        'PAX3/FKHR': 'PAX3-FKHR',
        'MLL1': 'MLL',
        'YY1/N1IC': 'N1IC'
    }
    g = list(re.finditer(re.escape(gene), sent))
    if g == []:
        g = list(re.finditer(re.escape(gene.lower()), sent.lower()))
    if g == []:
        trans_gene = trans.get(gene, gene)
        g = list(re.finditer(re.escape(trans_gene), sent))
    return g