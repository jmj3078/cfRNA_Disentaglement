import datetime
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import config

OPEN_TARGETS = "https://api.platform.opentargets.org/api/v4/graphql"
PAGE_SIZE = 1000

# Cancer phenotypes only (per project decision to focus this reference expansion on cancer
# cohorts). ICI-treated Cancer maps to None -- heterogeneous cohort, no single OT disease ID,
# excluded from analyses that require an OT reference gene set.
PHENO_QUERY = {
    'Colorectal Cancer (Chen)': 'colorectal carcinoma',
    'Esophagus Cancer (Chen)': 'esophageal carcinoma',
    'Liver Cancer (Chen)': 'hepatocellular carcinoma',
    'Liver Cancer (Roskams-Hieter)': 'hepatocellular carcinoma',
    'Lung Cancer (Chen)': 'lung carcinoma',
    'MGUS (Roskams-Hieter)': 'monoclonal gammopathy of undetermined significance',
    'MM (Roskams-Hieter)': 'multiple myeloma',
    'Pancreatic Cancer (Moore)': 'pancreatic carcinoma',
    'Stomach Cancer (Chen)': 'gastric carcinoma',
    'Other Cancer (Moore)': 'cancer',
    'ICI-treated Cancer (Raissadati)': None,
}


def gql(query, variables=None):
    body = json.dumps({'query': query, 'variables': variables or {}}).encode()
    req = urllib.request.Request(OPEN_TARGETS, data=body,
                                 headers={'Content-Type': 'application/json'})
    err = None
    for _ in range(4):
        try:
            return json.loads(urllib.request.urlopen(req, timeout=30).read())
        except Exception as e:
            err = e
            time.sleep(2)
    raise err


def resolve(q):
    d = gql('query($q:String!){search(queryString:$q,entityNames:["disease"],'
            'page:{index:0,size:1}){hits{id name}}}', {'q': q})
    h = d['data']['search']['hits']
    return (h[0]['id'], h[0]['name']) if h else (None, None)


def assoc_targets(efo, page_size=PAGE_SIZE):
    """Fetch ALL disease-target associations (not capped), paginating page_size at a time."""
    q = ('query($id:String!,$p:Pagination!){disease(efoId:$id){name '
         'associatedTargets(page:$p,orderByScore:"score"){count '
         'rows{target{approvedSymbol} score}}}}')
    genes, count, index = [], None, 0
    while True:
        d = gql(q, {'id': efo, 'p': {'index': index, 'size': page_size}})
        dd = d['data']['disease']
        if dd is None:
            return None, []
        at = dd['associatedTargets']
        count = at['count']
        rows = at['rows']
        genes.extend((r['target']['approvedSymbol'], round(r['score'], 4)) for r in rows)
        if len(rows) < page_size or len(genes) >= count:
            break
        index += 1
        time.sleep(0.2)
    return count, genes


def main():
    outdir = config.BENCHMARK_DIR / 'disease_reference'
    outdir.mkdir(parents=True, exist_ok=True)
    today = datetime.date.today().isoformat()
    for ph, q in PHENO_QUERY.items():
        stem = ph.replace('/', '_')
        if q is None:
            rec = {'phenotype': ph, 'query': None, 'efo': None, 'ot_disease': None,
                   'n_assoc': 0, 'genes': [], 'note': 'literature-only (heterogeneous)',
                   'source': 'Open Targets Platform GraphQL v4', 'retrieved': today}
        else:
            efo, name = resolve(q)
            cnt, genes = assoc_targets(efo) if efo else (0, [])
            rec = {'phenotype': ph, 'query': q, 'efo': efo, 'ot_disease': name,
                   'n_assoc': cnt, 'genes': genes,
                   'source': 'Open Targets Platform GraphQL v4', 'retrieved': today}
        json.dump(rec, open(outdir / f'{stem}.json', 'w'), indent=1)
        print(f'{ph:32s} {str(rec["efo"]):16s} n_genes={len(rec["genes"])}')
        time.sleep(0.3)
    print(f'\nsaved {len(PHENO_QUERY)} reference files to {outdir}')


if __name__ == '__main__':
    main()
