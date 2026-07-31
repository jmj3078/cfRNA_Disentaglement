import datetime
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

OPEN_TARGETS = "https://api.platform.opentargets.org/api/v4/graphql"
GWAS_CATALOG = "https://www.ebi.ac.uk/gwas/rest/api"
PAGE_SIZE = 1000
SUPPLEMENT_MIN_GENES = 100  # below this, top up with GWAS Catalog gene-trait hits

# phenotype (post-OOD sample-level label, MixedEffectsModeling/3_disease_scoring.ipynb) -> Open
# Targets disease search query. Heterogeneous / undefined-cohort phenotypes map to None
# (no OT gene reference; literature-only).
PHENO_QUERY = {
    'CAD_HF+': 'coronary artery disease',
    'CAD_HF-': 'coronary artery disease',
    'Colorectal Cancer': 'colorectal carcinoma',
    'Esophagus Cancer': 'esophageal carcinoma',
    'HIV': 'HIV infection',
    'HIV + Tuberculosis': 'tuberculosis',
    'ICI-m': 'myocarditis',
    'ICI-treated Cancer': None,
    'Liver Cancer': 'hepatocellular carcinoma',
    'Liver Cirrhosis': 'liver cirrhosis',
    'Lung Cancer': 'lung carcinoma',
    'ME/CFS': 'chronic fatigue syndrome',
    'MGUS': 'monoclonal gammopathy of undetermined significance',
    'MM': 'multiple myeloma',
    'Other Cancer': 'cancer',
    'Pancreatic Cancer': 'pancreatic carcinoma',
    'Pancreatitis': 'pancreatitis',
    'Pre-eclampsia': 'pre-eclampsia',
    'Stomach Cancer': 'gastric carcinoma',
    'Tuberculosis': 'tuberculosis',
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


def http_json(url):
    req = urllib.request.Request(url, headers={'Accept': 'application/json'})
    err = None
    for _ in range(4):
        try:
            return json.loads(urllib.request.urlopen(req, timeout=30).read())
        except Exception as e:
            err = e
            time.sleep(2)
    raise err


def gwas_trait_search(query):
    """findByEfoTrait needs an exact trait-name match. Progressively drop trailing words
    ("X of undetermined significance" -> "X") until GWAS Catalog's own vocabulary matches."""
    words = query.split()
    while words:
        q = urllib.parse.quote(' '.join(words))
        hits = http_json(f'{GWAS_CATALOG}/efoTraits/search/findByEfoTrait?trait={q}&page=0&size=1')
        traits = hits.get('_embedded', {}).get('efoTraits', [])
        if traits:
            return traits[0]['shortForm']
        words = words[:-1]
    return None


def gwas_supplement_genes(query, exclude):
    """Free, no-key fallback for OT-sparse diseases: genes reported by GWAS Catalog hits
    for the matching EFO trait (author-reported genes at each significant locus)."""
    short_form = gwas_trait_search(query)
    if short_form is None:
        return None, []
    assoc = http_json(f'{GWAS_CATALOG}/efoTraits/{short_form}/associations?projection=associationByEfoTrait')
    genes = set()
    for a in assoc.get('_embedded', {}).get('associations', []):
        for loc in a.get('loci', []):
            for ag in loc.get('authorReportedGenes', []):
                name = ag.get('geneName')
                if name and name not in exclude:
                    genes.add(name)
    return short_form, sorted(genes)


def main():
    outdir = Path(__file__).parent / 'disease_reference'
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
            if len(genes) < SUPPLEMENT_MIN_GENES:
                have = {g for g, _ in genes}
                gwas_trait, gwas_genes = gwas_supplement_genes(q, have)
                if gwas_genes:
                    rec['supplement'] = {'source': 'GWAS Catalog', 'trait_efo': gwas_trait,
                                         'genes': gwas_genes}
        json.dump(rec, open(outdir / f'{stem}.json', 'w'), indent=1)
        n_supp = len(rec.get('supplement', {}).get('genes', []))
        print(f'{ph:22s} {str(rec["efo"]):16s} n_genes={len(rec["genes"])}'
              + (f' +{n_supp} GWAS' if n_supp else ''))
        time.sleep(0.3)
    print(f'\nsaved {len(PHENO_QUERY)} reference files to {outdir}')


if __name__ == '__main__':
    main()
