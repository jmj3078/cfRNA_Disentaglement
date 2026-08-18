import difflib
import re


def strip_reactome_code(name):
    return name.split(" R-HSA-")[0]


def match_pathway_index(name, terms):
    stripped = [strip_reactome_code(t) for t in terms]

    # derive candidate forms: raw heading, text before an em-dash/hyphen tier-note suffix, and any
    # parenthetical content (agents sometimes wrote "Short Name (Actual Pathway Term)")
    candidates = [name]
    candidates.append(re.split(r'\s+[-—]{1,2}\s+', name)[0])
    paren = re.search(r'\(([^)]+)\)', name)
    if paren:
        candidates.append(paren.group(1))
        candidates.append(re.sub(r'\s*\([^)]+\)', "", name).strip())

    for cand in candidates:
        low = cand.lower().strip()
        if not low:
            continue
        for j, t in enumerate(stripped):
            if t.lower() == low:
                return j
        for j, t in enumerate(stripped):
            if t.lower().startswith(low) or low.startswith(t.lower()):
                return j
        close = difflib.get_close_matches(cand, stripped, n=1, cutoff=0.8)
        if close:
            return stripped.index(close[0])
    return None
