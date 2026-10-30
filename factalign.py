"""Fact-unit alignment checker for AI-text humanizer audits.

Extracts checkable units from a source text and decides, unit by unit, whether a rewrite keeps them.
Three groups:
  num    numbers and dates (digits, number words, dates; compared by value)
  ent    person (PERSON) and organization (ORG) names, recognized with spaCy en_core_web_sm 3.8.0
  quote  spans of at least 3 words inside quotation marks (compared verbatim after ignoring case,
         punctuation and whitespace)
Each unit: {group, key, surface, spans: [[start, end], ...], kept, evidence}.
Library use:  from factalign import align; units = align(source, rewrite)
Command line: python factalign.py --in pairs.jsonl --out units.jsonl  (each line needs "source" and "rewrite")
"""
import argparse, json, re, sys
from functools import lru_cache

VERSION = "v1"

MONTHS = {m: i + 1 for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"])}
MON_ABBR = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12}
WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
MONTH_RE = (r"(?:January|February|March|April|May|June|July|August|September|October|November|December|"
            r"Jan\.?|Feb\.?|Mar\.?|Apr\.?|Jun\.?|Jul\.?|Aug\.?|Sept?\.?|Oct\.?|Nov\.?|Dec\.?)")
UNITS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
         "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
         "seventeen": 17, "eighteen": 18, "nineteen": 19}
TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
SCALES = {"hundred": 100, "thousand": 1e3, "million": 1e6, "billion": 1e9, "trillion": 1e12}
SCALE_ABBR = {"k": 1e3, "m": 1e6, "mn": 1e6, "mln": 1e6, "bn": 1e9, "b": 1e9, "tn": 1e12, "t": 1e12}
ORDINALS = {"third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
            "eleventh": 11, "twelfth": 12, "thirteenth": 13, "fourteenth": 14, "fifteenth": 15, "sixteenth": 16,
            "seventeenth": 17, "eighteenth": 18, "nineteenth": 19, "twentieth": 20, "thirtieth": 30, "fortieth": 40,
            "fiftieth": 50, "hundredth": 100}
FIRST_SECOND_NOUNS = r"(?:quarter|half|time|year|place|round|phase|stage|term|day|week|month|annual|consecutive|anniversary)"
FRACTIONS = {"half": 0.5, "halves": 0.5, "third": 1 / 3, "thirds": 1 / 3, "quarter": 0.25, "quarters": 0.25,
             "fourth": 0.25, "fourths": 0.25, "fifth": 0.2, "fifths": 0.2, "tenth": 0.1, "tenths": 0.1}

NUMWORD = "(?:" + "|".join(sorted(list(UNITS) + list(TENS), key=len, reverse=True)) + ")"
WORDNUM_RE = re.compile(
    r"\b(?:" + NUMWORD + r"(?:[\s-]+" + NUMWORD + r")*(?:[\s-]+(?:hundred|thousand|million|billion|trillion)"
    r"(?:[\s-]+(?:and[\s-]+)?" + NUMWORD + r"(?:[\s-]+" + NUMWORD + r")*)?)*|a (?:hundred|thousand|million|billion|trillion)|a dozen|dozen)\b",
    re.I)
DIGIT_RE = re.compile(
    r"(?<![\w.,/])(?<![A-Za-z]-)(?P<cur>[$€£¥])?\s?(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+)"
    r"(?P<ord>st|nd|rd|th)?(?:\s?(?P<scale>hundred|thousand|million|billion|trillion|bn|mn|mln|tn|[kKmMbB](?![a-zA-Z])))?"
    r"(?P<pct>\s?(?:%|percent\b|per cent\b|percentage points?\b))?")
URL_RE = re.compile(r"https?://\S+|www\.\S+|\S+@\S+\.\w+")
QUOTE_RE = re.compile(r"[“\"]([^”\"]{3,}?)[”\"]")
GENERIC_ORG = {"the", "of", "and", "for", "on", "in", "at", "to", "a", "an", "department", "office", "national", "institute",
               "federal", "bank", "board", "committee", "agency", "administration", "university", "center", "centre",
               "council", "commission", "association", "corporation", "inc", "co", "company", "group", "foundation",
               "u.s.", "us", "united", "states", "america", "american", "service", "services", "bureau", "division",
               "program", "programme", "laboratory", "laboratories", "lab", "labs", "system", "systems", "society",
               "organization", "authority", "ministry", "fund", "trust", "college", "school", "academy", "network",
               "international", "global", "research", "science", "sciences", "technology", "technologies", "energy",
               "standards", "reserve", "open", "market", "public", "general", "state", "new"}
ROLE_RE = r"\b(?:Specialist|Manager|Director|Officer|Analyst|Engineer|Coordinator|Assistant|Position|Role|Intern|Associate|Lead)s?$"
NOT_ORG = {"DIY", "AI", "GDP", "CEO", "CFO", "CTO", "COVID", "COVID-19", "FAQ", "ROI", "KPI", "KPIS", "SEO", "PHD", "MBA",
           "CSA", "PLC", "WMS", "EV", "EVS", "HR", "IT", "PR", "UX", "UI", "API", "SAAS", "B2B", "LLM", "LLMS", "ESG", "STEM"}
ALIASES = {"federal reserve": ["fed"], "federal reserve board": ["fed", "federal reserve"],
           "board of governors of the federal reserve system": ["fed", "federal reserve", "federal reserve board"],
           "federal open market committee": ["fomc"], "department of energy": ["doe", "energy department"],
           "u.s. department of energy": ["doe", "energy department", "department of energy"],
           "national institute of standards and technology": ["nist"], "united nations": ["un", "u.n."],
           "european central bank": ["ecb"], "international monetary fund": ["imf"]}


def norm_text(s):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", s.lower().replace("’", "'"))).strip()


def words_to_num(s):
    s = s.lower().replace("-", " ")
    if s in ("a dozen", "dozen"):
        return 12.0
    toks = [t for t in s.split() if t != "and"]
    if toks and toks[0] == "a":
        toks[0] = "one"
    total, cur = 0.0, 0.0
    for t in toks:
        if t in UNITS:
            cur += UNITS[t]
        elif t in TENS:
            cur += TENS[t]
        elif t == "hundred":
            cur = (cur or 1) * 100
        elif t in SCALES:
            total += (cur or 1) * SCALES[t]
            cur = 0.0
        else:
            return None
    return total + cur


def vkey(v):
    return float(f"{v:.6g}")


def find_dates(text):
    """Return [(start, end, key)], where key is ('date', y, m, d), ('weekday', i), etc."""
    out = []
    pats = [
        (re.compile(r"\b(" + MONTH_RE + r")\s+(\d{1,2})(?:st|nd|rd|th)?(?:,?\s+(\d{4}))?\b"), "mdy"),
        (re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?(" + MONTH_RE + r")(?:,?\s+(\d{4}))?\b"), "dmy"),
        (re.compile(r"\b(" + MONTH_RE + r")(?:\s+of)?,?\s+(\d{4})\b"), "my"),
        (re.compile(r"\b(January|February|March|April|June|July|August|September|October|November|December)\b"), "m"),
        (re.compile(r"\b(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)s?\b"), "w"),
    ]
    taken = []

    def free(a, b):
        return all(b <= x or a >= y for x, y in taken)

    def mon(tok):
        t = tok.lower().rstrip(".")
        return MONTHS.get(t) or MON_ABBR.get(t)

    for rx, kind in pats:
        for m in rx.finditer(text):
            a, b = m.span()
            if not free(a, b):
                continue
            if kind == "mdy":
                key = ("date", m.group(3) and int(m.group(3)), mon(m.group(1)), int(m.group(2)))
            elif kind == "dmy":
                key = ("date", m.group(3) and int(m.group(3)), mon(m.group(2)), int(m.group(1)))
            elif kind == "my":
                key = ("date", int(m.group(2)), mon(m.group(1)), None)
            elif kind == "m":
                key = ("date", None, mon(m.group(1)), None)
            else:
                key = ("weekday", WEEKDAYS.index(m.group(1).lower()))
            taken.append((a, b))
            out.append((a, b, key))
    return out, taken


def find_numbers(text, taken):
    """Number units: return [(start, end, value)]; skip spans already taken by dates, URLs, list numbers at line start, and years (years are date units)."""
    out = []
    blocked = list(taken) + [m.span() for m in URL_RE.finditer(text)]

    def free(a, b):
        return all(b <= x or a >= y for x, y in blocked)

    for m in DIGIT_RE.finditer(text):
        a, b = m.span("num")
        if m.group("cur"):
            a = m.start("cur")
        b = m.end()
        if not free(m.start("num"), m.end("num")):
            continue
        line_start = text.rfind("\n", 0, m.start()) + 1
        if re.fullmatch(r"[\s*#>_-]*", text[line_start:m.start()]) and re.match(r"[.)]\s|[.)]\*|[.)]$", text[m.end("num"):m.end("num") + 2]):
            continue  # list numbering
        raw = m.group("num").replace(",", "")
        try:
            v = float(raw)
        except ValueError:
            continue
        sc = (m.group("scale") or "").lower()
        if sc in SCALES:
            v *= SCALES[sc]
        elif sc in SCALE_ABBR and (m.group("cur") or sc in ("bn", "mn", "mln", "tn")):
            v *= SCALE_ABBR[sc]
        elif sc:
            b = m.end("num") + (len(m.group("ord") or ""))
        if re.match(r"[-‑\s]?([1-7])/([2348])\b", text[m.end("num"):m.end("num") + 5]) and not m.group("pct") and not sc:
            fm = re.match(r"[-‑\s]?([1-7])/([2348])\b", text[m.end("num"):])
            if int(fm.group(1)) < int(fm.group(2)):  # Fed-style fraction: 3-1/2 = 3.5
                v += int(fm.group(1)) / int(fm.group(2))
                b = m.end("num") + fm.end()
                pm = re.match(r"\s?(?:%|percent\b|per cent\b|percentage points?\b)", text[b:])
                if pm:
                    b += pm.end()
                out.append((a, b, ("num", vkey(v), "pct" if pm else "")))
                blocked.append((a, b))
                continue
        is_year = (not m.group("cur") and not sc and not m.group("pct") and not m.group("ord")
                   and re.fullmatch(r"(1[89]\d\d|20\d\d)", m.group("num")) is not None)
        if is_year:
            out.append((a, b, ("date", int(raw), None, None)))
        else:
            out.append((a, b, ("num", vkey(v), "pct" if m.group("pct") else "")))
        blocked.append((a, b))
    for m in re.finditer(r"(?<![\w/])([1-7])/([2348])(?![\w/])", text):
        a, b = m.span()
        if free(a, b) and int(m.group(1)) < int(m.group(2)):
            out.append((a, b, ("num", vkey(int(m.group(1)) / int(m.group(2))), "")))
            blocked.append((a, b))
    for m in WORDNUM_RE.finditer(text):
        a, b = m.span()
        if not free(a, b):
            continue
        s = m.group(0)
        if s.lower() == "one":  # the pronoun "one" is not a quantity
            continue
        v = words_to_num(s)
        if v is None:
            continue
        tail = text[b:b + 40]
        mf = re.match(r"[\s-]+(halves|half|thirds?|quarters?|fourths?|fifths?|tenths?)\b", tail, re.I)
        if mf and v < 20:
            v = v * FRACTIONS[mf.group(1).lower()]
            b += mf.end()
        else:
            ms = re.match(r"\s+(percent|per cent|percentage points?)\b", tail, re.I)
            if ms:
                b += ms.end()
        out.append((a, b, ("num", vkey(v), "pct" if (not mf and ms) else "")))
        blocked.append((a, b))
    for m in re.finditer(r"\b(" + "|".join(ORDINALS) + r")\b", text, re.I):
        a, b = m.span()
        if free(a, b) and not re.match(r"[\s-]*(?:of\b)|,", text[b:b + 6], re.I):  # "Third," is a discourse marker
            out.append((a, b, ("num", float(ORDINALS[m.group(1).lower()]), "")))
            blocked.append((a, b))
    for m in re.finditer(r"\b(first|second)[\s-]+" + FIRST_SECOND_NOUNS + r"\b", text, re.I):
        a, b = m.start(1), m.end(1)
        if free(a, b):
            out.append((a, b, ("num", 1.0 if m.group(1).lower() == "first" else 2.0, "")))
            blocked.append((a, b))
    for m in re.finditer(r"\b(one|a)[\s-]+(half|third|quarter)\b", text, re.I):
        a, b = m.span()
        if free(a, b):
            out.append((a, b, ("num", vkey(FRACTIONS[m.group(2).lower()]), "")))
            blocked.append((a, b))
    return out


def num_inventory(text):
    """All number/date components in the rewrite, used to decide whether a unit is kept."""
    dates, taken = find_dates(text)
    nums = find_numbers(text, taken)
    vals, years, mds, months, wds = set(), set(), set(), set(), set()
    for _, _, k in dates + nums:
        if k[0] == "num":
            vals.add(k[1])
            if k[2] == "pct":
                vals.add(vkey(k[1] / 100))
        elif k[0] == "weekday":
            wds.add(k[1])
        else:
            _, y, mo, d = k
            if y:
                years.add(y)
                vals.add(float(y))
            if mo and d:
                mds.add((mo, d))
            if mo:
                months.add(mo)
            if d:
                vals.add(float(d))
    for m in re.finditer(r"\b(half|quarter)\b", text, re.I):  # lenient on the rewrite side: a bare half/quarter also counts
        vals.add(0.5 if m.group(1).lower() == "half" else 0.25)
    has_no = re.search(r"\b(no|none|zero|nothing|not a single|never)\b", text, re.I) is not None
    return {"vals": vals, "years": years, "mds": mds, "months": months, "wds": wds, "has_no": has_no}


def num_kept(key, inv):
    if key[0] == "num":
        if key[1] == 0.0 and inv["has_no"]:
            return True  # zero <-> no/none
        return key[1] in inv["vals"] or (key[2] == "pct" and vkey(key[1] / 100) in inv["vals"])
    if key[0] == "weekday":
        return key[1] in inv["wds"]
    _, y, mo, d = key
    ok = True
    if y:
        ok &= y in inv["years"]
    if mo and d:
        ok &= (mo, d) in inv["mds"]
    elif mo:
        ok &= mo in inv["months"]
    return ok


@lru_cache(maxsize=1)
def nlp():
    import spacy
    return spacy.load("en_core_web_sm")


def org_variants(name, source):
    n = re.sub(r"['’]s$", "", re.sub(r"^(?:the|a|an)\s+", "", name.strip(), flags=re.I)).strip()
    low = n.lower()
    vs = {norm_text(n)}
    bare = re.sub(r"^(?:U\.S\.|US|United States|America's)\s+", "", n)
    if bare != n and len(bare.split()) >= 2:
        vs.add(norm_text(bare))
        low = bare.lower()
    for k, al in ALIASES.items():
        if norm_text(low) == norm_text(k):
            vs.update(norm_text(x) for x in al)
    caps = [w for w in re.findall(r"[A-Za-z][\w.&'-]*", n)]
    init_all = "".join(w[0] for w in caps if w[0].isupper() or w.lower() == "of").upper()
    init_cap = "".join(w[0] for w in caps if w[0].isupper()).upper()
    for ac in {init_all, init_cap}:
        if len(ac) >= 2:
            vs.add(ac.lower())
    m = re.search(re.escape(n) + r"\s*\(([A-Z][A-Za-z.&-]{1,9})\)", source)
    if m:
        vs.add(m.group(1).lower())
    distinct = [w for w in caps if w.lower() not in GENERIC_ORG and (w[0].isupper() or any(c.isupper() for c in w[1:]))]
    return vs, distinct


def ent_units(source):
    doc = nlp()(source)
    units = {}
    for e in doc.ents:
        if e.label_ not in ("PERSON", "ORG"):
            continue
        if "\n" in e.text or not any(t.pos_ == "PROPN" for t in e):
            continue  # skip entities spanning lines or without a proper noun (sentence-initial imperatives, headings)
        if len(e) == 1 and e[0].is_sent_start and not e.text.isupper():
            continue  # single sentence-initial words (Aim, Invest, Foster) are mostly false positives
        line = source[source.rfind("\n", 0, e.start_char) + 1:e.start_char]
        if re.match(r"\s*#", line):
            continue  # Markdown heading line
        if e.label_ == "ORG" and (e.text.strip().upper() in NOT_ORG or (len(e) == 1 and e.text.isupper() and len(e.text) > 5)):
            continue  # common non-organization acronyms (DIY, COVID-19) and all-caps datelines (GAITHERSBURG)
        if e.label_ == "ORG" and re.search(ROLE_RE, e.text.strip()):
            continue  # job titles (Warehouse Automation Specialist) are not organizations
        text = e.text.strip().strip("*_#").strip()
        text = re.sub(r"[’']s$", "", text).strip()
        text = re.sub(r"^(?:the|The|a|A|an|An)\s+", "", text)
        if not re.search(r"[A-Z]", text) or len(text) < 2:
            continue
        toks = re.findall(r"[A-Za-z][\w.&'-]*", text)
        if e.label_ == "ORG" and all(t.lower() in GENERIC_ORG for t in toks) and not (len(toks) >= 2 or text.isupper()):
            continue  # a single generic word (e.g. "Energy") is not an organization name
        if e.label_ == "PERSON" and any(t.lower() in MONTHS for t in toks):
            continue
        key = ("ent", e.label_, norm_text(text))
        u = units.setdefault(key, {"group": "ent", "key": key, "surface": text, "label": e.label_, "spans": []})
        u["spans"].append([e.start_char, e.end_char])
    return list(units.values())


def ent_kept(u, source, rewrite):
    rw_norm = " " + norm_text(rewrite) + " "
    if u["label"] == "PERSON":
        toks = [t for t in re.findall(r"[A-Za-z][\w'-]*", u["surface"]) if t[0].isupper() and len(t) > 1]
        toks = [t for t in toks if t.lower() not in ("jr", "sr", "ii", "iii", "dr", "mr", "ms", "mrs")]
        if not toks:
            return True, "no-name-token"
        sur = toks[-1]
        if re.search(r"\b" + re.escape(sur) + r"\b", rewrite):
            return True, f"surname:{sur}"
        return False, f"surname-missing:{sur}"
    vs, distinct = org_variants(u["surface"], source)
    for v in vs:
        if v and (" " + v + " ") in rw_norm:
            return True, f"variant:{v}"
    for w in distinct:
        if re.search(r"\b" + re.escape(w) + r"\b", rewrite):
            return True, f"distinct:{w}"
    return False, "org-missing"


def quote_units(source):
    units = []
    for m in QUOTE_RE.finditer(source):
        q = m.group(1)
        if len(norm_text(q).split()) < 3:
            continue
        units.append({"group": "quote", "key": ("quote", norm_text(q)), "surface": q, "spans": [[m.start(1), m.end(1)]]})
    return units


def align(source, rewrite):
    dates, taken = find_dates(source)
    nums = find_numbers(source, taken)
    inv = num_inventory(rewrite)
    by = {}
    for a, b, k in dates + nums:
        u = by.setdefault(k, {"group": "num", "key": k, "surface": source[a:b], "spans": []})
        u["spans"].append([a, b])
    units = []
    # nested units: "October 3, 2025" and a separate "2025" in the same text are both kept (different keys)
    for k, u in by.items():
        u["kept"] = num_kept(k, inv)
        u["evidence"] = "value-found" if u["kept"] else "value-missing"
        units.append(u)
    for u in ent_units(source):
        u["kept"], u["evidence"] = ent_kept(u, source, rewrite)
        units.append(u)
    rw = " " + norm_text(rewrite) + " "
    rwq = " " + " ".join(norm_text(m.group(1)) for m in QUOTE_RE.finditer(rewrite)) + " "
    for u in quote_units(source):
        u["kept"] = (" " + u["key"][1] + " ") in rw or (" " + u["key"][1] + " ") in rwq
        u["evidence"] = "verbatim" if u["kept"] else "not-verbatim"
        units.append(u)
    for u in units:
        u["key"] = list(u["key"])
    return units


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    with open(a.out, "w", encoding="utf-8") as f:
        for l in open(a.inp, encoding="utf-8"):
            if not l.strip():
                continue
            r = json.loads(l)
            units = align(r["source"], r["rewrite"])
            f.write(json.dumps({"item_id": r.get("item_id") or r.get("id"), "version": VERSION, "units": units},
                               ensure_ascii=False) + "\n")


if __name__ == "__main__":
    sys.exit(main())
