"""Build web/languages.json: the languages the page's language picker offers.

Each entry is [iso639-3, iso639-1 or "", name, whisper code or "", mms]:
the language under the name a linguist would search for, the two codes that
recognisers ask for, the code Whisper uses for it (not always its 639-1 code:
Javanese is "jw", Hawaiian and Cantonese have none), and whether MMS has an
adapter for it. A language is listed if any recogniser in Lana can use the
code, which is every 639-1 language plus every MMS language.

usage: build.py <SIL iso-639-3.tab> <mms_languages.json> <out.json>
The SIL table: https://iso639-3.sil.org/code_tables/download_tables
"""
import csv, json, sys

# Whisper's language tokens (openai/whisper tokenizer.py, LANGUAGES), keyed by
# the ISO 639-3 code of the language each stands for.
WHISPER = {
 'eng':'en','zho':'zh','deu':'de','spa':'es','rus':'ru','kor':'ko','fra':'fr','jpn':'ja','por':'pt','tur':'tr',
 'pol':'pl','cat':'ca','nld':'nl','ara':'ar','swe':'sv','ita':'it','ind':'id','hin':'hi','fin':'fi','vie':'vi',
 'heb':'he','ukr':'uk','ell':'el','msa':'ms','ces':'cs','ron':'ro','dan':'da','hun':'hu','tam':'ta','nor':'no',
 'tha':'th','urd':'ur','hrv':'hr','bul':'bg','lit':'lt','lat':'la','mri':'mi','mal':'ml','cym':'cy','slk':'sk',
 'tel':'te','fas':'fa','lav':'lv','ben':'bn','srp':'sr','aze':'az','slv':'sl','kan':'kn','est':'et','mkd':'mk',
 'bre':'br','eus':'eu','isl':'is','hye':'hy','nep':'ne','mon':'mn','bos':'bs','kaz':'kk','sqi':'sq','swa':'sw',
 'glg':'gl','mar':'mr','pan':'pa','sin':'si','khm':'km','sna':'sn','yor':'yo','som':'so','afr':'af','oci':'oc',
 'kat':'ka','bel':'be','tgk':'tg','snd':'sd','guj':'gu','amh':'am','yid':'yi','lao':'lo','uzb':'uz','fao':'fo',
 'hat':'ht','pus':'ps','tuk':'tk','nno':'nn','mlt':'mt','san':'sa','ltz':'lb','mya':'my','bod':'bo','tgl':'tl',
 'mlg':'mg','asm':'as','tat':'tt','haw':'haw','lin':'ln','hau':'ha','bak':'ba','jav':'jw','sun':'su','yue':'yue',
}

tab, mms_path, out = sys.argv[1:4]
sil = {r['Id']: r for r in csv.DictReader(open(tab, encoding='utf-8'), delimiter='\t')}
# MMS names are the friendlier ones ("Avar (Avaric)"); a code with several
# adapters (one per script or dialect) is one language here.
mms = {}
for code, name in json.load(open(mms_path, encoding='utf-8')):
    base = code.split('-')[0]
    if base not in mms or '-' not in code:
        mms[base] = name if '-' not in code else sil.get(base, {}).get('Ref_Name', name)
codes = set(mms) | {k for k, r in sil.items() if r['Part1']} | set(WHISPER)
rows = []
for c in sorted(codes):
    r = sil.get(c)
    name = mms.get(c) or (r['Ref_Name'] if r else None)
    if not name:
        continue
    rows.append([c, r['Part1'] if r else '', name, WHISPER.get(c, ''), 1 if c in mms else 0])
rows.sort(key=lambda x: x[2].lower())
json.dump(rows, open(out, 'w', encoding='utf-8'), ensure_ascii=False, separators=(',', ':'))
print(len(rows), 'languages;', sum(1 for r in rows if r[3]), 'Whisper;', sum(r[4] for r in rows), 'MMS')
