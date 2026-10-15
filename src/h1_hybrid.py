"""CLAUSESHIFT H1 final-safe hybrid inference.

Inputs are id/text_a/text_b only. No gold or DEV exports are read.
Runs baseline for every document, then fixed H1 Apertus calls, then complete export.
Zero successful Apertus edits is a hard failure for the H1 submission method.
"""
from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path
import time
import unicodedata
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from . import cli
from .core import LANGS, load_jsonl, validate_input_rows, valid_prediction

VERSION = 'clauseshift-h1-frozen-v115'
UPSTREAM_SHA = '1807a42100e742ed03d337c54c4b9ea86995f565'
HF_SHA = 'c5955035435e2bf121cde7f3c8863ef52ff35d82'
ENCODER = 'jhu-clsp/mmBERT-base'
MODEL = 'swiss-ai/apertus-v1.5-8b'
ENDPOINT = 'https://api.publicai.co/v1'
N_PER_LANG = 18
WIN = 32
REQUEST_PER_SIDE = 12
BLEND = 0.35
MAX_CALLS = 54
MAX_SUBTOKENS = 3072

class ContractError(ValueError):
    pass

def digest(x):
    return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def atomic(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    with temp.open('w',encoding='utf-8') as f:
        json.dump(obj,f,ensure_ascii=False,sort_keys=True,allow_nan=False)
        f.flush();os.fsync(f.fileno())
    temp.replace(path)

def check_row(row):
    for side in 'ab':
        txt=row[f'text_{side}'].split()
        vals=row[f'labels_{side}']
        if not isinstance(vals,list) or len(vals)!=len(txt):
            raise ContractError('Wrong baseline label length')
        for token,v in zip(txt,vals):
            if type(v) not in (int,float) or not math.isfinite(v) or not -1<=v<=1:
                raise ContractError('Nonfinite/out of range baseline score')
            if v==-1 and not punctuation(token):
                raise ContractError('Semantic token incorrectly masked')
    valid_prediction(row,row)

def punctuation(token):
    return bool(token) and all(unicodedata.category(c).startswith('P') for c in token)

def checkpoint(root,phase,sig,lang,doc):
    return Path(root)/'checkpoints'/'h1_final'/hashlib.sha256(sig.encode()).hexdigest()[:16]/phase/lang/(doc+'.json')

def inputs(input_dir,split):
    files={}
    for lang in LANGS:
        rows=list(load_jsonl(cli.path_in(input_dir,split,lang)))
        validate_input_rows(rows)
        for row in rows:
            if set(row)!= {'id','text_a','text_b'}:
                raise ContractError('Input must contain only id/text fields')
            if not row['id'].startswith('admin_'+lang+'_'):
                raise ContractError('Invalid language document id')
        files[lang]=rows
    return files

def load_baseline(root,sig,lang,row):
    p=checkpoint(root,'baseline',sig,lang,row['id'])
    if not p.exists():return None
    cp=json.loads(p.read_text(encoding='utf-8'))
    if cp.get('version')!=VERSION or cp.get('signature')!=sig or cp.get('text_hash')!=digest([row['text_a'],row['text_b']]):
        raise ContractError('Baseline checkpoint identity/signature mismatch')
    pred=cp.get('prediction')
    if not isinstance(pred,dict):raise ContractError('Malformed baseline checkpoint')
    valid_prediction(pred,row)
    check_row(pred)
    return pred

def predict_all(root,sig,files,model_factory=None,max_subtokens=MAX_SUBTOKENS):
    pending=[(lang,row) for lang in LANGS for row in files[lang] if load_baseline(root,sig,lang,row) is None]
    if pending:
        if model_factory is None:
            import torch
            from huggingface_hub import snapshot_download
            from rsd.recognizers import DiffAlign
            torch.set_num_threads(min(8,os.cpu_count() or 1))
            # Pin revision and use only pinned snapshot; no floating `main`.
            snapshot=snapshot_download(repo_id=ENCODER,revision=HF_SHA,local_files_only=True)
            model=DiffAlign(snapshot,layer=-1)
        else:model=model_factory()
        tokenizer=model.pipeline.tokenizer
        for lang,row in pending:
            sizes=[len(tokenizer(row[f'text_{s}'],add_special_tokens=True,truncation=False)['input_ids']) for s in 'ab']
            if max(sizes)>max_subtokens:
                raise ContractError(f'{lang}/{row["id"]}: {sizes} subtokens exceed frozen cap {max_subtokens}')
            start=time.monotonic()
            prediction=model.predict(a=row['text_a'],b=row['text_b'])
            pred={'id':row['id'],'text_a':row['text_a'],'text_b':row['text_b']}
            for side in 'ab':
                got=tuple(getattr(prediction,'tokens_'+side))
                expected=tuple(row['text_'+side].split())
                values=getattr(prediction,'labels_'+side)
                if got!=expected or len(values)!=len(expected):
                    raise ContractError('DiffAlign token coverage/order mismatch')
                pred['labels_'+side]=[float(v) for v in values]
            valid_prediction(pred,row);check_row(pred)
            atomic(checkpoint(root,'baseline',sig,lang,row['id']),
                   {'version':VERSION,'signature':sig,'text_hash':digest([row['text_a'],row['text_b']]),
                    'prediction':pred})
            print(f'[H1 BASELINE SAVED] {lang}/{row["id"]} elapsed={time.monotonic()-start:.2f}s',flush=True)
    return {lang:[load_baseline(root,sig,lang,row) for row in files[lang]] for lang in LANGS}

def select(rows):
    if len(rows)<N_PER_LANG:raise ContractError('Fewer than 18 input documents in language')
    ordered=sorted(rows,key=lambda r:(len(r['text_a'].split())+len(r['text_b'].split()),r['id']))
    places=sorted({round(i*(len(ordered)-1)/(N_PER_LANG-1)) for i in range(N_PER_LANG)})
    if len(places)!=N_PER_LANG:raise ContractError('Selection collision')
    return [ordered[i] for i in places]

def req(row,side,center):
    tokens=row['text_'+side].split();scores=row['labels_'+side]
    st=min(max(0,center-WIN//2),max(0,len(tokens)-WIN));en=min(st+WIN,len(tokens))
    ranked=sorted(((i,scores[i]) for i in range(st,en)
                   if scores[i]>=0 and not punctuation(tokens[i])),key=lambda x:(-x[1],x[0]))[:REQUEST_PER_SIDE]
    return {'start':st,'end':en,'want':sorted(i-st for i,_ in ranked)}

def windows(row):
    toks={s:row['text_'+s].split() for s in 'ab'}
    choices=[(row['labels_'+s][i],s,i) for s in 'ab' for i,t in enumerate(toks[s])
             if row['labels_'+s][i]>=0 and not punctuation(t)]
    if not choices:
        return {s:{'start':0,'end':min(WIN,len(toks[s])),'want':[]} for s in 'ab'}
    _,side,idx=sorted(choices,key=lambda v:(-v[0],v[1],v[2]))[0]
    other='b' if side=='a' else 'a'
    scaled=round(idx/max(len(toks[side])-1,1)*max(len(toks[other])-1,0))
    return {s:req(row,s,(idx if s==side else scaled)) for s in 'ab'}

def plan(root,sig,baselines):
    chosen={lang:select(baselines[lang]) for lang in LANGS}
    entries=[]
    for i in range(N_PER_LANG):
        for lang in LANGS:
            row=chosen[lang][i]
            entries.append({'lang':lang,'id':row['id'],
             'text_hash':digest([row['text_a'],row['text_b']]),
             'scores_hash':digest([row['labels_a'],row['labels_b']]),
             'spec':windows(row)})
    value={'version':VERSION,'signature':sig,'entries':entries}
    path=Path(root)/'plans'/'h1_final'/hashlib.sha256(sig.encode()).hexdigest()[:16]/'plan.json'
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8'))!=value:
            raise ContractError('Frozen H1 plan drift')
    else:atomic(path,value)
    return entries

def prompt(row,spec):
    payload={}
    for side,name in (('a','english'),('b','other_language')):
        part=row['text_'+side].split()[spec[side]['start']:spec[side]['end']]
        payload[name]=[{'i':i,'token':token} for i,token in enumerate(part)]
        payload['requested_'+side]=spec[side]['want']
    return ('You are checking translation equivalence in English and another language. '
        'For EACH requested token index, rate semantic SIMILARITY of that token '
        'against the meaning of the OTHER passage, including an omitted or added fact. '
        '0=unmatched or contradictory; 1=weak; 2=partial; 3=related; '
        '4=close; 5=fully equivalent. Compare meaning, not spelling. '
        'Both excerpts are LOCAL windows; do not assume their sentence boundaries align exactly. '
        'Score only requested indices, local to each window, including neither unrequested indices '
        'nor punctuation. Return EXACTLY ONE JSON object with keys a and b; '
        'each value maps requested decimal indices to numeric scores 0..5. '
        'Do not omit any requested index. Empty side must be {}. '
        'Do not repeat the input. Do not wrap in Markdown. '
        'Format example: {"a":{"0":5},"b":{"1":5}}. '
        'Payload:\n'+json.dumps(payload,ensure_ascii=False,separators=(',',':'))+
        '\nReturn JSON only:')

def parse(raw,spec,row):
    def unique(pairs):
        obj={}
        for k,v in pairs:
            if k in obj:raise ContractError('Duplicate JSON key')
            obj[k]=v
        return obj
    try:obj=json.loads(raw.strip(),object_pairs_hook=unique)
    except (json.JSONDecodeError,AttributeError,TypeError) as exc:raise ContractError('Invalid JSON') from exc
    if type(obj)!=dict or set(obj)!= {'a','b'}:raise ContractError('Wrong JSON envelope')
    output={}
    for side in 'ab':
        mapping=obj[side];want={str(i) for i in spec[side]['want']}
        if type(mapping)!=dict or set(mapping)!=want:raise ContractError('Missing/extra requested scores')
        output[side]={}
        for k,v in mapping.items():
            if type(v) not in (int,float) or not math.isfinite(v) or not 0<=v<=5:
                raise ContractError('Invalid similarity score')
            idx=spec[side]['start']+int(k)
            if row['labels_'+side][idx]<0 or punctuation(row['text_'+side].split()[idx]):
                raise ContractError('Overriding masked/punctuation token')
            output[side][str(idx)]=float(v)
    return output

def apply(row,corrected):
    pred={k:row[k] for k in ('id','text_a','text_b')}
    for side in 'ab':pred['labels_'+side]=list(row['labels_'+side])
    if corrected:
        for side in 'ab':
            for k,v in corrected[side].items():
                i=int(k);old=pred['labels_'+side][i]
                if not 0<=old<=1:raise ContractError('Cannot correct punctuation')
                pred['labels_'+side][i]=(1-BLEND)*old+BLEND*(1-v/5)
    valid_prediction(pred,row);check_row(pred)
    return pred

def call_api(message,key,timeout=52,max_tokens=600):
    body={'model':MODEL,'messages':[{'role':'user','content':message}],
          'temperature':0,'max_tokens':max_tokens,'stream':False}
    req=Request(ENDPOINT+'/chat/completions',data=json.dumps(body).encode(),
        headers={'Authorization':'Bearer '+key,'Content-Type':'application/json','User-Agent':'CLAUSESHIFT-H1/FROZEN'},method='POST')
    try:
        with urlopen(req,timeout=timeout) as r:result=json.load(r)
    except HTTPError as exc:raise ContractError(f'Provider HTTP {exc.code}') from exc
    except (URLError,TimeoutError,OSError,json.JSONDecodeError) as exc:raise ContractError('Provider transport/JSON error') from exc
    try:return result['choices'][0]['message']['content']
    except (KeyError,IndexError,TypeError) as exc:raise ContractError('Provider envelope invalid') from exc

def get_h1_cp(root,sig,entry,row):
    p=checkpoint(root,'apertus',sig,entry['lang'],entry['id'])
    if not p.exists():return None
    cp=json.loads(p.read_text(encoding='utf-8'))
    if cp.get('signature')!=sig or cp.get('id')!=entry['id'] or cp.get('text_hash')!=entry['text_hash'] or cp.get('scores_hash')!=entry['scores_hash']:
        raise ContractError('Apertus checkpoint mismatch')
    if cp.get('status') not in ('started','success','error'):raise ContractError('Invalid checkpoint status')
    if cp['status']=='success':
        answer=cp.get('corrections')
        if type(answer)!=dict or set(answer)!= {'a','b'}:raise ContractError('Invalid saved correction')
        mapping={side:{str(int(k)-entry['spec'][side]['start']):v for k,v in answer[side].items()} for side in 'ab'}
        if parse(json.dumps(mapping),entry['spec'],row)!=answer:
            raise ContractError('Saved correction tampered')
    return cp

def adjudicate(root,sig,entries,baselines,api=call_api,max_calls=MAX_CALLS):
    if max_calls<0 or max_calls>MAX_CALLS:raise ContractError('Invalid call budget')
    if os.getenv('LLM_NAME')!=MODEL or os.getenv('LLM_BASE_URL','').rstrip('/')!=ENDPOINT:
        raise ContractError('Provider endpoint/model must match frozen config')
    key=os.getenv('LLM_API_KEY','')
    if not key or any(ord(c)<32 or ord(c)==127 for c in key) or key!=key.strip():
        raise ContractError('Malformed or absent LLM_API_KEY')
    rowmap={lang:{r['id']:r for r in baselines[lang]} for lang in LANGS}
    calls=0;accepted=0;failed=0
    for e in entries:
        row=rowmap[e['lang']][e['id']]
        if get_h1_cp(root,sig,e,row) is not None:continue
        if calls>=max_calls:break
        p=checkpoint(root,'apertus',sig,e['lang'],e['id'])
        base={'signature':sig,'id':e['id'],'lang':e['lang'],
              'text_hash':e['text_hash'],'scores_hash':e['scores_hash']}
        if not e['spec']['a']['want'] and not e['spec']['b']['want']:
            atomic(p,{**base,'status':'error','error':'no semantic tokens'})
            failed+=1;continue
        atomic(p,{**base,'status':'started'})  # reserve BEFORE network: never double-pay
        calls+=1;t0=time.monotonic()
        try:
            corrections=parse(api(prompt(row,e['spec']),key),e['spec'],row)
            atomic(p,{**base,'status':'success','corrections':corrections})
            accepted+=1
            print(f'[H1 APERTUS SAVED] {e["lang"]}/{e["id"]} tokens={sum(len(v) for v in corrections.values())} elapsed={time.monotonic()-t0:.2f}s',flush=True)
        except (ContractError,ValueError,RuntimeError) as exc:
            atomic(p,{**base,'status':'error','error':type(exc).__name__})
            failed+=1
            print(f'[H1 APERTUS FALLBACK] {e["lang"]}/{e["id"]} {type(exc).__name__}',flush=True)
    print(f'[H1 CALLS] new={calls} success={accepted} fail={failed}',flush=True)
    return calls

def finalize(root,sig,files,baselines,entries,output_dir,*,allow_partial=False):
    rowmap={lang:{r['id']:r for r in baselines[lang]} for lang in LANGS}
    corrections={}
    incomplete=0; successes=0;changed=0
    for e in entries:
        row=rowmap[e['lang']][e['id']]
        cp=get_h1_cp(root,sig,e,row)
        if cp is None:incomplete+=1;continue
        if cp['status']=='success':
            successes+=1;corrections[(e['lang'],e['id'])]=cp['corrections']
    if incomplete and not allow_partial:raise ContractError(f'H1 incomplete: {incomplete} pending calls; no exports')
    if not successes:raise ContractError('No Apertus success: refusing to represent baseline as H1')
    results={}
    for lang in LANGS:
        rows=[]
        for inp,base in zip(files[lang],baselines[lang]):
            pred=apply(base,corrections.get((lang,inp['id'])))
            if any(pred['labels_'+s]!=base['labels_'+s] for s in 'ab'):changed+=1
            valid_prediction(pred,inp)
            rows.append(pred)
        results[lang]=rows
    if not changed:raise ContractError('No actual Apertus effect on submitted scores')
    if allow_partial:
        return results
    # Validate all languages before writing any outputs.
    for lang in LANGS:
        p=cli.path_out(output_dir,lang)
        if p.exists():p.unlink()
    for lang in LANGS:
        p=cli.path_out(output_dir,lang)
        cli.write_jsonl(p,results[lang]);print(f'[H1 EXPORTED] {lang} docs={len(results[lang])} file={p}',flush=True)
    return results

def run_dataset(split,input_dir,output_dir,checkpoint_root,sig,*,model_factory=None,api=call_api,max_paid_calls=MAX_CALLS,max_subtokens=MAX_SUBTOKENS):
    if split not in ('dev','final'):raise ContractError('Unknown split')
    if split=='final' and not sig.startswith('final:'):
        raise ContractError('Final H1 only after frozen signed authorization')
    if split=='dev' and not sig.startswith('dev:'):
        raise ContractError('DEV signature required')
    files=inputs(input_dir,split)
    baselines=predict_all(checkpoint_root,sig,files,model_factory,max_subtokens)
    entries=plan(checkpoint_root,sig,baselines)
    adjudicate(checkpoint_root,sig,entries,baselines,api=api,max_calls=max_paid_calls)
    return finalize(checkpoint_root,sig,files,baselines,entries,output_dir)
