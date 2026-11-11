#!/usr/bin/env python3
"""Offline factual spot checks using the same loader and generation path as Chat.
Checks are transparent string checks, not a general factual-accuracy benchmark.
"""
import argparse
import importlib.util
import json
import re
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('studio_chat', ROOT / 'Backend/chat_runner.py')
chat = importlib.util.module_from_spec(spec)
spec.loader.exec_module(chat)


def passes(answer, case):
    normalized = answer.replace(' ', '').replace('AI', 'ai').lower()
    required = all(any(term.lower().replace(' ', '') in normalized for term in group) for group in case.get('requires', []))
    forbidden = any(term.lower().replace(' ', '') in normalized for term in case.get('forbids', []))
    minimum = case.get('at_least', {})
    enough = sum(term.lower().replace(' ', '') in normalized for term in minimum.get('terms', [])) >= minimum.get('count', 0)
    invented_contact = case.get('category') == '未知边界' and bool(re.search(r'\d{7,}', answer))
    return required and enough and not forbidden and not invented_contact


def evaluate(model, adapters, cases, output, max_tokens=128, greedy=False):
    import threading
    import mlx.core as mx
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler
    worker = chat.Worker()
    chat.emit = lambda kind, **values: None
    worker.load({'model':model, 'adapter':adapters or '', 'quantization':{'bits':4,'group_size':128,'mode':'affine'}})
    results=[]
    for case in cases:
        worker.reset_cache()
        events=[]
        chat.emit=lambda kind, **values: events.append(dict(type=kind,**values))
        started=time.monotonic()
        error=None
        try:
            messages=[{'role':'user','content':case['question']}]
            if greedy:
                prompt=worker.tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
                tokens=worker.tokenizer.encode(prompt,add_special_tokens=False)
                mx.random.seed(0)
                answer=''.join(r.text for r in stream_generate(worker.model,worker.tokenizer,tokens,max_tokens=max_tokens,sampler=make_sampler(temp=0),prefill_step_size=256))
            else:
                worker.generate({'messages':messages,'thinking':False,'max_tokens':max_tokens},threading.Event())
                answer=''.join(e.get('text','') for e in events if e['type']=='delta')
        except Exception as exc:
            answer='';error=str(exc)
        result=dict(id=case['id'],category=case['category'],question=case['question'],expected=case['expected'],answer=answer,
                    passed=passes(answer,case) and error is None,error=error,seconds=round(time.monotonic()-started,2))
        results.append(result)
        print(f"{case['id']}: {'PASS' if result['passed'] else 'FAIL'} {answer[:110]}",file=sys.stderr,flush=True)
    report=dict(model=model,adapter=adapters,sampling='greedy' if greedy else 'Chat defaults; seed 0; thinking off',max_tokens=max_tokens,
                passed=sum(r['passed'] for r in results),total=len(results),results=results,
                limitations='Small development question set; keyword checks can miss extra false claims. Inspect raw answers. Not a statistical accuracy estimate.')
    Path(output).write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'output':str(output),'passed':report['passed'],'total':report['total']},ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',default='Qwen/Qwen3-0.6B')
    p.add_argument('--adapter',default='')
    p.add_argument('--cases',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--max-tokens',type=int,default=128)
    p.add_argument('--greedy',action='store_true')
    a=p.parse_args()
    evaluate(a.model,a.adapter,json.loads(Path(a.cases).read_text()),a.output,a.max_tokens,a.greedy)
