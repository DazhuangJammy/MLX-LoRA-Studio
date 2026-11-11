#!/usr/bin/env python3
"""Source-grounded profile evaluation, including generated multi-turn history.
Automatic checks are preliminary; review full answers before claiming accuracy.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import re
import sys
import threading
import time
import unicodedata

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('chat_worker',ROOT/'Backend/chat_runner.py')
chat=importlib.util.module_from_spec(spec);spec.loader.exec_module(chat)


def normalize(text):
    return re.sub(r'[\s\W_]+','',unicodedata.normalize('NFKC',text).lower())


def negated(text,start):
    return bool(re.search(r'没有|不能|不是|并非|无法|未|不等于|不代表|不能确认|不知道',text[max(0,start-24):start]))


def check(answer,case,catalog,finish_reason=None):
    failures=[];clean=normalize(answer)
    for group in case.get('required',[]):
        if not any(normalize(term) in clean for term in group):failures.append('必要信息未覆盖：'+' / '.join(group))
    for phrase in case.get('forbidden',[]):
        for match in re.finditer(re.escape(phrase),answer,re.I):
            if not negated(answer,match.start()):failures.append('不支持的事实断言：'+phrase)
    allowed_titles=[normalize(s) for s in catalog['courses']]+[normalize(e['title']) for e in catalog['events']]
    for match in re.finditer('《([^》]+)》',answer):
        title=normalize(match.group(1))
        if title not in allowed_titles and not negated(answer,match.start()):failures.append('课程/活动名称与来源不符：'+match.group(1))
    for match in re.finditer('青少年科技大会|一等奖|二等奖|金奖|银奖',answer):
        if not negated(answer,match.start()):failures.append('来源未支持的获奖断言：'+match.group(0))
    if case['category']=='服务客户':
        for provider in ['科大讯飞','微软','阿里巴巴','字节跳动','商汤','蚂蚁集团','Datawhale','影刀','亚马逊云科技','豆包','浪潮信息','魔塔社区','讯飞星火','工信部']:
            for match in re.finditer(re.escape(provider),answer,re.I):
                if not negated(answer,match.start()):failures.append('疑似将认证/合作机构列为客户：'+provider)
    if case['category'] in ['边界与纠错','多轮追问'] and re.search(r'\d{7,}',answer):failures.append('疑似编造号码，需人工核对')
    if finish_reason=='length':failures.append('输出达到长度上限，未完整结束')
    if not answer.strip():failures.append('没有回答')
    return list(dict.fromkeys(failures))


def run(args):
    cases=json.loads(Path(args.cases).read_text())
    if len({c['id'] for c in cases}) != len(cases):
        raise ValueError('Evaluation case IDs must be unique.')
    catalog=json.loads(Path(args.catalog).read_text())
    dev=json.loads(Path(args.dev).read_text()) if args.dev else cases
    refs={c['questions'][0]:c for c in dev if len(c['questions'])==1}
    worker=chat.Worker();chat.emit=lambda *a,**kw:None
    worker.load(dict(model=args.model,adapter=args.adapter,quantization=dict(bits=4,group_size=128,mode='affine')))
    results=[]
    output=Path(args.output)
    for index,case in enumerate(cases):
        worker.reset_cache();history=[];turns=[];start=time.monotonic()
        for pos,question in enumerate(case['questions']):
            rubric=(case['turn_rubrics'][pos] if 'turn_rubrics' in case else (case if pos==len(case['questions'])-1 else refs.get(question)))
            budget=args.max_tokens if args.fixed_budget else min(args.max_tokens,max(128,len((rubric or case)['reference'])*3+64))
            events=[];chat.emit=lambda kind,**v:events.append(dict(type=kind,**v))
            history.append(dict(role='user',content=question))
            error=None
            try:
                worker.generate(dict(messages=history,thinking=False,max_tokens=budget),threading.Event())
                answer=''.join(e.get('text','') for e in events if e['type']=='delta')
                done=next((e for e in reversed(events) if e['type']=='done'),{})
            except Exception as exc:answer='';done={};error=f'{type(exc).__name__}: {exc}'
            flags=check(answer,rubric,catalog,done.get('finish_reason')) if rubric else []
            if error:flags.append(error)
            history.append(dict(role='assistant',content=answer))
            turns.append(dict(question=question,answer=answer,output_budget=budget,finish_reason=done.get('finish_reason'),reference=rubric['reference'] if rubric else None,failures=flags,automatic_pass=not flags))
        results.append(dict(id=case['id'],category=case['category'],facts=case['facts'],automatic_pass=all(t['automatic_pass'] for t in turns),turns=turns,seconds=round(time.monotonic()-start,2)))
        if (index+1)%10==0 or index+1==len(cases):print(f'{args.label}: {index+1}/{len(cases)} cases; automatic passes {sum(r["automatic_pass"] for r in results)}',flush=True)
        # Persist incrementally so long evaluations are reviewable even if interrupted.
        report=dict(label=args.label,model=args.model,adapter=args.adapter,max_tokens=args.max_tokens,fixed_budget=args.fixed_budget,cases=str(args.cases),total=len(cases),completed=len(results),automatic_passes=sum(r['automatic_pass'] for r in results),results=results,
                    method='Chat loader and sampler, seed0, thinking off, fresh history per case; multi-turn answers generated by the tested model. No source answers inserted into prompts.',
                    warning='Automatic checks are preliminary. Missing-word flags can be false negatives and extra unsupported claims require manual review.')
        output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',default='Qwen/Qwen3-0.6B');p.add_argument('--adapter',default='')
    p.add_argument('--cases',required=True);p.add_argument('--catalog',required=True);p.add_argument('--dev')
    p.add_argument('--label',required=True);p.add_argument('--output',required=True);p.add_argument('--max-tokens',type=int,default=640)
    p.add_argument('--fixed-budget',action='store_true',help='Use the same maximum output allowance for every question.')
    run(p.parse_args())
