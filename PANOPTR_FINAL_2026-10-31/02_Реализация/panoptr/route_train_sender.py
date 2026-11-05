"""Train-side disk outbox. Acknowledged packets move to sent; retries use identical IDs."""
from pathlib import Path
from urllib.request import Request,build_opener,HTTPRedirectHandler
from urllib.parse import urlsplit
import argparse
import hashlib
import json
import os
import time
from route_memory import session_packets,validate

class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self,request,fp,code,msg,headers,newurl):
        raise ValueError('Перенаправление не разрешено: укажите конечный адрес приёмника.')

def spool_session(session,train,run,outbox):
    outbox=Path(outbox);pending=outbox/'pending';sent=outbox/'sent'
    pending.mkdir(parents=True,exist_ok=True);sent.mkdir(parents=True,exist_ok=True)
    count=0
    for packet in session_packets(session,train,run):
        validate(packet)
        key=hashlib.sha256(json.dumps([train,run,packet['observations'][0]['observation_id']]).encode()).hexdigest()+'.json'
        if (sent/key).exists() or (pending/key).exists():continue
        target=pending/key;temp=target.with_suffix('.tmp')
        temp.write_text(json.dumps(packet,ensure_ascii=False),encoding='utf-8');temp.replace(target);count+=1
    return count

def deliver_outbox(outbox,url,token):
    parsed=urlsplit(url)
    if parsed.scheme not in ('https','http') or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError('Нужен адрес HTTPS без пароля в URL.')
    if parsed.scheme=='http' and parsed.hostname not in ('127.0.0.1','localhost','::1'):
        raise ValueError('HTTP разрешён только для localhost. Удалённому серверу нужен HTTPS.')
    if not token:raise ValueError('Установите PANOPTR_ROUTE_TOKEN в окружении отправителя.')
    outbox=Path(outbox);sent=outbox/'sent';sent.mkdir(parents=True,exist_ok=True);count=0
    for path in sorted((outbox/'pending').glob('*.json')):
        raw=path.read_bytes();validate(json.loads(raw))
        request=Request(url,raw,{'Authorization':'Bearer '+token,'Content-Type':'application/json'},method='POST')
        with build_opener(NoRedirect()).open(request,timeout=8) as response:
            result=json.loads(response.read(65536))
            if response.status!=200 or sum(result.get(k,0) for k in ('new','repeat','unlocated','duplicate'))!=len(json.loads(raw)['observations']):
                raise ValueError('Сервер не подтвердил весь пакет; он оставлен в очереди.')
        path.replace(sent/path.name);count+=1
    return count

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--session',required=True);p.add_argument('--train-id',required=True);p.add_argument('--run-id',required=True)
    p.add_argument('--outbox',required=True);p.add_argument('--url');p.add_argument('--watch',action='store_true')
    args=p.parse_args()
    while True:
        try:
            queued=spool_session(args.session,args.train_id,args.run_id,args.outbox)
            sent=deliver_outbox(args.outbox,args.url,os.environ.get('PANOPTR_ROUTE_TOKEN','')) if args.url else 0
            print(f'Queued: {queued}; acknowledged: {sent}',flush=True)
        except Exception as exc:
            print(f'Not delivered ({type(exc).__name__}); pending files retained.',flush=True)
            if not args.watch:return 1
        if not args.watch:return 0
        time.sleep(5)

if __name__=='__main__':raise SystemExit(main())
