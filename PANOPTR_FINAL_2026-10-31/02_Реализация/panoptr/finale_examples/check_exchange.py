"""Local rehearsal only: no train connection and no external network requests."""
from pathlib import Path
import sys,json,uuid,shutil
from datetime import datetime,timezone
from urllib.error import HTTPError
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from route_memory import RouteMemory,validate
from route_memory_api import Receiver
from route_train_sender import deliver_outbox

def main():
    root=Path(sys.argv[1] if len(sys.argv)>1 else Path(__file__).parent/'results')/uuid.uuid4().hex
    pending=root/'outbox/pending';pending.mkdir(parents=True)
    stamp=datetime.now(timezone.utc).isoformat()
    packet={'schema':'panoptr.route.v1','train_id':'DEMO_NOT_REAL_TRAIN','run_id':'REHEARSAL',
       'observations':[{'observation_id':'demo-1','kind':'track_defect','observed_at':stamp,'quality_ok':True,
        'confidence':.9,'location':{'route_id':'DEMO_750','track_id':'I','chainage_m':10.,'uncertainty_m':1.,'position_kind':'object','verified':True}},
        {'observation_id':'demo-2','kind':'person','observed_at':stamp,'quality_ok':True,'confidence':.8,'location':None}]}
    validate(packet);raw=json.dumps(packet,ensure_ascii=False,indent=2);(root/'example_packet.json').write_text(raw,encoding='utf-8');(pending/'packet.json').write_text(raw,encoding='utf-8')
    memory=RouteMemory(root/'demo.sqlite');token=uuid.uuid4().hex;server=Receiver(memory,token)
    try:
        url=f'http://127.0.0.1:{server.port}/v1/observations'
        try:deliver_outbox(root/'outbox',url,'incorrect-token')
        except HTTPError as e:assert e.code==401
        else:raise AssertionError('Unauthorized request accepted')
        assert (pending/'packet.json').is_file()
        assert deliver_outbox(root/'outbox',url,token)==1
        assert len(memory.objects())==1 and len(memory.unlocated())==1
        shutil.copy2(root/'outbox/sent/packet.json',pending/'packet.json')
        assert deliver_outbox(root/'outbox',url,token)==1
        assert memory.objects()[0]['observations']==1
        report={'passed':True,'scope':'Fictional local rehearsal; no real train connected',
                'authorization_checked':True,'pending_retained_on_failure':True,'retry_deduplicated':True,
                'unknown_location_not_placed_on_map':True,'schema':'panoptr.route.v1'}
        (root/'verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        print('PASS:',root)
    finally:server.close()
if __name__=='__main__':main()
