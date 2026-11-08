from dataset_lab.bootstrap import setup
setup()
import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from pathlib import Path
from datetime import datetime,timezone,timedelta
import copy,json,tempfile,unittest
from urllib.request import Request,urlopen
from urllib.error import HTTPError
from route_memory import RouteMemory,SCHEMA,validate,import_file,session_packets,seed_demo
from route_memory_api import Receiver

NOW=datetime(2026,10,19,12,tzinfo=timezone.utc)

def packet(obs='1',run='A',kind='track_defect',m=1000,age=10,**changes):
    row={'observation_id':obs,'kind':kind,'observed_at':(NOW-timedelta(seconds=age)).isoformat(),
         'quality_ok':True,'confidence':.9,'location':{'route_id':'R','track_id':'I','chainage_m':m,
         'uncertainty_m':2,'position_kind':'object','verified':True}}
    row.update(changes)
    return {'schema':SCHEMA,'train_id':'T','run_id':run,'observations':[row]}

class MemoryChecks(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.root=Path(self.temp.name); self.db=self.root/'map.sqlite';self.memory=RouteMemory(self.db)
    def tearDown(self):self.temp.cleanup()
    def ingest(self,p):return self.memory.ingest(p,NOW)
    def first(self):return self.memory.objects(NOW)[0]
    def confirm(self):self.memory.decide(self.first()['id'],'confirm','Оператор',now=NOW)
    def test_restart_history_and_idempotence(self):
        self.assertEqual(self.ingest(packet())['new'],1)
        self.assertEqual(RouteMemory(self.db).ingest(packet(),NOW)['duplicate'],1)
        self.assertEqual(self.ingest(packet('2','B',m=1002))['repeat'],1)
        row=self.first();self.assertEqual(row['runs'],2);self.assertEqual(row['observations'],2)
    def test_concurrent_delivery_commits_only_one_copy(self):
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=8) as pool:
            results=list(pool.map(lambda _:RouteMemory(self.db).ingest(packet(),NOW),range(16)))
        self.assertEqual(sum(r['new'] for r in results),1)
        self.assertEqual(sum(r['duplicate'] for r in results),15)
        self.assertEqual(self.first()['observations'],1)
    def test_conflicting_id_rolls_back_batch(self):
        self.ingest(packet())
        p=packet('2');p['observations']+=packet(m=2000)['observations']
        with self.assertRaisesRegex(ValueError,'Конфликт'):self.ingest(p)
        self.assertEqual(len(self.memory.objects()),1)
    def test_no_merging_other_path_route_kind_or_far_object(self):
        self.ingest(packet())
        for field,value in [('track_id','II'),('route_id','OTHER'),('chainage_m',1050)]:
            p=packet(field);p['observations'][0]['location'][field]=value; self.ingest(p)
        self.ingest(packet('rock',kind='rock'))
        self.assertEqual(len(self.memory.objects()),5)
    def test_transient_incident_does_not_merge_old_person(self):
        self.ingest(packet('old',kind='person',age=500))
        self.ingest(packet('new',kind='person',age=10))
        self.assertEqual(len(self.memory.objects()),2)
    def test_unlocated_train_position_and_large_error_never_warn(self):
        p=packet(location=None);self.ingest(p)
        for field,value in [('position_kind','train'),('verified',False),('uncertainty_m',26)]:
            p=packet(field);p['observations'][0]['location'][field]=value;self.ingest(p)
        self.assertEqual(len(self.memory.unlocated()),4);self.assertEqual(self.memory.objects(),[])
        self.assertEqual(self.memory.warnings('R','I',0,now=NOW),[])
    def test_operator_places_object_and_retransmit_is_still_idempotent(self):
        p=packet(location=None);self.ingest(p);obs=self.memory.unlocated()[0]
        self.memory.locate(obs['id'],packet()['observations'][0]['location'],'Оператор',NOW)
        self.assertEqual(self.ingest(p)['duplicate'],1)
        self.confirm();self.assertEqual(len(self.memory.warnings('R','I',0,now=NOW)),1)
    def test_confirmation_required_and_quality_is_checked(self):
        self.ingest(packet(quality_ok=False))
        with self.assertRaisesRegex(ValueError,'Нет свежего'):self.confirm()
        self.assertEqual(self.memory.warnings('R','I',0,now=NOW),[])
        self.ingest(packet('good'));self.confirm();self.assertTrue(self.first()['fresh'])
    def test_repeat_does_not_extend_confirmation(self):
        self.ingest(packet(kind='person',age=100));self.confirm()
        self.ingest(packet('new','B',kind='person',age=1))
        later=NOW+timedelta(seconds=21)
        self.assertEqual(self.memory.warnings('R','I',0,now=later),[])
        self.memory.decide(self.first()['id'],'confirm','Оператор',now=later)
        self.assertEqual(len(self.memory.warnings('R','I',0,now=later)),1)
    def test_stale_or_future_cannot_confirm(self):
        self.ingest(packet(kind='person',age=121))
        with self.assertRaises(ValueError):self.confirm()
        with self.assertRaises(ValueError):self.ingest(packet('future',age=-301))
    def test_direction_route_path_lookahead_and_same_run(self):
        self.ingest(packet());self.confirm()
        self.assertEqual(len(self.memory.warnings('R','I',2000,-1,2000,'NEXT',NOW)),1)
        for args in [('R','I',2000,1,2000,'NEXT'),('R','II',0,1,2000,'NEXT'),('X','I',0,1,2000,'NEXT'),('R','I',0,1,500,'NEXT'),('R','I',0,1,2000,'A')]:
            self.assertEqual(self.memory.warnings(*args,now=NOW),[])
    def test_close_requires_reason_and_audit_survives(self):
        self.ingest(packet());self.confirm();obj=self.first()['id']
        with self.assertRaises(ValueError):self.memory.decide(obj,'close','Оператор',now=NOW)
        self.memory.decide(obj,'close','Оператор','Натурная проверка: устранено',now=NOW)
        self.assertEqual(self.memory.warnings('R','I',0,now=NOW),[])
        output=self.root/'archive.json';self.memory.export(output)
        self.assertEqual(len(json.loads(output.read_text(encoding='utf-8'))['audit']),3)
        self.ingest(packet('again','B'));self.assertEqual(len(self.memory.objects()),2)
    def test_out_of_order_times_and_bad_quality_do_not_clear(self):
        self.ingest(packet(age=100));self.confirm()
        self.ingest(packet('older','B',age=500));self.ingest(packet('poor','B',quality_ok=False))
        self.assertTrue(self.first()['fresh']);self.assertEqual(self.first()['state'],'confirmed')
    def test_invalid_whole_packet_has_no_partial_write(self):
        p=packet();p['observations']+=packet('bad',confidence=float('nan'))['observations']
        with self.assertRaises(ValueError):self.ingest(p)
        self.assertEqual(self.memory.objects(),[])
    def test_input_validation(self):
        for changes in [{'kind':'unknown'},{'observed_at':'2026-10-19T12:00:00'}, {'quality_ok':'yes','confidence':True}]:
            with self.assertRaises(ValueError):self.ingest(packet(**changes))
        for field,value in [('chainage_m',float('inf')),('chainage_m',-1),('position_kind','pixel'),('uncertainty_m',None)]:
            p=packet();p['observations'][0]['location'][field]=value
            with self.assertRaises(ValueError):self.ingest(p)
    def test_json_and_jsonl_import(self):
        p=packet();p['observations'][0]['observed_at']=datetime.now(timezone.utc).isoformat()
        file=self.root/'input.jsonl';file.write_text(json.dumps(p)+'\n'+json.dumps(p),encoding='utf-8')
        self.assertEqual(import_file(self.memory,file)['duplicate'],1)
    def test_existing_event_store_bridge_no_invented_location(self):
        session=self.root/'session';folder=session/'events'/('a'*32);folder.mkdir(parents=True)
        (session/'session.json').write_text('{}')
        e={'event_id':'a'*32,'class_name':'person','role':'person','created_utc':NOW.isoformat(),'quality':{'assessed':True}}
        (folder/'event.json').write_text(json.dumps(e))
        p=list(session_packets(session,'T','A'))[0];self.ingest(p)
        self.assertEqual(len(self.memory.unlocated()),1)
        e['hits']=99;(folder/'event.json').write_text(json.dumps(e))
        self.assertEqual(self.ingest(list(session_packets(session,'T','A'))[0])['duplicate'],1)
    def test_demo_contains_all_review_cases(self):
        seed_demo(self.memory)
        self.assertEqual(len(self.memory.objects()),4);self.assertEqual(len(self.memory.unlocated()),1)
        self.assertEqual(len(self.memory.warnings('DEMO-ВСМ','I',124000)),2)

class APIChecks(unittest.TestCase):
    setUp=MemoryChecks.setUp
    tearDown=MemoryChecks.tearDown
    def test_redirect_does_not_forward_token_or_acknowledge_packet(self):
        import threading
        from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
        from route_train_sender import deliver_outbox
        received=[]
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args):pass
            def do_POST(self):
                self.rfile.read(int(self.headers.get('Content-Length','0')))
                self.send_response(302);self.send_header('Location',f'http://127.0.0.1:{self.server.server_port}/capture')
                self.send_header('Content-Length','0');self.end_headers()
            def do_GET(self):
                received.append(self.headers.get('Authorization'));self.send_response(200);self.end_headers()
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            pending=self.root/'outbox/pending';pending.mkdir(parents=True)
            p=packet();p['observations'][0]['observed_at']=datetime.now(timezone.utc).isoformat()
            file=pending/'packet.json';file.write_text(json.dumps(p))
            with self.assertRaisesRegex(ValueError,'Перенаправление'):
                deliver_outbox(pending.parent,f'http://127.0.0.1:{server.server_port}/v1/observations','secret-token')
            self.assertTrue(file.is_file());self.assertEqual(received,[])
        finally:server.shutdown();server.server_close();thread.join(2)
    def test_http_delivery_auth_and_warnings(self):
        server=Receiver(self.memory,'a'*32)
        try:
            url=f'http://127.0.0.1:{server.port}'
            with self.assertRaises(HTTPError) as caught:urlopen(url+'/health',timeout=3)
            self.assertEqual(caught.exception.code,401)
            headers={'Authorization':'Bearer '+'a'*32,'Content-Type':'application/json'}
            p=packet();p['observations'][0]['observed_at']=datetime.now(timezone.utc).isoformat()
            for _ in range(2):
                with urlopen(Request(url+'/v1/observations',json.dumps(p).encode(),headers),timeout=3) as r:self.assertEqual(r.status,200)
            self.memory.decide(self.memory.objects()[0]['id'],'confirm','Оператор')
            with urlopen(Request(url+'/v1/warnings?route_id=R&track_id=I&chainage_m=0&run_id=NEXT',headers=headers),timeout=3) as r:
                result=json.load(r);self.assertEqual(len(result['warnings']),1);self.assertFalse(result['clear_track_certified'])
            with self.assertRaises(HTTPError) as caught:
                urlopen(Request(url+'/v1/observations',b'{',headers),timeout=3)
            self.assertEqual(caught.exception.code,400)
            oversized={**headers,'Content-Length':str(2*1024*1024+1)}
            with self.assertRaises(HTTPError) as caught:
                urlopen(Request(url+'/v1/observations',b'{}',oversized),timeout=3)
            self.assertEqual(caught.exception.code,413)
        finally:server.close()

    def test_train_outbox_survives_failure_and_deduplicates_delivery(self):
        from route_train_sender import spool_session,deliver_outbox
        session=self.root/'session'; folder=session/'events'/('a'*32);folder.mkdir(parents=True)
        (session/'session.json').write_text('{}')
        event={'event_id':'a'*32,'class_name':'person','created_utc':datetime.now(timezone.utc).isoformat(),'quality':{'assessed':True}}
        (folder/'event.json').write_text(json.dumps(event))
        outbox=self.root/'outbox';self.assertEqual(spool_session(session,'T','A',outbox),1)
        self.assertEqual(spool_session(session,'T','A',outbox),0)
        with self.assertRaises(ValueError):deliver_outbox(outbox,'http://remote.example/v1/observations','token')
        server=Receiver(self.memory,'b'*32)
        try:
            url=f'http://127.0.0.1:{server.port}/v1/observations'
            with self.assertRaises(HTTPError):deliver_outbox(outbox,url,'incorrect-token')
            self.assertEqual(len(list((outbox/'pending').glob('*.json'))),1)
            self.assertEqual(deliver_outbox(outbox,url,'b'*32),1)
            self.assertEqual(deliver_outbox(outbox,url,'b'*32),0)
            self.assertEqual(spool_session(session,'T','A',outbox),0)
            self.assertEqual(len(self.memory.unlocated()),1)
        finally:server.close()

class GUIChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.app=QApplication.instance() or QApplication([])
    def test_demo_review_map_preview_and_restart(self):
        from route_memory_ui import RouteMemoryWindow
        with tempfile.TemporaryDirectory() as temp:
            db=Path(temp)/'map.sqlite';memory=RouteMemory(db);seed_demo(memory)
            window=RouteMemoryWindow(db);window.timer.stop();window.show();self.app.processEvents()
            try:
                self.assertEqual(window.table.rowCount(),4);self.assertEqual(window.unlocated_table.rowCount(),1)
                window.table.selectRow(0);window.show_detail();self.assertIn('История',window.detail.toPlainText())
                from PySide6.QtGui import QFontMetrics
                self.assertTrue(QFontMetrics(window.font()).inFontUcs4(ord('Ж')))
                window.position.setValue(124);window.preview();self.assertIn('125.000',window.preview_text.toPlainText())
                selected=window.selected_id();window.refresh();self.assertEqual(window.selected_id(),selected)
                window.state.setCurrentText('Подтверждено');self.assertEqual(window.table.rowCount(),2)
                window.state.setCurrentText('Все');self.assertEqual(window.table.rowCount(),4)
                window.resize(1100,760);self.app.processEvents()
                self.assertFalse(window.grab().isNull())
            finally:window.close();self.app.processEvents()
    def test_main_window_has_integrated_button(self):
        from unittest.mock import patch
        import main
        with tempfile.TemporaryDirectory() as temp:
            config=main.load_config();config['event_session']=str(Path(temp)/'session')
            with patch.object(main,'discover_camera_devices',return_value=[]):window=main.MainWindow(config)
            try:
                from PySide6.QtWidgets import QPushButton
                self.assertIn('Карта рейсов',[b.text() for b in window.findChildren(QPushButton)])
                with patch('route_memory_ui.DEFAULT_DB',Path(temp)/'map.sqlite'):
                    window._open_route_memory();self.app.processEvents()
                self.assertTrue(window.route_memory_window.isVisible())
            finally:window.close();self.app.processEvents()
            self.assertFalse(window.route_memory_window.timer.isActive())

    def test_unlocated_selection_and_dialog_target_survive_incoming_events(self):
        from route_memory_ui import RouteMemoryWindow
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as temp:
            memory=RouteMemory(Path(temp)/'map.sqlite')
            def add(identifier):
                p=packet(identifier,location=None);p['observations'][0]['observed_at']=datetime.now(timezone.utc).isoformat()
                memory.ingest(p)
            add('original')
            window=RouteMemoryWindow(memory.path);window.timer.stop()
            try:
                window.unlocated_table.selectRow(0);original=window.unknown[0]['id']
                add('arrives-before-dialog');window.refresh()
                self.assertEqual(window.unknown[window.unlocated_table.currentRow()]['id'],original)
                def dialog(*args):
                    add('arrives-during-dialog');window.refresh()
                    return {'actor':'Оператор','route':'R','track':'I','km':'1','error':'2'}
                with patch('route_memory_ui.form',side_effect=dialog):window.locate()
                self.assertEqual(memory.observations(memory.objects()[0]['id'])[0]['external_id'],'original')
                self.assertEqual(len(memory.unlocated()),2)
            finally:window.close();self.app.processEvents()

if __name__=='__main__':unittest.main()
