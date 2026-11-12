"""Optional authenticated loopback receiver. External deployment needs a TLS gateway."""
import hmac
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit,parse_qs

class Receiver:
    def __init__(self,memory,token,port=0):
        if len(token)<24: raise ValueError('Нужен токен длиной не менее 24 символов.')
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args): pass
            def setup(self):
                super().setup(); self.connection.settimeout(5)
            def reply(self,status,data):
                raw=json.dumps(data,ensure_ascii=False,allow_nan=False).encode()
                self.send_response(status); self.send_header('Content-Type','application/json; charset=utf-8')
                self.send_header('Content-Length',str(len(raw))); self.end_headers(); self.wfile.write(raw)
            def authorized(self):
                if not hmac.compare_digest(self.headers.get('Authorization','').encode(), ('Bearer '+token).encode()):
                    self.reply(401,{'error':'Требуется Bearer token.'}); return False
                return True
            def do_POST(self):
                if not self.authorized(): return
                if self.path!='/v1/observations': self.reply(404,{'error':'Not found'}); return
                try:
                    size=int(self.headers.get('Content-Length','0'))
                    if not 0<size<=2*1024*1024: self.reply(413,{'error':'Нужен JSON-пакет не более 2 МБ.'}); return
                    packet=json.loads(self.rfile.read(size))
                    self.reply(200,memory.ingest(packet))
                except (ValueError,TypeError,KeyError) as e: self.reply(400,{'error':str(e)})
                except Exception: self.reply(503,{'error':'Хранилище временно недоступно; повторите тот же пакет.'})
            def do_GET(self):
                if not self.authorized(): return
                url=urlsplit(self.path)
                if url.path=='/health': self.reply(200,{'status':'ok','schema':'panoptr.route.v1'}); return
                if url.path!='/v1/warnings': self.reply(404,{'error':'Not found'}); return
                try:
                    q=parse_qs(url.query)
                    rows=memory.warnings(q['route_id'][0],q['track_id'][0],float(q['chainage_m'][0]),
                        int(q.get('direction',['1'])[0]),float(q.get('lookahead_m',['10000'])[0]),q.get('run_id',[''])[0])
                    self.reply(200,{'warnings':rows,'advisory_only':True,'clear_track_certified':False})
                except (ValueError,KeyError,TypeError) as e: self.reply(400,{'error':str(e)})
                except Exception: self.reply(503,{'error':'Хранилище временно недоступно.'})
        self.server=ThreadingHTTPServer(('127.0.0.1',port),Handler)
        self.server.daemon_threads=True
        self.port=self.server.server_address[1]
        self.thread=threading.Thread(target=self.server.serve_forever,name='route-receiver',daemon=True)
        self.thread.start()
    def close(self):
        self.server.shutdown(); self.server.server_close(); self.thread.join(2)
