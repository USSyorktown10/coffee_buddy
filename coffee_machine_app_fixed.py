# coffee_machine_app_fixed.py
# One-file AI coffee machine app

import os, sys, json, time, math, glob, subprocess, webbrowser, threading
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
from pathlib import Path

def _ensure(pkgs: List[str]):
    import importlib
    missing=[]
    for p in pkgs:
        try: importlib.import_module(p.split("[")[0])
        except Exception: missing.append(p)
    if missing:
        print("Installing:", missing)
        subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])

_ensure(["fastapi", "uvicorn[standard]", "pydantic", "numpy"])

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np

BASE = Path(__file__).parent.resolve()
DATA = BASE / "data"
MODELS = BASE / "models"
for d in (DATA, MODELS): d.mkdir(exist_ok=True)

HW_PATH = BASE / "hardware_map.json"
if not HW_PATH.exists():
    HW_PATH.write_text(json.dumps({
        "pins": {
            "heater_relay": 17, "pump_relay": 27, "grinder_relay": 26,
            "pressure_valve": 18, "stepper_enable": 16, "stepper_step": 20, "stepper_dir": 21,
            "flow_sensor": 12, "start_button": 24, "led_ring": 25
        },
        "hx711": {"dout": 5, "pd_sck": 6, "scale": 415.0, "offset": 0.0},
        "safety": {"max_temp_c": 98.0, "max_pressure_bar": 10.0, "max_pump_seconds": 50.0, "max_heater_minutes": 7.0},
        "dose": {"target_grams": 18.0, "tolerance_grams": 0.3}
    }, indent=2))
HW = json.loads(HW_PATH.read_text())

try:
    import RPi.GPIO as GPIO
    GPIO.setmode(GPIO.BCM)
    HAS_GPIO = True
except Exception:
    HAS_GPIO = False
    class GPIO:
        BCM="BCM"; OUT="OUT"; IN="IN"; PUD_UP=None; HIGH=1; LOW=0; RISING=None
        @staticmethod
        def setmode(*a, **k): pass
        @staticmethod
        def setup(*a, **k): pass
        @staticmethod
        def output(*a, **k): print(f"[GPIO-DUMMY] output{a}")
        class _PWM:
            def start(self,*a): pass
            def ChangeDutyCycle(self,*a): pass
            def stop(self): pass
        @staticmethod
        def PWM(*a, **k): return GPIO._PWM()
        @staticmethod
        def add_event_detect(*a, **k): pass
        @staticmethod
        def cleanup(): pass

def read_temp_c() -> Optional[float]:
    try:
        devs = glob.glob('/sys/bus/w1/devices/28-*/w1_slave')
        if not devs: return None
        with open(devs[0], 'r') as f:
            lines=f.readlines()
        if not lines[0].strip().endswith('YES'): return None
        p = lines[1].find('t=')
        if p == -1: return None
        return float(lines[1][p+2:]) / 1000.0
    except Exception:
        return None

try:
    import board, busio, adafruit_ads1x15.ads1115 as ADS
    from adafruit_ads1x15.analog_in import AnalogIn
    _i2c = busio.I2C(board.SCL, board.SDA)
    _ads = ADS.ADS1115(_i2c)
    _tds = AnalogIn(_ads, ADS.P0); _ph = AnalogIn(_ads, ADS.P1); _voc = AnalogIn(_ads, ADS.P2)
    HAS_ADS=True
except Exception:
    HAS_ADS=False
    _tds=_ph=_voc=None

def _safe_v(ch):
    try: return round(ch.voltage, 3)
    except Exception: return 0.0

try:
    from hx711 import HX711
    HAS_HX=True
except Exception:
    HAS_HX=False
    HX711=None

_hx=None
def hx_init():
    global _hx
    if not HAS_HX: return False
    dout=HW["hx711"]["dout"]; sck=HW["hx711"]["pd_sck"]
    scale=HW["hx711"].get("scale",415.0); off=HW["hx711"].get("offset",0.0)
    _hx = HX711(dout, sck); _hx.set_scale(scale); _hx.set_offset(off); _hx.tare()
    return True

def hx_read_grams(n: int=5) -> float:
    if _hx is None: return 0.0
    try: return float(_hx.get_weight(n))
    except Exception: return 0.0

FLOW_COUNT=0
def _flow_cb(channel):
    global FLOW_COUNT
    FLOW_COUNT += 1

def flow_init():
    pin = HW["pins"].get("flow_sensor")
    if not pin: return
    try:
        GPIO.setup(pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        GPIO.add_event_detect(pin, GPIO.RISING, callback=_flow_cb, bouncetime=1)
    except Exception:
        pass

def flow_read_and_reset() -> int:
    global FLOW_COUNT
    c = FLOW_COUNT; FLOW_COUNT = 0; return c

def read_taste_profile():
    return {
        "tds": _safe_v(_tds) if HAS_ADS else 0.0,
        "ph":  _safe_v(_ph)  if HAS_ADS else 0.0,
        "aroma": _safe_v(_voc) if HAS_ADS else 0.0,
        "temp_c": read_temp_c()
    }

class SimpleTasteModel:
    def __init__(self, n_users=1000, seed=42):
        rng = np.random.default_rng(seed)
        self.user_emb = rng.normal(scale=0.1, size=(n_users, 8))
        self.W = rng.normal(scale=0.1, size=(16+8+8, 4))
        self.b = np.zeros((4,), dtype=float)
        self.n_users = n_users

    def forward(self, user_id: int, bean_feat, brew_feat):
        u = self.user_emb[user_id % self.n_users]
        x = np.concatenate([np.array(bean_feat, float), np.array(brew_feat, float), u])
        y = x @ self.W + self.b
        return np.clip(y, 0.0, 10.0)

    def train_step(self, user_id, bean_feat, brew_feat, target, lr=1e-2):
        u_idx = user_id % self.n_users
        u = self.user_emb[u_idx]
        x = np.concatenate([np.array(bean_feat, float), np.array(brew_feat, float), u])
        y = x @ self.W + self.b
        err = y - np.array(target, float)
        gW = np.outer(x, err) * 2.0
        gb = err * 2.0
        self.W -= lr * gW
        self.b -= lr * gb
        gu = (self.W[-8:, :] @ err) * 2.0
        self.user_emb[u_idx] -= lr * gu
        return float(np.mean(err**2))

MODEL_FILE = MODELS / "simple_model.json"
def _model_load() -> SimpleTasteModel:
    m = SimpleTasteModel()
    if MODEL_FILE.exists():
        d = json.loads(MODEL_FILE.read_text())
        m.n_users = d["n_users"]
        m.user_emb = np.array(d["user_emb"])
        m.W = np.array(d["W"])
        m.b = np.array(d["b"])
    return m
def _model_save(m: SimpleTasteModel):
    MODEL_FILE.write_text(json.dumps({
        "n_users": m.n_users,
        "user_emb": m.user_emb.tolist(),
        "W": m.W.tolist(),
        "b": m.b.tolist()
    }))
MODEL = _model_load()

@dataclass
class BOConfig:
    brew_bounds: List[Tuple[float,float]] = field(default_factory=lambda: [(-2.0, 2.0)]*8)
    noise: float = 1e-6
    lengthscale: float = 1.0
    variance: float = 1.0
    xi: float = 0.01

@dataclass
class BOState:
    X: List[List[float]] = field(default_factory=list)
    y: List[float] = field(default_factory=list)
    config: BOConfig = field(default_factory=BOConfig)
    def to_json(self):
        return {"X": self.X, "y": self.y, "config": {
            "brew_bounds": self.config.brew_bounds, "noise": self.config.noise,
            "lengthscale": self.config.lengthscale, "variance": self.config.variance, "xi": self.config.xi
        }}
    @staticmethod
    def from_json(d):
        cfg = d.get("config", {})
        return BOState(d.get("X",[]), d.get("y",[]),
                       BOConfig(cfg.get("brew_bounds",[(-2,2)]*8), cfg.get("noise",1e-6),
                                cfg.get("lengthscale",1.0), cfg.get("variance",1.0), cfg.get("xi",0.01)))

class SimpleGP:
    def __init__(self, X, y, l=1.0, s2=1.0, noise=1e-6):
        self.X = np.array(X, float) if len(X)>0 else np.zeros((0,1))
        self.y = np.array(y, float) if len(y)>0 else np.zeros((0,))
        self.l, self.s2, self.noise = float(l), float(s2), float(noise)
        if len(self.X)>0: self._fit()
    def _rbf(self, A, B):
        a2 = np.sum(A*A, axis=1, keepdims=True)
        b2 = np.sum(B*B, axis=1, keepdims=True).T
        sq = a2 + b2 - 2*np.dot(A, B.T)
        return self.s2 * np.exp(-0.5 * sq / (self.l**2 + 1e-12))
    def _fit(self):
        n = self.X.shape[0]
        self.K = self._rbf(self.X, self.X)
        self.K.flat[::n+1] += self.noise
        self.L = np.linalg.cholesky(self.K + 1e-12*np.eye(n))
    def predict(self, Xs):
        Xs = np.array(Xs, float)
        if len(self.X)==0:
            return np.zeros((Xs.shape[0],)), np.ones((Xs.shape[0],))*self.s2
        Kxs = self._rbf(self.X, Xs)
        alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, self.y))
        mu = Kxs.T @ alpha
        v = np.linalg.solve(self.L, Kxs)
        var = self.s2 - np.sum(v*v, axis=0)
        return mu, np.maximum(var, 1e-12)

def expected_improvement(mu, var, best, xi=0.01):
    sigma = np.sqrt(var)
    Z = (mu - best - xi) / (sigma + 1e-12)
    pdf = (1.0/np.sqrt(2*np.pi))*np.exp(-0.5*Z*Z)
    cdf = 0.5*(1.0 + np.erf(Z/np.sqrt(2)))
    ei = (mu - best - xi)*cdf + sigma*pdf
    ei[sigma<1e-12]=0.0
    return ei

BO_FILE = DATA / "bo_state.json"
def _bo_load():
    if BO_FILE.exists():
        return BOState.from_json(json.loads(BO_FILE.read_text()))
    return BOState()
def _bo_save(st: BOState):
    BO_FILE.write_text(json.dumps(st.to_json(), indent=2))
def bo_suggest(n_candidates=128):
    st = _bo_load()
    bnds = st.config.brew_bounds
    lo = np.array([a for a,b in bnds]); hi = np.array([b for a,b in bnds])
    cand = np.random.uniform(lo, hi, size=(n_candidates, len(bnds)))
    gp = SimpleGP(st.X, st.y, st.config.lengthscale, st.config.variance, st.config.noise)
    mu, var = gp.predict(cand)
    best = np.max(st.y) if st.y else 0.0
    ei = expected_improvement(mu, var, best, st.config.xi)
    return cand[int(np.argmax(ei))].tolist()
def bo_observe(brew_feat: List[float], score: float):
    st=_bo_load()
    st.X.append(list(brew_feat)); st.y.append(float(score)); _bo_save(st)

HOPPER_FILE = DATA / "hopper_prefs.json"
def _hopper_load():
    if HOPPER_FILE.exists():
        try: return json.loads(HOPPER_FILE.read_text())
        except: pass
    d={"light_score":0, "dark_score":0}
    HOPPER_FILE.write_text(json.dumps(d))
    return d
def _hopper_save(d):
    HOPPER_FILE.write_text(json.dumps(d, indent=2))

@dataclass
class BrewStatus:
    session_id: str
    running: bool = False
    step: str = "idle"
    last_pred: Optional[Dict[str, float]] = None
    last_sensors: Optional[Dict[str, float]] = None
    message: str = ""
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    progress: float = 0.0
    eta_seconds: Optional[float] = None

class Controller:
    def __init__(self):
        self.pins = HW["pins"]
        self.safety = HW["safety"]
        self.dose_cfg = HW["dose"]
        for k in ("heater_relay","pump_relay","grinder_relay","pressure_valve","stepper_enable","stepper_step","stepper_dir","led_ring"):
            if k in self.pins:
                try: GPIO.setup(self.pins[k], GPIO.OUT)
                except: pass
        flow_init()
        if HAS_HX: hx_init()
        self.valve_pwm = None
        try:
            self.valve_pwm = GPIO.PWM(self.pins["pressure_valve"], 50); self.valve_pwm.start(0)
        except Exception:
            self.valve_pwm = None
        self._stop = threading.Event()
        self.status = BrewStatus(session_id="none")
        self._thread: Optional[threading.Thread] = None

    def _relay(self,name,on):
        p=self.pins.get(name); 
        if p is None: return
        GPIO.output(p, GPIO.HIGH if on else GPIO.LOW)
    def heater(self,on): self._relay("heater_relay",on)
    def pump(self,on):   self._relay("pump_relay",on)
    def grinder(self,on):self._relay("grinder_relay",on)
    def ring(self,on):   self._relay("led_ring",on)
    def valve_set(self, frac: float):
        if not self.valve_pwm: return
        v=max(0.0,min(1.0,float(frac)))
        self.valve_pwm.ChangeDutyCycle(2.5+10.0*v)

    def stepper_enable(self, en: bool):
        pin = self.pins.get("stepper_enable"); 
        if pin is None: return
        GPIO.output(pin, GPIO.LOW if en else GPIO.HIGH)
    def stepper_move(self, steps: int, direction: int, delay=0.001):
        d = self.pins.get("stepper_dir"); s = self.pins.get("stepper_step")
        if d is None or s is None: return
        GPIO.output(d, GPIO.HIGH if direction else GPIO.LOW)
        for _ in range(abs(steps)):
            GPIO.output(s, GPIO.HIGH); time.sleep(delay)
            GPIO.output(s, GPIO.LOW);  time.sleep(delay)
    def select_hopper(self, which: str):
        steps = 800
        self.stepper_enable(True)
        if which == "light":
            self.stepper_move(steps, 0)
        else:
            self.stepper_move(steps, 1)
        self.stepper_enable(False)

    def stop(self): self._stop.set()

    def start(self, session_id: str, user_id: int, targets: Dict[str,float], bayesian: bool, hopper_mode: str):
        if self._thread and self._thread.is_alive():
            raise RuntimeError("Brew already running")
        self._stop.clear()
        self.status = BrewStatus(session_id=session_id, running=True, step="preheat")
        self._thread = threading.Thread(target=self._run, args=(session_id,user_id,targets,bayesian,hopper_mode), daemon=True)
        self._thread.start()

    def _predict(self, user_id, bean_feat, brew_feat):
        y = MODEL.forward(user_id, bean_feat, brew_feat)
        return [float(np.clip(v,0,10)) for v in y]

    def _run(self, session_id, user_id, targets, bayesian, hopper_mode):
        try:
            bean_feat=[0.0]*16; brew_feat=[0.0]*8
            self.ring(True)
            prefs = _hopper_load()
            if hopper_mode == "light":
                chosen = "light"
            elif hopper_mode == "dark":
                chosen = "dark"
            else:
                chosen = "light" if prefs.get("light_score",0) >= prefs.get("dark_score",0) else "dark"
            self.select_hopper(chosen)
            self.status.message = f"Using {chosen} hopper"
            self.status.step="preheat"; self.status.progress = 0.05; self.status.eta_seconds = 180.0
            self.heater(True); t0=time.time()
            while True:
                if self._stop.is_set(): break
                s=read_taste_profile(); self.status.last_sensors=s
                tc=s.get("temp_c")
                if tc is not None and tc >= 94.0: break
                if time.time()-t0 > 7.0*60: raise RuntimeError("Heater timeout")
                time.sleep(0.5)
            self.heater(False)
            self.status.progress = 0.25; self.status.eta_seconds = 120.0
            self.status.step="dose"
            target_g=18.0; tol=0.3
            self.grinder(True); d0=time.time()
            while True:
                if self._stop.is_set(): break
                g=hx_read_grams(3)
                if g >= (target_g - tol): break
                if time.time()-d0 > 25.0: break
                time.sleep(0.1)
            self.grinder(False)
            self.status.progress = 0.40; self.status.eta_seconds = 75.0
            self.status.step="preinfuse"
            self.pump(True); time.sleep(2.0); self.pump(False)
            self.status.progress = 0.45; self.status.eta_seconds = 60.0
            self.status.step="control"
            pumped=0.0; pump_limit=50.0
            for i in range(20):
                if self._stop.is_set(): break
                s=read_taste_profile()
                y=self._predict(user_id, bean_feat, brew_feat)
                self.status.last_sensors=s
                self.status.last_pred={"bitter":y[0],"acidity":y[1],"body":y[2],"aroma":y[3]}
                i_prog = 0.45 + (i / 20.0) * 0.50
                self.status.progress = min(0.95, i_prog)
                remaining = (20 - i) * 2.0
                self.status.eta_seconds = max(10.0, remaining + 10.0)
                if bayesian:
                    brew_feat = bo_suggest(64)
                tc=s.get("temp_c")
                if tc and tc>98.0: self.heater(False)
                if y[0] > targets.get("bitter",6.0):
                    self.heater(False); time.sleep(0.6)
                else:
                    self.heater(True);  time.sleep(0.3)
                if y[2] < targets.get("body",6.0) and pumped < pump_limit:
                    self.pump(True); time.sleep(1.2); self.pump(False); pumped+=1.2
                else:
                    time.sleep(0.4)
                if y[1] > targets.get("acidity",6.0):
                    self.valve_set(0.2); time.sleep(0.5)
                else:
                    self.valve_set(0.8); time.sleep(0.2)
            self.status.step="finish"
            self.heater(False); self.pump(False); self.valve_set(0.0); self.ring(False)
            self.status.running=False; self.status.finished_at=time.time(); self.status.message=f"Brew complete ({chosen} hopper)"
            self.status.progress = 1.0; self.status.eta_seconds = 0.0
        except Exception as e:
            self.heater(False); self.pump(False); self.valve_set(0.0); self.ring(False)
            self.status.running=False; self.status.message=f"Error: {e}"
            self.status.eta_seconds = None
        finally:
            try: GPIO.cleanup()
            except: pass

CONTROLLER = Controller()

FEEDBACK_LOG = DATA / "feedback.jsonl"
BREW_LOG     = DATA / "brews.jsonl"

def log_line(path: Path, rec: dict):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec)+"\n")

def train_from_feedback(epochs=2, base_samples=1200, seed=123):
    rng=np.random.default_rng(seed)
    liked=[]; disliked=[]
    if FEEDBACK_LOG.exists() and BREW_LOG.exists():
        fb={}
        for line in FEEDBACK_LOG.read_text().splitlines():
            try: r=json.loads(line); fb[r["session_id"]]=bool(r.get("liked"))
            except: pass
        for line in BREW_LOG.read_text().splitlines():
            try:
                r=json.loads(line)
                if r.get("event")=="start" and r.get("session_id") in fb:
                    tgt=r.get("target",{})
                    (liked if fb[r["session_id"]] else disliked).append(tgt)
            except: pass
    samples = base_samples + 200*len(liked) + 50*len(disliked)
    bean = rng.normal(size=(samples,16))
    brew = rng.normal(size=(samples,8))
    users= rng.integers(0, MODEL.n_users, size=(samples,))
    for ep in range(epochs):
        perm=rng.permutation(samples)
        total=0.0
        for i in perm:
            base = (bean[i,:4].mean()+brew[i,:4].mean())*2 + rng.normal(scale=0.3, size=4)
            mn, mx = base.min(), base.max()
            if mx-mn<1e-6: target=np.zeros(4)
            else:          target=((base-mn)/(mx-mn)*10.0)
            if liked and rng.random()<0.4:
                t = liked[rng.integers(0,len(liked))]
                bias=np.array([t.get("bitter",6.0), t.get("acidity",6.0), t.get("body",6.0), t.get("aroma",6.0)])
                target = 0.7*target + 0.3*bias
            if disliked and rng.random()<0.2:
                t = disliked[rng.integers(0,len(disliked))]
                bias=np.array([t.get("bitter",6.0), t.get("acidity",6.0), t.get("body",6.0), t.get("aroma",6.0)])
                target = 0.9*target + 0.1*(10.0-bias)
            loss = MODEL.train_step(int(users[i]), bean[i], brew[i], target, lr=5e-3)
            total += loss
        print(f"[feedback-train] epoch {ep+1}/{epochs} loss={total/samples:.4f}")
    _model_save(MODEL)
    return {"ok": True, "message": "Model updated from feedback."}

app = FastAPI(title="Coffee Machine")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# IMPORTANT FIX: No f-prefix here, so braces in JS/CSS are not interpreted.
INDEX_HTML = """<!doctype html>
<html><head><meta charset='utf-8'/><meta name='viewport' content='width=device-width, initial-scale=1'>
<title>Coffee Kiosk</title>
<style>
html,body{height:100%;margin:0;background:#0f0f10;color:#fff;font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial}
.wrap{display:flex;flex-direction:column;height:100%}
header{padding:16px 20px;background:#1a1a1c;border-bottom:1px solid #222}
main{flex:1;display:flex;gap:20px;padding:20px;flex-wrap:wrap}
.card{background:#161618;border:1px solid #26262a;border-radius:16px;padding:16px;flex:1;min-width:340px}
label{display:block;margin:8px 0 4px;color:#ddd}
input,select{width:100%;padding:10px;border-radius:10px;border:1px solid #2e2e33;background:#0f0f10;color:#fff}
button{padding:12px 16px;border-radius:12px;border:0;background:#2a7efc;color:#fff;font-weight:600;cursor:pointer;margin-right:8px}
button.secondary{background:#303036}
.big{font-size:20px;padding:14px 22px}
.ok{background:#27ae60} .bad{background:#e74c3c}
pre{background:#0c0c0d;border:1px solid #26262a;padding:12px;border-radius:12px;max-height:240px;overflow:auto}
.small{color:#bbb;font-size:14px}
.barwrap{height:12px;border-radius:8px;background:#27272a}
.barinner{height:12px;width:0%;border-radius:8px;background:#2a7efc}
</style>
</head>
<body>
<div class="wrap">
<header>☕ Coffee Kiosk</header>
<main>
<section class="card">
  <h2>Brew</h2>
  <label>Session ID</label>
  <input id="sess" value="brew-001"/>
  <label><input id="bayes" type="checkbox"/> Bayesian mode</label>
  <label>Hopper</label>
  <select id="hopper">
    <option value="auto" selected>Auto (learns from feedback)</option>
    <option value="light">Light Roast</option>
    <option value="dark">Dark Roast</option>
  </select>
  <div style="display:flex;gap:10px">
    <div style="flex:1"><label>Target Bitter</label><input id="tb" type="number" value="6.0" step="0.1"/></div>
    <div style="flex:1"><label>Target Body</label><input id="tbo" type="number" value="6.0" step="0.1"/></div>
  </div>
  <div style="display:flex;gap:10px">
    <div style="flex:1"><label>Target Acidity</label><input id="ta" type="number" value="6.0" step="0.1"/></div>
    <div style="flex:1"><label>Target Aroma</label><input id="tar" type="number" value="6.0" step="0.1"/></div>
  </div><br/>
  <button class="big" onclick="startBrew()">▶ Brew</button>
  <button class="big secondary" onclick="stopBrew()">■ Stop</button>
  <button class="secondary" onclick="statusBrew()">Refresh Status</button>
  <div style="margin:10px 0;">
    <div class="barwrap"><div id="bar" class="barinner"></div></div>
    <div id="eta" class="small">ETA: —</div>
  </div>
  <pre id="brewOut"></pre>
</section>

<section class="card">
  <h2>Feedback</h2>
  <button class="big ok" onclick="sendFeedback(true)">👍 Thumbs Up</button>
  <button class="big bad" onclick="sendFeedback(false)">👎 Thumbs Down</button>
  <br/><br/>
  <button class="secondary" onclick="train()">Train from Feedback</button>
  <pre id="fbOut"></pre>
</section>

<section class="card">
  <h2>Debug</h2>
  <button class="secondary" onclick="getSensors()">/sensors/read</button>
  <button class="secondary" onclick="getStatus()">/debug/status</button>
  <pre id="dbgOut"></pre>
</section>
</main>
</div>
<script>
const API = location.origin;

async function startBrew(){
  const body = {
    session_id: document.getElementById('sess').value || 'brew-001',
    user_id: 1,
    target_bitter: parseFloat(document.getElementById('tb').value),
    target_body: parseFloat(document.getElementById('tbo').value),
    target_acidity: parseFloat(document.getElementById('ta').value),
    target_aroma: parseFloat(document.getElementById('tar').value),
    bayesian: document.getElementById('bayes').checked,
    hopper_mode: document.getElementById('hopper').value
  };
  const r = await fetch(API + '/brew/start', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)});
  document.getElementById('brewOut').textContent = await r.text();
}
async function stopBrew(){
  const r = await fetch(API + '/brew/stop', {method:'POST'});
  document.getElementById('brewOut').textContent = await r.text();
}
async function statusBrew(){
  const r = await fetch(API + '/brew/status');
  document.getElementById('brewOut').textContent = await r.text();
}
async function sendFeedback(liked){
  const body = { session_id: document.getElementById('sess').value || 'brew-001', liked };
  const r = await fetch(API + '/feedback', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)});
  document.getElementById('fbOut').textContent = await r.text();
}
async function train(){
  const r = await fetch(API + '/train/feedback', {method:'POST'});
  document.getElementById('fbOut').textContent = await r.text();
}
async function getSensors(){
  const r = await fetch(API + '/sensors/read'); document.getElementById('dbgOut').textContent = await r.text();
}
async function getStatus(){
  const r = await fetch(API + '/debug/status'); document.getElementById('dbgOut').textContent = await r.text();
}
async function pollStatus(){
  try{
    const r = await fetch(API + '/debug/status');
    const s = await r.json();
    const p = Math.max(0, Math.min(1, (s.progress ?? 0)));
    document.getElementById('bar').style.width = Math.round(p*100)+'%';
    const eta = s.eta_seconds!=null ? Math.max(0, Math.round(s.eta_seconds))+'s' : '—';
    document.getElementById('eta').textContent = 'ETA: ' + eta;
  }catch(e){}
}
setInterval(pollStatus, 1000);
</script>
</body></html>
"""

class PredictBody(BaseModel):
    user_id: int
    bean_feat: List[float]
    brew_feat: List[float]

class BrewStartBody(BaseModel):
    session_id: str
    user_id: int = 1
    target_bitter: float = 6.0
    target_body: float = 6.0
    target_acidity: float = 6.0
    target_aroma: float = 6.0
    bayesian: bool = False
    hopper_mode: str = "auto"

class FeedbackBody(BaseModel):
    session_id: str
    liked: bool

class BOSuggestBody(BaseModel):
    n_candidates: int = 128

class BOObserveBody(BaseModel):
    brew_feat: List[float]
    score: float

@app.get("/", response_class=HTMLResponse)
def root():
    return INDEX_HTML

@app.get("/debug/status")
def dbg_status():
    s = CONTROLLER.status
    return {
        "running": s.running,
        "step": s.step,
        "last_pred": s.last_pred,
        "last_sensors": s.last_sensors,
        "message": s.message,
        "progress": s.progress,
        "eta_seconds": s.eta_seconds
    }

@app.get("/sensors/read")
def sensors_read():
    return read_taste_profile()

@app.post("/predict")
def predict(b: PredictBody):
    y = MODEL.forward(b.user_id, b.bean_feat, b.brew_feat).tolist()
    return {"predicted_ratings": [float(v) for v in y]}

@app.post("/brew/start")
def brew_start(b: BrewStartBody):
    if CONTROLLER.status.running:
        return {"ok": False, "error": "brew already running", "status": CONTROLLER.status.__dict__}
    targets = {"bitter": b.target_bitter, "body": b.target_body, "acidity": b.target_acidity, "aroma": b.target_aroma}
    CONTROLLER.start(b.session_id, b.user_id, targets, b.bayesian, b.hopper_mode)
    log_line(BREW_LOG, {"t": time.time(), "event":"start", "session_id": b.session_id, "target": targets, "bayesian": b.bayesian, "hopper": b.hopper_mode})
    return {"ok": True, "status": CONTROLLER.status.__dict__}

@app.get("/brew/status")
def brew_status():
    s = CONTROLLER.status
    return {"running": s.running, "step": s.step, "last_pred": s.last_pred, "last_sensors": s.last_sensors, "message": s.message, "progress": s.progress, "eta_seconds": s.eta_seconds}

@app.post("/brew/stop")
def brew_stop():
    CONTROLLER.stop()
    return {"ok": True}

@app.post("/feedback")
def feedback(b: FeedbackBody):
    rec={"t": time.time(), "session_id": b.session_id, "liked": b.liked}
    log_line(FEEDBACK_LOG, rec)
    used = None
    if BREW_LOG.exists():
        for line in BREW_LOG.read_text().splitlines()[::-1]:
            try:
                r = json.loads(line)
                if r.get("event")=="start" and r.get("session_id")==b.session_id:
                    hm = r.get("hopper")
                    if hm in ("light","dark"):
                        used = hm
                    break
            except:
                pass
    if used is None and CONTROLLER.status and isinstance(CONTROLLER.status.message, str):
        if "Using light hopper" in CONTROLLER.status.message:
            used = "light"
        elif "Using dark hopper" in CONTROLLER.status.message:
            used = "dark"
    if used in ("light","dark"):
        prefs = _hopper_load()
        delta = 1 if b.liked else -1
        key = "light_score" if used=="light" else "dark_score"
        prefs[key] = int(prefs.get(key, 0)) + delta
        _hopper_save(prefs)
    score = 8.5 if b.liked else 3.5
    bo_observe([0.0]*8, score)
    return {"ok": True}

@app.post("/train/feedback")
def api_train_feedback():
    return train_from_feedback()

@app.post("/bo/suggest")
def api_bo_suggest(b: BOSuggestBody):
    return {"ok": True, "suggested_brew_feat": bo_suggest(b.n_candidates)}

@app.post("/bo/observe")
def api_bo_observe(b: BOObserveBody):
    if len(b.brew_feat)!=8: return {"ok": False, "error":"brew_feat must have length 8"}
    bo_observe(b.brew_feat, b.score)
    return {"ok": True}

def _open_browser():
    try: webbrowser.open("http://127.0.0.1:8000")
    except: pass

if __name__ == "__main__":
    import uvicorn
    print("Starting server at http://127.0.0.1:8000")
    threading.Timer(1.0, _open_browser).start()
    uvicorn.run("coffee_machine_app_fixed:app", host="127.0.0.1", port=8000, reload=False)
