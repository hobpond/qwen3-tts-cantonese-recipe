#!/usr/bin/env python3
"""The standard page's shared parts: its CSS, the pitch-and-energy panel (drawn in the page by the vendored pitchy,
fft.js and Meyda libraries, embedded byte for byte from external/js/), the copy button, and the contour and
calibration that feed the panel.

From the lab's build_pool_page.py, the particle-pool page that every later listening page was built on (owner,
2026-10-11: "Listen test should follow the same format, hot keys, and visualization as the other skill, build on
it"). Only what build_ab_page.py reads is kept; the pool page itself, its layout, controls and harvest are not.

THE LIBRARIES ARE PART OF THE INSTRUMENT. vendored_bundle() refuses on any sha256 mismatch against
external/js/PINNED.json: a changed byte in the panel's pitch tracker is a changed instrument. MIT only (pitchfinder,
GPL-3.0, and Essentia.js, AGPL-3.0, were excluded on licence in the lab).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]          # src/ -> the repository

TAIL = "。？！?!"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def core_of(text: str) -> str:
    return text.rstrip(TAIL)


def particle_index(text: str) -> int:
    """The particle's index in the displayed text, by code point, which is how Python and
    the page's Array.from both count. A mark is allowed only below it."""
    return len(core_of(text)) - 1


CSS = """
:root{--bg:#12100e;--fg:#efe9e1;--dim:#8b8177;--acc:#c98a4b;--bad:#a8524a;
--line:#262019;--panel:#191612;--ok:#5a8a5f;--tone:#b8863a;--miss:#6b4f8f;--unk:#55606b}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--fg);font:16px/1.55 system-ui,sans-serif;margin:0;
padding:22px 16px 150px;display:flex;justify-content:center}
.w{width:100%;max-width:760px}
h1{font-size:20px;margin:0 0 4px}
p.sub{color:var(--dim);margin:0 0 12px;font-size:14px}
p.ckpt{color:var(--dim);margin:0 0 12px;font-size:13px}
p.ckpt b{color:var(--fg);font:600 13px ui-monospace,Menlo,monospace;background:#221d18;
border:1px solid #322a22;border-radius:6px;padding:1px 7px}
.legend{display:flex;gap:12px;flex-wrap:wrap;font-size:12px;color:var(--dim);margin:0 0 6px;align-items:center}
.legend b{font-weight:600;padding:1px 6px;border-radius:4px;color:#fff}
.legend b.p{background:var(--bad)} .legend b.t{background:var(--tone)} .legend b.e{background:#4a6f9f}
.legend b.g{background:#1e3324;color:var(--fg);box-shadow:inset 0 -3px 0 var(--ok)}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:16px 18px;margin:8px 0 14px}
.pos{font:600 12px/1 ui-monospace,Menlo,monospace;letter-spacing:.12em;color:var(--acc);margin-bottom:8px}
.txt{font-size:30px;line-height:1.7;letter-spacing:.04em;margin:4px 0 10px;word-break:break-all;
-webkit-user-select:none;user-select:none}
.txt .pt{background:#3a2a12;box-shadow:inset 0 -3px 0 var(--acc);border-radius:4px;padding:0 3px}
/* PARTBASE1's marks: three fault kinds in three colours, and praise underlined, not filled,
   so a good mark never reads at a glance as damage. */
.txt span.c{cursor:pointer;border-radius:4px;padding:0 1px}
.txt span.c:hover{outline:1px solid var(--acc)}
.txt span.mk-p{background:var(--bad);color:#fff} .txt span.mk-t{background:var(--tone);color:#fff}
.txt span.mk-e{background:#4a6f9f;color:#fff}
.txt span.mk-gp,.txt span.mk-gt,.txt span.mk-ge{background:#1e3324;box-shadow:inset 0 -3px 0 var(--ok)}
.txt span.mk-gt{box-shadow:inset 0 -3px 0 var(--tone)} .txt span.mk-ge{box-shadow:inset 0 -3px 0 #6f9fd8}
audio{width:100%;height:34px;margin:4px 0 10px}
.btns{display:flex;flex-wrap:wrap;gap:8px}
button.v{font:inherit;font-size:14px;padding:8px 12px;border-radius:8px;border:1px solid #322a22;
background:#221d18;color:var(--fg);cursor:pointer}
button.v:hover{border-color:var(--acc)}
button.v kbd{font:600 11px ui-monospace,monospace;color:var(--dim);margin-right:6px}
button.v.on{color:#fff;border-color:transparent}
button.v.on[data-v=c]{background:var(--ok)} button.v.on[data-v=p]{background:var(--bad)}
button.v.on[data-v=t]{background:var(--tone)} button.v.on[data-v=m]{background:var(--miss)}
button.v.on[data-v=u]{background:var(--unk)}
.grid{display:flex;flex-wrap:wrap;gap:3px;margin:6px 0 10px}
.grid span{width:14px;height:14px;border-radius:3px;background:#2a231c;cursor:pointer}
.grid span.cur{outline:2px solid var(--acc)}
.grid span[data-v=c]{background:var(--ok)} .grid span[data-v=p]{background:var(--bad)}
.grid span[data-v=t]{background:var(--tone)} .grid span[data-v=m]{background:var(--miss)}
.grid span[data-v=u]{background:var(--unk)}
.grid span.m{box-shadow:inset 0 0 0 2px var(--fg)}
.dock{position:fixed;left:0;right:0;bottom:0;background:#0d0b09f2;border-top:1px solid var(--line);
padding:10px 16px;backdrop-filter:blur(8px)}
.dockin{max-width:760px;margin:0 auto;display:flex;gap:10px;align-items:center}
textarea{flex:1;min-width:0;height:52px;background:#0d0b09;color:var(--fg);border:1px solid #2a231c;
border-radius:8px;padding:8px 10px;font:12px/1.4 ui-monospace,monospace}
.count{color:var(--dim);font-size:13px;white-space:nowrap;font-variant-numeric:tabular-nums}
button.big{padding:9px 12px;font:inherit;font-size:13px;border-radius:8px;border:1px solid #322a22;
background:#221d18;color:var(--fg);cursor:pointer;white-space:nowrap}
#saved{font-size:12px}
"""

JS = r"""
const SESS=__SESS__, N=__N__, T=__T__, S=__S__, A=__A__;
const V={}, M={};
let cur=0;
// A NEW STORE SINCE THE MARKS (2026-10-02). A session saved by a page without them must not
// restore into this one: its clips were judged with no way to mark another character, and
// their empty mark lists would read as "nothing else was wrong".
const STORE="partpool-m:"+SESS;
const au=document.getElementById('au');
function save(){
  try{ localStorage.setItem(STORE, JSON.stringify({v:V,m:M,t:new Date().toISOString()}));
    document.getElementById('saved').textContent="Saved in this browser at "+
      new Date().toLocaleTimeString()+". Reopening this file restores it; the code is a copy, not the record.";
  }catch(e){ document.getElementById('saved').textContent=
    "COULD NOT SAVE LOCALLY ("+e+"). Copy the code before closing this tab."; }
}
function restore(){
  let d=null; try{ d=JSON.parse(localStorage.getItem(STORE)||"null"); }catch(e){ return; }
  if(!d) return;
  Object.assign(V, d.v||{}); Object.assign(M, d.m||{});
  document.getElementById('saved').textContent="Restored a session saved at "+(d.t||"an earlier time")+".";
}
function faulted(i){ const o=M[i]; return !!o && Object.values(o).some(k=>k.charAt(0)!=='+'); }
function emit(){
  const judged=Object.keys(V).map(Number).sort((a,b)=>a-b);
  const items=judged.map(i=>i+":"+V[i]);
  // A clip's marks travel with its verdict: a clip still being marked is not yet judged.
  // An unsuffixed mark is a wrong sound, as on PARTBASE1.
  const marks=[];
  judged.forEach(i=>{ const o=M[i]||{};
    Object.keys(o).map(Number).sort((a,b)=>a-b).forEach(c=>{
      const k=o[c]; marks.push(i+":"+c+(k==="p"?"":":"+k)); }); });
  document.getElementById('code').value=SESS+"|V="+items.join(",")+"|M="+marks.join(",");
  const faults=judged.filter(i=>"ptm".includes(V[i])).length;
  const marked=judged.filter(i=>faulted(i)).length;
  document.getElementById('cnt').textContent=items.length+"/"+N+" judged, "+faults+
    " particle faults, "+marked+" with other marks";
  save();
}
function grid(){
  const g=document.getElementById('grid'); g.innerHTML="";
  for(let i=0;i<N;i++){ const s=document.createElement('span');
    if(V[i]) s.dataset.v=V[i];
    if(i===cur) s.classList.add('cur');
    if(faulted(i)) s.classList.add('m');
    s.title="clip "+(i+1); s.onclick=()=>show(i,true); g.appendChild(s); }
}
function play(){
  const go=()=>{ try{ au.currentTime=S[cur]; }catch(e){} const p=au.play(); if(p) p.catch(()=>{}); };
  if(au.readyState>=1) go(); else au.addEventListener('loadedmetadata', go, {once:true});
}
// PARTBASE1's cycle (skills/blind-ab-testing/build_ab_marking_page.py): a click cycles the
// fault kinds, a shift-click the same kinds as notably good. Switching sign starts that
// sign's cycle, so a shift-click on a red mark turns it green instead of clearing it.
const ORDER=['','p','t','e'];
function cls(k){ return 'mk-'+(k.charAt(0)==='+'?'g'+k.slice(1):k); }
function mark(el,ci,ev){
  const i=cur; M[i]=M[i]||{};
  const good=!!(ev&&ev.shiftKey);
  const c=M[i][ci]||'', cGood=c.charAt(0)==='+';
  const base=(good===cGood)?(cGood?c.slice(1):c):'';
  const nxt=ORDER[(ORDER.indexOf(base)+1)%ORDER.length];
  el.classList.remove('mk-p','mk-t','mk-e','mk-gp','mk-gt','mk-ge');
  if(nxt){ M[i][ci]=(good?'+':'')+nxt; el.classList.add(cls(M[i][ci])); }
  else{ delete M[i][ci]; if(!Object.keys(M[i]).length) delete M[i]; }
  emit(); grid();
}
function show(i, autoplay){
  cur=i; const t=T[i], o=M[i]||{};
  document.getElementById('n').textContent=(i+1)+" of "+N;
  const tx=document.getElementById('txt'); tx.innerHTML="";
  // By code point, as Python indexes the text: a character outside the BMP is one mark.
  Array.from(t[0]).forEach((ch,ci)=>{
    const s=document.createElement('span'); s.className='c'; s.textContent=ch;
    if(o[ci]) s.classList.add(cls(o[ci]));
    s.addEventListener('click', ev=>mark(s,ci,ev)); tx.appendChild(s); });
  const pt=document.createElement('span'); pt.className='pt'; pt.textContent=t[1]; tx.appendChild(pt);
  tx.appendChild(document.createTextNode(t[2]));
  au.src=A[i];
  document.querySelectorAll('button.v[data-v]').forEach(b=>b.classList.toggle('on', V[i]===b.dataset.v));
  grid();
  if(autoplay) play();
}
function nextUnjudged(i){
  for(let k=1;k<=N;k++){ const j=(i+k)%N; if(!V[j]) return j; }
  return -1;
}
function judge(v){
  V[cur]=v; emit();
  const j=nextUnjudged(cur);
  if(j<0){ show(cur,false); document.getElementById('done').textContent=
    "Every clip has a verdict. Copy the code below and send it."; au.pause(); return; }
  show(j,true);
}
function copyCode(btn){
  const ta=document.getElementById('code');
  const say=(t)=>{ btn.textContent=t; setTimeout(()=>btn.textContent="Copy code",4000); };
  ta.removeAttribute('readonly'); ta.select(); ta.setSelectionRange(0,999999);
  let ok=false; try{ ok=document.execCommand('copy'); }catch(e){ ok=false; }
  ta.setAttribute('readonly','');
  if(ok){ say("Copied ✓"); return; }
  if(navigator.clipboard && window.isSecureContext){
    navigator.clipboard.writeText(ta.value).then(()=>say("Copied ✓")).catch(()=>say("COPY FAILED — press Ctrl+C"));
    return; }
  ta.focus(); ta.select(); say("COPY FAILED — the text is selected, press Ctrl+C");
}
document.addEventListener('keydown', e=>{
  if(e.target && e.target.tagName==='TEXTAREA') return;
  if(e.ctrlKey||e.metaKey||e.altKey) return;
  const k=e.key;
  const map={"1":"c","2":"p","3":"t","4":"m","5":"u"};
  if(map[k]){ e.preventDefault(); judge(map[k]); }
  else if(k===" "||e.code==="Space"){ e.preventDefault(); play(); }
  else if(k==="ArrowRight"){ e.preventDefault(); show(Math.min(N-1,cur+1),true); }
  else if(k==="ArrowLeft"){ e.preventDefault(); show(Math.max(0,cur-1),true); }
});
document.querySelectorAll('button.v[data-v]').forEach(b=>b.onclick=()=>judge(b.dataset.v));
restore(); emit();
show(Math.max(0,nextUnjudged(N-1)),false);
"""

VIZ_HOP_S = 0.02
# PANEL V2 (owner, 2026-10-10: "add those in ... Make sure we are not too cluttered though so explinations and
# glossery / legend should be pop over or other forms of out-band display"). Two additions: the written
# particle's tone target, from pycantonese's Jyutping in context, drawn on a 1-5 tone scale normalised to this
# page's speaker; and a spectrogram strip. The bar holds two buttons and nothing else; every explanation is in
# a pop-over. PARTPOOL24 went out with v1 (pitch and energy only) and keeps it.
VIZ_VERSION = 4
# PANEL V4 (owner's screenshot, 2026-10-10): the particle band sits at the end of the MAIN speech, before any pause of
# 0.4 s or more, and a detached sound after the sentence is marked on the panel. Up to v3 the band followed the
# render's recorded speech end, which lands on such a sound (55 of 57 clips that have one).
# PANEL V3 (owner, 2026-10-10: "Do it. We don't have to be purist, as long as the process is repeatable, we can
# blend both python at build time and embedded library (at load/run time)"). Two pinned MIT libraries are embedded
# and compute LIVE layers from the embedded audio, beside the RECORDED build-time layers, which stay the
# measurement. Each verdict's S entry carries the live layers that were on (pos:letters), so what was visible at
# a verdict is on record.
LIVE_LAYERS = {"m": "second pitch tracker (pitchy, McLeod method)", "l": "loudness (Meyda)",
               "c": "brightness: spectral centroid (Meyda)", "f": "noisiness: spectral flatness (Meyda)"}
JS_LIBS = REPO / "external/js"


def vendored_bundle(root: Path | None = None) -> str:
    """pitchy + fft.js + Meyda as one classic script, from the files pinned in external/js/PINNED.json.

    REFUSES on any sha256 mismatch: the libraries are part of the instrument, and a changed byte is a changed
    instrument. pitchy is an ES module importing fft.js (CommonJS), so the wrap is textual and checked: inline
    fft.js's module.exports, drop pitchy's one import and its two `export` keywords, and expose only
    PitchDetector. Meyda's browser build is UMD and sets globalThis.Meyda itself."""
    import re
    root = root or JS_LIBS
    pin = json.loads((root / "PINNED.json").read_text(encoding="utf-8"))
    dirs = {}
    for name, pk in pin["packages"].items():
        d = root / f"{name}-{pk['version']}"
        for f, meta in pk["files"].items():
            got = sha256_file(d / f)
            if got != meta["sha256"]:
                raise SystemExit(f"{d / f}: sha256 {got[:16]} is not the pinned {meta['sha256'][:16]}")
        dirs[name] = d
    fft = (dirs["fft.js"] / "fft.js").read_text(encoding="utf-8")
    pit = (dirs["pitchy"] / "index.js").read_text(encoding="utf-8")
    imp = 'import FFT from "fft.js";\n'
    if pit.count(imp) != 1 or pit.count("export class ") != 2 or fft.count("module.exports = FFT;") != 1:
        raise SystemExit("the vendored pitchy or fft.js no longer has the shape the wrap expects")
    body = pit.replace(imp, "").replace("export class ", "class ")
    if re.search(r"^\s*export\b", body, re.M):
        raise SystemExit("pitchy still exports something after the wrap")
    mey = (dirs["meyda"] / "meyda.min.js").read_text(encoding="utf-8")
    for src in (fft, body, mey):
        if "</script" in src.lower():
            raise SystemExit("a vendored library contains </script")
    return ("(function(){var module={exports:{}},exports=module.exports;\n" + fft + "\nvar FFT=module.exports;\n"
            + body + "\nglobalThis.PitchDetector=PitchDetector;})();\n" + mey + "\n")


SPEC_BINS = 48
# Tone number -> (start, end) on the 1-5 scale, the standard values for Hong Kong Cantonese.
TONE_TARGETS = {"1": (5, 5), "2": (3, 5), "3": (3, 3), "4": (2, 1), "5": (2, 3), "6": (2, 2)}
TONE_NAMES = {"1": "high level", "2": "high rising", "3": "mid level", "4": "low falling", "5": "low rising",
              "6": "low level"}
VIZ_CSS = """
.vizbar{display:flex;align-items:center;gap:8px;margin:0 0 8px}
.vizbar button{font:inherit;font-size:13px;padding:6px 10px;border-radius:8px;border:1px solid #322a22;
background:#221d18;color:var(--fg);cursor:pointer}
.vizbar button.on{border-color:var(--acc);color:var(--acc)}
.vizbar button.q{width:30px;padding:6px 0;font-weight:600}
#vz{width:100%;height:240px;display:block;background:#0f0d0b;border:1px solid var(--line);border-radius:8px;
margin:0 0 10px}
#vz[hidden]{display:none}
dialog#vzhelp{max-width:560px;background:var(--panel);color:var(--fg);border:1px solid var(--line);border-radius:12px;
padding:18px 20px;font-size:14px;line-height:1.5}
dialog#vzhelp::backdrop{background:#000a}
dialog#vzhelp h2{font-size:15px;margin:14px 0 6px;color:var(--acc)} dialog#vzhelp h2:first-child{margin-top:0}
dialog#vzhelp table{border-collapse:collapse;font-size:13px;width:100%}
dialog#vzhelp td,dialog#vzhelp th{padding:3px 8px;border-bottom:1px solid var(--line);text-align:left}
dialog#vzhelp .sw{display:inline-block;width:22px;height:4px;border-radius:2px;vertical-align:middle;margin-right:6px}
dialog#vzhelp button{font:inherit;font-size:13px;padding:6px 12px;border-radius:8px;border:1px solid #322a22;
background:#221d18;color:var(--fg);cursor:pointer;float:right}
#vzclip{font:13px ui-monospace,monospace;color:var(--fg);background:#221d18;border-radius:6px;padding:4px 8px}
"""
VIZ_HTML = ("<div class=vizbar><button id=vzb onclick='toggleViz()' title='V'>Show pitch &amp; energy</button>"
            "<button id=vzl onclick='layersViz()' title='Live layers'>Layers</button>"
            "<button class=q onclick='helpViz()' title='What am I looking at? (?)'>?</button></div>"
            "<div id=vzlay hidden></div>"
            "<canvas id=vz hidden></canvas>"
            "<dialog id=vzhelp><button onclick='this.closest(\"dialog\").close()'>Close</button>"
            "<h2>This clip</h2><div id=vzclip></div>"
            "<h2>What the panel shows</h2>"
            "<p><span class=sw style='background:#c98a4b'></span><b>Top strip, spectrogram:</b> energy by frequency, low at "
            "the bottom. Bright bands are vowel resonances; a sharp stop at a syllable's end is a -p, -t or -k coda.</p>"
            "<p><span class=sw style='background:#7fb3d5'></span><b>Blue line, pitch,</b> on the 1-5 tone scale, where 1 is "
            "this speaker's low and 5 their high across the whole page. Gaps are where no pitch was found.</p>"
            "<p><span class=sw style='background:rgba(201,138,75,.45)'></span><b>Shaded area, energy,</b> loudness over time.</p>"
            "<p><span class=sw style='background:rgba(90,138,95,.6)'></span><b>Green band, the particle's target:</b> the "
            "contour the written particle's tone should follow, over the clip's last voiced syllable.</p>"
            "<p><span class=sw style='background:#efe9e1'></span><b>White line:</b> where playback is.</p>"
            "<p><span class=sw style='background:rgba(168,82,74,.6)'></span><b>Red span, a detached sound:</b> the model "
            "fell silent after the sentence and then made a short sound before stopping. It is not the particle; the green "
            "band stays on the sentence's last syllable.</p>"
            "<h2>Live layers (Layers button)</h2>"
            "<p>Computed in your browser from this clip's audio, by two embedded, pinned libraries (pitchy and Meyda, both "
            "MIT). The layers above were measured when the page was built and are the record; these are a second look. "
            "Which ones are on is saved with each verdict.</p>"
            "<p><span class=sw style='background:#e3c26b'></span><b>Dashed gold, second pitch tracker:</b> McLeod's method, "
            "on the same 1-5 scale. Where it disagrees with the blue line, one of them is wrong.</p>"
            "<p><span class=sw style='background:#efe9e1;height:2px'></span><b>Thin white in the spectrogram, brightness:</b> "
            "the spectral centroid. Higher means brighter or more fricative.</p>"
            "<p><span class=sw style='background:#d58fb0'></span><b>Pink, loudness:</b> perceptual loudness, scaled to the clip.</p>"
            "<p><span class=sw style='background:#9fbf8f'></span><b>Grey-green, noisiness:</b> spectral flatness, low for a "
            "clear tone, high for breath, hiss or creak.</p>"
            "<h2>Cantonese tones</h2><table><tr><th>Tone</th><th>Name</th><th>Contour (1-5)</th><th>Particle example</th></tr>"
            "<tr><td>1</td><td>high level</td><td>55</td><td>啦 laa1, 咩 me1, 啫 ze1</td></tr>"
            "<tr><td>2</td><td>high rising</td><td>35</td><td>&nbsp;</td></tr>"
            "<tr><td>3</td><td>mid level</td><td>33</td><td>喇 laa3, 啊 aa3, 㗎 gaa3 / ga3</td></tr>"
            "<tr><td>4</td><td>low falling</td><td>21</td><td>&nbsp;</td></tr>"
            "<tr><td>5</td><td>low rising</td><td>23</td><td>&nbsp;</td></tr>"
            "<tr><td>6</td><td>low level</td><td>22</td><td>&nbsp;</td></tr></table>"
            "<p>Syllables ending in -p, -t or -k carry short versions of tones 1, 3 and 6.</p>"
            "<h2>Reading it with care</h2>"
            "<p>The target is a dictionary reading, and particles can take more than one (呀 may be aa3 or aa4). The band's "
            "timing is the last voiced stretch, not an alignment. Pitch tracking can jump an octave on creaky or quiet "
            "ends. Your ear decides; the panel is there to look at.</p>"
            "<h2>Keys</h2><p><b>V</b> shows or hides the panel for every clip (hidden is the default). <b>?</b> opens "
            "this. <b>Esc</b> closes it.</p></dialog>")
VIZ_CSS += """
.vizbar button#vzl.on{border-color:#e3c26b;color:#e3c26b}
#vzlay{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:8px 12px;margin:0 0 8px;
font-size:13px;display:flex;flex-wrap:wrap;gap:6px 16px}
#vzlay[hidden]{display:none}
#vzlay label{cursor:pointer;color:var(--fg)} #vzlay small{color:var(--dim);width:100%}
"""
VIZ_JS = r"""
const VZ=__VZ__, VZC=__VZC__, VSTORE="partpool-viz:"+SESS, SSTORE="partpool-s:"+SESS, LSTORE="partpool-lay:"+SESS;
const LIVEL=VZC.layers||{}, LIVE={};
let VIS=false, SH={}, LAY="";
try{ VIS=localStorage.getItem(VSTORE)==="1"; SH=JSON.parse(localStorage.getItem(SSTORE)||"{}")||{};
  LAY=(localStorage.getItem(LSTORE)||"").split("").filter(k=>LIVEL[k]).sort().join(""); }catch(e){}
async function live(i){
  if(LIVE[i]) return; LIVE[i]="busy";
  try{
    const buf=await (await fetch(A[i])).arrayBuffer();
    const ab=await new OfflineAudioContext(1,1,24000).decodeAudioData(buf);
    const x=ab.getChannelData(0), sr=ab.sampleRate, N=2048, hop=Math.round(VZ[i].h*sr), n=VZ[i].f.length;
    const det=PitchDetector.forFloat32Array(N), fr=new Float32Array(N), o={m:[],l:[],c:[],f:[]};
    Meyda.bufferSize=N; Meyda.sampleRate=sr;
    for(let k=0;k<n;k++){ const s0=k*hop-N/2;
      for(let j=0;j<N;j++){ const q=s0+j; fr[j]=(q>=0&&q<x.length)?x[q]:0; }
      const [p,cl]=det.findPitch(fr,sr); o.m.push(cl>0.9&&p>65&&p<500?12*Math.log2(p/100):null);
      const e=Meyda.extract(["loudness","spectralCentroid","spectralFlatness"],fr)||{};
      const ok=v=>(typeof v==="number"&&isFinite(v))?v:null;
      o.l.push(ok(e.loudness&&e.loudness.total)); o.c.push(ok(e.spectralCentroid)===null?null:e.spectralCentroid*sr/N);
      o.f.push(ok(e.spectralFlatness)); }
    LIVE[i]=o;
  }catch(err){ LIVE[i]={err:String(err)}; }
}
function layersViz(){
  const el=document.getElementById('vzlay');
  if(!el.hidden){ el.hidden=true; return; }
  el.innerHTML=Object.entries(LIVEL).map(([k,t])=>"<label><input type=checkbox data-k="+k+(LAY.includes(k)?" checked":"")+"> "+t+"</label>").join("")
    +"<small>Computed live from this clip's audio; saved with each verdict. See ? for what each shows.</small>";
  el.querySelectorAll('input').forEach(b=>b.onchange=()=>{
    LAY=Array.from(el.querySelectorAll('input:checked')).map(q=>q.dataset.k).sort().join("");
    document.getElementById('vzl').classList.toggle('on',!!LAY);
    try{ localStorage.setItem(LSTORE,LAY); }catch(e){} });
  el.hidden=false;
}
const cv=document.getElementById('vz'), vb=document.getElementById('vzb'), SPEC={};
function spec(i){
  if(SPEC[i]!==undefined) return SPEC[i];
  const d=VZ[i]; if(!d.s){ SPEC[i]=null; return null; }
  const raw=atob(d.s), B=VZC.bins, n=raw.length/B, off=document.createElement('canvas');
  off.width=n; off.height=B; const g=off.getContext('2d'), im=g.createImageData(n,B);
  for(let t=0;t<n;t++) for(let b=0;b<B;b++){ const v=raw.charCodeAt(t*B+b), k=((B-1-b)*n+t)*4;
    im.data[k]=Math.min(255,v*1.1); im.data[k+1]=v*0.62; im.data[k+2]=v*0.32; im.data[k+3]=255; }
  g.putImageData(im,0,0); SPEC[i]=off; return off;
}
function paintViz(){
  if(!VIS) return;
  const d=VZ[cur], dpr=window.devicePixelRatio||1, W=cv.clientWidth, H=cv.clientHeight;
  cv.width=W*dpr; cv.height=H*dpr;
  const g=cv.getContext('2d'); g.setTransform(dpr,0,0,dpr,0,0); g.clearRect(0,0,W,H);
  const n=d.f.length, dur=n*d.h, x=t=>t/dur*W, SH0=78, top=SH0+8, PH=H-top-6;
  const sp=spec(cur); if(sp){ g.imageSmoothingEnabled=true; g.drawImage(sp,0,0,W,SH0); }
  // energy behind the pitch, -60 dB to the clip's peak
  const peak=Math.max(...d.e), lo=peak-60, ey=v=>top+PH-((Math.max(v,lo)-lo)/60)*(PH*0.55);
  g.beginPath(); g.moveTo(0,top+PH); d.e.forEach((v,i)=>g.lineTo(x(i*d.h),ey(v))); g.lineTo(W,top+PH); g.closePath();
  g.fillStyle="rgba(201,138,75,0.16)"; g.fill();
  // the 1-5 tone scale for this page's speaker
  const ch=st=>1+4*(st-VZC.lo)/(VZC.hi-VZC.lo), py=c=>top+PH-((Math.max(0.5,Math.min(5.5,c))-0.5)/5)*PH;
  g.font="10px system-ui"; g.fillStyle="#5d554c";
  for(let c=1;c<=5;c++){ g.strokeStyle="#211c17"; g.lineWidth=1; g.beginPath(); g.moveTo(14,py(c)); g.lineTo(W,py(c)); g.stroke();
    g.fillText(String(c),3,py(c)+3); }
  // the written particle's target over the last voiced syllable
  if(d.pt){ const [t0,t1,tone,jy]=d.pt, tg=VZC.targets[tone];
    if(tg){ g.fillStyle="rgba(90,138,95,0.30)"; g.beginPath();
      g.moveTo(x(t0),py(tg[0]+0.45)); g.lineTo(x(t1),py(tg[1]+0.45)); g.lineTo(x(t1),py(tg[1]-0.45)); g.lineTo(x(t0),py(tg[0]-0.45));
      g.closePath(); g.fill(); g.fillStyle="#8fbf94"; g.font="11px ui-monospace,monospace";
      g.fillText(jy, Math.max(16, Math.min(W-60, x(t0))), py(Math.max(tg[0],tg[1])+0.75)); } }
  // a detached sound after the sentence (v4): a hatched red span, so it is seen as the model's, not the particle
  if(d.tail){ const [a,b]=d.tail, xa=x(a)-3, xb=Math.max(x(b)+3, xa+8);
    g.fillStyle="rgba(168,82,74,0.25)"; g.fillRect(xa,top,xb-xa,PH); g.fillStyle="#c9786f"; g.font="10px system-ui";
    g.fillText("detached", Math.min(W-52, xa-2), top+11); }
  // pitch
  g.strokeStyle="#7fb3d5"; g.lineWidth=2.2; g.beginPath(); let on=false;
  d.f.forEach((v,i)=>{ if(v===null){ on=false; return; } const X=x(i*d.h), Y=py(ch(v));
    if(on) g.lineTo(X,Y); else g.moveTo(X,Y); on=true; });
  g.stroke();
  if(LAY){ live(cur); const L=LIVE[cur];
    if(L && L.m){
      const line=(arr,f,col,w,dash)=>{ g.strokeStyle=col; g.lineWidth=w; g.setLineDash(dash||[]); g.beginPath(); let on=false;
        arr.forEach((v,i)=>{ if(v===null){ on=false; return; } const X=x(i*d.h), Y=f(v); if(on) g.lineTo(X,Y); else g.moveTo(X,Y); on=true; });
        g.stroke(); g.setLineDash([]); };
      const span=a=>{ const v=a.filter(q=>q!==null); return v.length?[Math.min(...v),Math.max(...v)]:[0,1]; };
      if(LAY.includes("m")) line(L.m, v=>py(ch(v)), "#e3c26b", 1.8, [5,4]);
      if(LAY.includes("l")){ const [a,b]=span(L.l); line(L.l, v=>top+PH-((v-a)/(b-a||1))*PH*0.45, "#d58fb0", 1.3); }
      if(LAY.includes("f")){ line(L.f, v=>top+PH-Math.min(1,v*4)*PH*0.45, "#9fbf8f", 1.3); }
      if(LAY.includes("c")){ const mel=hz=>2595*Math.log10(1+hz/700), top8=mel(8000);
        line(L.c, v=>SH0-Math.min(1,mel(v)/top8)*SH0, "rgba(239,233,225,0.85)", 1.2); }
    } else if(L && L.err){ g.fillStyle="#a8524a"; g.font="11px system-ui"; g.fillText("live layers failed: "+L.err.slice(0,60),20,top+12); }
  }
  const t=au.currentTime||0; g.strokeStyle="#efe9e1"; g.lineWidth=1.5;
  g.beginPath(); g.moveTo(x(t),0); g.lineTo(x(t),H); g.stroke();
}
function loop(){ paintViz(); requestAnimationFrame(loop); }
function setViz(v){
  VIS=v; cv.hidden=!v; vb.classList.toggle('on',v);
  vb.textContent=v?"Hide pitch & energy":"Show pitch & energy";
  try{ localStorage.setItem(VSTORE, v?"1":"0"); }catch(e){}
}
function toggleViz(){ setViz(!VIS); }
function helpViz(){
  const d=VZ[cur], el=document.getElementById('vzclip');
  el.textContent=(d.jp||"(no Jyutping)")+(d.pt?"  ·  particle "+d.pt[3]+", "+(VZC.names[d.pt[2]]||"")+" ("+(VZC.targets[d.pt[2]]||[]).join("")+")":"");
  document.getElementById('vzhelp').showModal();
}
const _emit=emit, _judge=judge;
emit=function(){
  _emit();
  const shown=Object.keys(SH).map(Number).filter(i=>V[i]).sort((a,b)=>a-b);
  // A view is the live layers on at that verdict; none is just the position (the panel's recorded layers only).
  document.getElementById('code').value+="|S="+shown.map(i=>(typeof SH[i]==="string"&&SH[i])?i+":"+SH[i]:String(i)).join(",");
  try{ localStorage.setItem(SSTORE, JSON.stringify(SH)); }catch(e){}
};
judge=function(v){ if(VIS) SH[cur]=LAY; else delete SH[cur]; _judge(v); };
document.addEventListener('keydown', e=>{
  if(e.target && e.target.tagName==='TEXTAREA') return;
  if(document.getElementById('vzhelp').open){ if(e.key!=="Escape") e.stopImmediatePropagation(); return; }
  if(e.ctrlKey||e.metaKey||e.altKey) return;
  if(e.key==="v"||e.key==="V"){ e.preventDefault(); toggleViz(); }
  else if(e.key==="?"){ e.preventDefault(); helpViz(); }
}, true);
setViz(VIS); document.getElementById('vzl').classList.toggle('on',!!LAY); emit(); loop();
"""


# SHARED WITH build_ab_page.py (W44, 2026-10-11; owner: "Listen test should follow the same format, hot keys, and
# visualization as the other skill, build on it"). The A/B and audition pages draw the panel with these functions and
# copy their code with this one, cut from the strings above rather than copied, so a change here reaches both kinds of
# page. Cutting a string changes no byte of any page this file writes; `pages.py seam` proves it after any edit here.
def js_cut(src: str, start: str, end: str) -> str:
    """The text of `src` from `start` up to, not including, `end`. Each marker must occur exactly once, in order."""
    if src.count(start) != 1 or src.count(end) != 1 or src.index(start) > src.index(end):
        raise SystemExit(f"the shared page script no longer has {start!r} once before {end!r}")
    return src[src.index(start):src.index(end)]


# live() and layersViz(), then spec() and paintViz(). paintViz reads the globals cur, cv, au, VIS, VZ, VZC, LAY.
VIZ_DRAW_JS = (js_cut(VIZ_JS, "async function live(i){", "const cv=document.getElementById('vz')")
               + js_cut(VIZ_JS, "function spec(i){", "function loop(){"))
COPY_JS = js_cut(JS, "function copyCode(btn){", "document.addEventListener('keydown'")
# The help pop-over's live-layer section, which says nothing about particles.
VIZ_HELP_LIVE = js_cut(VIZ_HTML, "<h2>Live layers (Layers button)</h2>", "<h2>Cantonese tones</h2>")


def particle_jyutping(text: str, index: int) -> str | None:
    """The written particle's Jyutping in context, from pycantonese, by the character's index; None when the
    dictionary's syllables do not line up with the characters."""
    import pycantonese
    chars, sylls = [], []
    for word, jp in pycantonese.characters_to_jyutping(text):
        if jp is None:
            chars += list(word)
            sylls += [None] * len(word)
            continue
        parts = jp.split()
        if len(parts) != len(word):
            return None
        chars += list(word)
        sylls += parts
    return sylls[index] if index < len(sylls) and chars[index] == text[index] else None


def sentence_jyutping(text: str) -> str:
    import pycantonese
    return " ".join(jp for _, jp in pycantonese.characters_to_jyutping(text) if jp)


def contour(flac: Path, text: str | None = None, speech: tuple[float, float] | None = None,
            hop_s: float = VIZ_HOP_S) -> dict:
    """The panel's data for one clip, over the whole file the page plays, every hop_s seconds: pitch in
    semitones from 100 Hz (None where unvoiced), so a page's clips share one scale; energy in dB; a mel
    spectrogram as SPEC_BINS bytes a frame; the written particle's span and tone; the sentence's Jyutping."""
    import base64
    import librosa
    import numpy as np
    import soundfile as sf
    y, sr = sf.read(flac, dtype="float32")
    y = y.mean(axis=1) if y.ndim > 1 else y
    hop = int(round(hop_s * sr))
    f0, voiced, _ = librosa.pyin(y, fmin=65, fmax=500, sr=sr, frame_length=2048, hop_length=hop)
    ok = voiced & np.isfinite(f0)
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop)[0]
    mel = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=hop,
                                                             n_mels=SPEC_BINS, fmax=8000), ref=np.max)
    n = min(len(f0), len(rms), mel.shape[1])
    spec = np.clip((mel[:, :n] + 70.0) / 70.0 * 255.0, 0, 255).astype(np.uint8).T     # frame-major
    out = {"h": hop_s,
           "f": [round(float(12 * np.log2(f0[i] / 100.0)), 2) if ok[i] else None for i in range(n)],
           "e": [round(float(20 * np.log10(rms[i] + 1e-6)), 1) for i in range(n)],
           "s": base64.b64encode(spec.tobytes()).decode("ascii")}
    if text is not None:
        out["jp"] = sentence_jyutping(text)
        idx = particle_index(text)
        jy = particle_jyutping(text, idx)
        if jy and jy[-1] in TONE_TARGETS and speech:
            import build_pairs
            tail = build_pairs.detached_tail_of(y, sr)
            end_s = tail["main_end_s"] if tail else speech[0] + speech[1]
            if tail:
                out["tail"] = [tail["tail_start_s"], tail["tail_end_s"]]
            end = min(n - 1, int(end_s / hop_s))
            k = end
            while k > 0 and not ok[k]:
                k -= 1
            last, gap = k, 0
            while k > 0 and gap <= 2:
                k -= 1
                gap = 0 if ok[k] else gap + 1
            start = max(k + gap, last - int(0.4 / hop_s))
            if last - start >= int(0.08 / hop_s):
                out["pt"] = [round(start * hop_s, 3), round((last + 1) * hop_s, 3), jy[-1], jy]
    return out


def calibration(clips: list[dict]) -> dict:
    """The page's 1-5 tone scale: 1 and 5 at the 5th and 95th percentiles of every voiced frame on the page,
    one speaker, so heights compare across clips."""
    import numpy as np
    v = np.array([x for c in clips for x in c["f"] if x is not None])
    return {"lo": round(float(np.percentile(v, 5)), 2), "hi": round(float(np.percentile(v, 95)), 2),
            "bins": SPEC_BINS, "targets": TONE_TARGETS, "names": TONE_NAMES, "version": VIZ_VERSION}
