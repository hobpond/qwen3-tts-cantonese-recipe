#!/usr/bin/env python3
"""The standard listening page (the lab's build_ab_page.py, pruned to what build_story_page.py uses): build_html, the
answer code's grammar and parse_code, per-character marks, notes, 1-5 ratings and the About dialog.

WHY THIS PAGE. The owner, 2026-10-11: "Listen test should follow the same format, hot keys, and visualization as the
other skill, build on it." A trial plays its voices in order (Space) or one at a time (a, b, c); verdict keys advance
at once; the pitch-and-energy panel is hidden by default (V toggles, ? opens help), and the |S= field records what
was showing at each verdict.

THE CODE.  <ID>:<one answer character per trial, in page order, '-' unanswered>|S=<pos>[:<layers>],...
  then |M= marks, |N= notes and |R= ratings where the page offers them (MARK_GRAMMAR, NOTES_GRAMMAR,
  RATINGS_GRAMMAR). parse_code(code, key) is the one reader; build_story_page.py round-trips a code through it
  before it writes a page.

PER-CHARACTER MARKS (owner, 2026-10-17: "For all listen test we need to be able to mark each words"). A click toggles
a CJK character as mispronounced on that side. The key gives each trial its cjk_len, never the sentence. Marks are
secondary and never decide a verdict, and a mark is a positive, never a clearance.

FLAC, never MP3: a lossy codec is entitled to alter exactly the high-frequency detail a listener is asked to mark.
The lab's A/B, audition and ship-gate page builders, their v1 rebuilds and their self-test harness are not here.
"""
from __future__ import annotations

import base64
import hashlib
import html
import io
import json
import re

import build_pool_page as bp  # noqa: E402

GAP_MS = 400                     # between two clips of a trial played in order

RESERVED_KEYS = {"v", "V", "?", " "}      # the panel's keys and Space

# PER-CHARACTER MARKS (owner, 2026-10-17: "For all listen test we need to be able to mark each words"). Opt-in per
# page (build_html's `marks`), so every page built before stays byte-identical. Under each player the trial's sentence
# is shown as chips; a CJK character toggles a mark for that side, anything else is shown and not markable. A mark is a
# POSITIVE, never a clearance: no page plants a mispronunciation, so an unmarked character is not certified right.
MARK_GRAMMAR = ("|M=<token>,...  (pages whose key says \"marks\": true, since 2026-10-17; required there and refused "
                "everywhere else): exactly one token per trial in page order, joined by ','. A token is '-' for no marks, "
                "or [A<i>[.<i>]...][B<i>[.<i>]...] with at least one side: A's indices first, then B's, each side at most "
                "once and never empty. <i> is a 0-based index among the trial's CJK characters only (punctuation, digits "
                "and Latin are shown and not counted), by code point, decimal with no leading zero, strictly ascending "
                "and below that trial's CJK length. A mark says the listener heard that character mispronounced on that "
                "side. Marks need no answer: a trial with marks and no answer is kept.")
CODE_GRAMMAR = ("<ID>:<one answer character per trial in page order, '-' unanswered>|S=<pos>[:<layers>],...[|M=<token>,...]  "
                "The part before |S= is the v1 page's code exactly. |S= is present on every page that offers the panel: "
                "the 0-based trials answered while it was showing, each with the live layers on at that verdict (m pitchy, "
                "l loudness, c brightness, f noisiness). Every S position must carry an answer.  " + MARK_GRAMMAR)
# Han ideographs: URO, Ext A, the compatibility block, and the supplementary planes' Ext B-I and compatibility
# supplement. One list, written into the page, so Python and the page count the same characters by construction.
CJK_RANGES = [(0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF), (0x20000, 0x2FA1F), (0x30000, 0x323AF)]


def is_cjk(ch: str) -> bool:
    c = ord(ch)
    return any(lo <= c <= hi for lo, hi in CJK_RANGES)


def cjk_chars(text: str) -> list[str]:
    """The markable characters of a sentence, in order; a mark's index is a position in this list."""
    return [c for c in text if is_cjk(c)]


V1 = {
    "BM-064-TAILS": {
        "title": "Stray sounds after speech (set T)",
        "question": "Is there a stray sound after the speech has ended (a click, breath-like blip or murmur, separate "
                    "from the words)?",
        "options": [("a", "A only"), ("b", "B only"), ("2", "Both"), ("n", "Neither"), ("?", "Can't tell")],
        # which sides each answer says carry the defect; None is no claim
        "sides": {"a": ["A"], "b": ["B"], "2": ["A", "B"], "n": [], "?": None},
        "arms": ("ORIG", "CLEAN"), "calibration": "page",
    },
    "BM-LANG-REPAIR": {
        "title": "Lost speech check (set R)",
        "question": "Does either version lose or cut off any speech (a syllable, part of the last word)?",
        "options": [("a", "A lost speech"), ("b", "B lost speech"), ("n", "Neither"), ("?", "Can't tell")],
        "sides": {"a": ["A"], "b": ["B"], "n": [], "?": None},
        "arms": ("ORIGINAL", "REPAIRED"), "calibration": "trial",
    },
}


def flac_bytes(y, sr) -> bytes:
    """As v1 wrote its planted clips: soundfile's FLAC at its default subtype (PCM_16)."""
    import soundfile as sf
    b = io.BytesIO()
    sf.write(b, y, sr, format="FLAC")
    return b.getvalue()


# ----------------------------------------------------------------- audio identity
FLAC_URI = re.compile(r"data:audio/flac;base64,([A-Za-z0-9+/=]+)")


def page_audio(page_html: str) -> list[bytes]:
    """Every inlined FLAC on a page, in page order."""
    return [base64.b64decode(m) for m in FLAC_URI.findall(page_html)]


# ----------------------------------------------------------------- the page
AB_CSS = """
p.qn{font-size:18px;margin:2px 0 10px}
p.qn2{font-size:15px;margin:12px 0 6px;color:var(--dim)}
p.say{font-size:26px;letter-spacing:.04em;margin:0 0 10px}
.grid span.part{background:#5a4a33}
.slot{display:flex;align-items:center;gap:10px;margin:4px 0}
.slot button.lab{font:600 15px ui-monospace,Menlo,monospace;min-width:46px;height:34px;border-radius:8px;
border:1px solid #322a22;background:#221d18;color:var(--fg);cursor:pointer}
.slot.playing button.lab{border-color:var(--acc);color:var(--acc);background:#2c2117}
.slot audio{flex:1;margin:0}
canvas.vz{width:100%;height:200px;display:block;background:#0f0d0b;border:1px solid var(--line);border-radius:8px;
margin:2px 0 10px}
canvas.vz[hidden]{display:none}
.btns{margin-top:10px}
button.v.on{background:var(--acc);color:#fff;border-color:transparent}
.grid span[data-v]{background:var(--acc)}
.grid span[data-v="?"],.grid span[data-v="u"]{background:var(--unk)}
.legend kbd{font:600 11px ui-monospace,monospace;background:#221d18;border:1px solid #322a22;border-radius:4px;
padding:0 5px;color:var(--fg)}
"""

# The marking page's chips use the pool page's own character classes (bp.CSS: .txt span.c clickable, .txt span.mk-p
# the wrong-sound fill), so a marked character looks the same on both kinds of page; this only sizes them as chips.
AB_MARK_CSS = """
.txt.mk{font-size:22px;line-height:1.9;margin:0 0 8px 56px}
.txt.mk span.c{background:#221d18;border:1px solid #322a22;margin:0 1px;padding:0 4px}
.txt.mk span.c.mk-p{background:var(--bad);border-color:var(--bad);color:#fff}
.txt.mk span.pu{color:var(--dim);padding:0 2px}
"""

# Spliced in at the __MK*__ markers of AB_JS on a marking page only; on any other page every marker is removed, so its
# script is the one it always was. Function declarations are hoisted, so emit/grid/show may call these.
AB_MARK_JS = r"""
// Per-character marks: MK[trial] = {A:[i,...], B:[i,...]}, i the index among the sentence's CJK characters only,
// by code point (Array.from), as the harvest counts them. Toggling is the only edit; it never touches an answer.
const CJKR=__CJKR__, MK={};
function isCJK(ch){ const c=ch.codePointAt(0); return CJKR.some(r=>c>=r[0]&&c<=r[1]); }
function mkHas(t,side,ci){ return ((MK[t]||{})[side]||[]).indexOf(ci)>=0; }
function mkToggle(t,side,ci){
  const o=MK[t]=MK[t]||{}, l=o[side]=o[side]||[], j=l.indexOf(ci);
  if(j<0){ l.push(ci); l.sort((x,y)=>x-y); } else l.splice(j,1);
  if(!l.length) delete o[side];
  if(!Object.keys(o).length) delete MK[t];
}
function mkToken(t){
  const o=MK[t]||{}, s=["A","B"].filter(sd=>o[sd]&&o[sd].length).map(sd=>sd+o[sd].join(".")).join("");
  return s||"-";
}
function mkCode(){ const a=[]; for(let i=0;i<N;i++) a.push(mkToken(i)); return a.join(","); }
function mkRestore(m){
  if(!m||typeof m!=="object") return;
  Object.keys(m).forEach(k=>{ const t=Number(k); if(!(Number.isInteger(t)&&t>=0&&t<N)) return;
    const n=Array.from(TX[t]||"").filter(isCJK).length;
    ["A","B"].forEach(sd=>{ ((m[k]||{})[sd]||[]).forEach(ci=>{
      if(Number.isInteger(ci)&&ci>=0&&ci<n&&!mkHas(t,sd,ci)) mkToggle(t,sd,ci); }); }); });
}
function chips(t,k){
  const side=T[t].l[k], d=document.createElement('div'); d.className='txt mk'; let n=0;
  Array.from(TX[t]||"").forEach(ch=>{
    const s=document.createElement('span'); s.textContent=ch;
    if(isCJK(ch)){ const ci=n++; s.className='c'; s.dataset.side=side; s.dataset.ci=String(ci);
      if(mkHas(t,side,ci)) s.classList.add('mk-p');
      s.title="Mispronounced in "+side+" (click again to clear)";
      s.addEventListener('click',()=>{ mkToggle(t,side,ci); s.classList.toggle('mk-p',mkHas(t,side,ci)); emit(); grid(); });
    } else s.className='pu';
    d.appendChild(s); });
  return d;
}"""
# marker -> what a marking page puts there
AB_MARK_SPLICE = {"__MKDECL__": AB_MARK_JS, "__MKSAVE__": ",mk:MK", "__MKREST__": " mkRestore(d.mk);",
                  "__MKEMIT__": '+"|M="+mkCode()', "__MKCNT__": '+", "+Object.keys(MK).length+" with marks"',
                  "__MKGRID__": " if(MK[i]) s.classList.add('m');", "__MKROW__": " box.appendChild(chips(t,k));"}
# A marking page keeps its answers in a store of its own, as the pool page did when it gained marks: answers saved by
# the page without marks must not restore here, where their empty mark lists would read as "nothing mispronounced".
AB_MARK_STORES = {'const STORE="abpage-v:"+SESS,': 'const STORE="abpage-vm:"+SESS,',
                  'SSTORE="abpage-s:"+SESS,': 'SSTORE="abpage-sm:"+SESS,'}

AB_JS = r"""
const SESS=__SESS__, N=__N__, A=__A__, S=__S__, T=__T__, OPT=__OPT__, MULTI=__MULTI__, GAP_MS=__GAP__;
const VZ=__VZ__, VZC=__VZC__, CAL=__CAL__;
const V={}, KEYMAP={}, SLOTK={}, XG=__XG__, W={}, TX=__TX__;__MKDECL__
// The main answer is V; a second answer group (XG, e.g. T) has its own keys and its own field in the code.
OPT.forEach(o=>KEYMAP[o[0]]=["",o[1]]);
XG.forEach(g=>{ W[g.n]={}; g.o.forEach(o=>KEYMAP[o[0]]=[g.n,o[1]]); });
function complete(i){ return !!V[i] && XG.every(g=>!!W[g.n][i]); }
function touched(i){ return !!V[i] || XG.some(g=>!!W[g.n][i]); }
T[0].l.forEach((l,k)=>SLOTK[l.toLowerCase()]=k);
// A store of its own: answers given on the v1 page (no panel) never restore here as if judged with it on offer.
const STORE="abpage-v:"+SESS, VSTORE="abpage-viz:"+SESS, SSTORE="abpage-s:"+SESS, LSTORE="abpage-lay:"+SESS;
const LIVEL=VZC.layers||{}, LIVE={}, SPEC={};
let VIS=false, SH={}, LAY="";
try{ VIS=localStorage.getItem(VSTORE)==="1"; SH=JSON.parse(localStorage.getItem(SSTORE)||"{}")||{};
  LAY=(localStorage.getItem(LSTORE)||"").split("").filter(k=>LIVEL[k]).sort().join(""); }catch(e){}
let tr=0, focus=0, playing=-1, seqT=null;
let cur=0, au=null, cv=null;          // what the panel's shared paintViz() draws from, set per clip
const auds=[], cvs=[];
function save(){
  try{ localStorage.setItem(STORE, JSON.stringify({v:V,w:W__MKSAVE__,t:new Date().toISOString()}));
    document.getElementById('saved').textContent="Saved in this browser at "+
      new Date().toLocaleTimeString()+". Reopening this file restores it; the code is a copy, not the record.";
  }catch(e){ document.getElementById('saved').textContent=
    "COULD NOT SAVE LOCALLY ("+e+"). Copy the code before closing this tab."; }
}
function restore(){
  let d=null; try{ d=JSON.parse(localStorage.getItem(STORE)||"null"); }catch(e){ return; }
  if(!d) return;
  Object.assign(V, d.v||{}); XG.forEach(g=>Object.assign(W[g.n], (d.w||{})[g.n]||{}));__MKREST__
  document.getElementById('saved').textContent="Restored a session saved at "+(d.t||"an earlier time")+".";
}
function emit(){
  let s=""; for(let i=0;i<N;i++) s+=(V[i]||"-");
  const xs=XG.map(g=>{ let x="|"+g.n+"="; for(let i=0;i<N;i++) x+=(W[g.n][i]||"-"); return x; }).join("");
  const shown=Object.keys(SH).map(Number).filter(i=>touched(i)).sort((a,b)=>a-b);
  document.getElementById('code').value=SESS+":"+s+xs+"|S="+
    shown.map(i=>(typeof SH[i]==="string"&&SH[i])?i+":"+SH[i]:String(i)).join(",")__MKEMIT__;
  let n=0; for(let i=0;i<N;i++) n+=complete(i)?1:0;
  document.getElementById('cnt').textContent=n+"/"+N+" answered"__MKCNT__;
  try{ localStorage.setItem(SSTORE, JSON.stringify(SH)); }catch(e){}
  save();
}
function grid(){
  const g=document.getElementById('grid'); g.innerHTML="";
  for(let i=0;i<N;i++){ const s=document.createElement('span');
    if(complete(i)) s.dataset.v=V[i]; else if(touched(i)) s.classList.add('part');
    if(i===tr) s.classList.add('cur');__MKGRID__
    s.title=__UNIT__+" "+(i+1); s.onclick=()=>show(i,true); g.appendChild(s); }
}
function lit(){ document.querySelectorAll('.slot').forEach((el,k)=>el.classList.toggle('playing',k===playing)); }
function stopAll(){ clearTimeout(seqT); seqT=null; auds.forEach(a=>{ a.onended=null; a.pause(); }); playing=-1; lit(); }
function playSlot(k, then){
  if(k<0||k>=auds.length) return;
  stopAll(); const a=auds[k]; focus=k;
  const go=()=>{ try{ a.currentTime=S[T[tr].c[k]]; }catch(e){} const p=a.play(); if(p) p.catch(()=>{}); };
  a.onended=()=>{ a.onended=null; if(then) seqT=setTimeout(then,GAP_MS); };
  if(a.readyState>=1) go(); else a.addEventListener('loadedmetadata',go,{once:true});
}
function playFrom(k){ if(k<auds.length) playSlot(k,()=>playFrom(k+1)); }
function show(t, autoplay){
  stopAll(); tr=t; focus=0;
  document.getElementById('n').textContent=(t+1)+" of "+N;
  const box=document.getElementById('slots'); box.innerHTML=""; auds.length=0; cvs.length=0;
  T[t].c.forEach((ci,k)=>{
    const row=document.createElement('div'); row.className='slot';
    const b=document.createElement('button'); b.className='lab'; b.textContent=T[t].l[k];
    b.title="Play "+T[t].l[k]+" (key "+T[t].l[k].toLowerCase()+")"; b.onclick=()=>playSlot(k);
    const a=document.createElement('audio'); a.controls=true; a.preload='auto'; a.src=A[ci];
    a.addEventListener('play',()=>{ auds.forEach((o,j)=>{ if(j!==k) o.pause(); }); playing=k; focus=k; lit(); });
    a.addEventListener('pause',()=>{ if(playing===k){ playing=-1; lit(); } });
    row.append(b,a); box.appendChild(row); auds.push(a);__MKROW__
    if(MULTI){ const c=document.createElement('canvas'); c.className='vz'; c.hidden=!VIS; box.appendChild(c); cvs.push(c); }
  });
  if(!MULTI){ const c=document.createElement('canvas'); c.className='vz'; c.hidden=!VIS; box.appendChild(c); cvs.push(c); }
  if(TX){ document.getElementById('say').textContent=TX[t]||""; }
  lightButtons();
  document.getElementById('done').textContent="";
  grid();
  if(autoplay) playFrom(0);
}
function lightButtons(){
  document.querySelectorAll('button.v[data-v]').forEach(b=>b.classList.toggle('on',
    (b.dataset.g?W[b.dataset.g][tr]:V[tr])===b.dataset.v));
}
function nextUnjudged(i){
  for(let k=1;k<=N;k++){ const j=(i+k)%N; if(!complete(j)) return j; }
  return -1;
}
function judge(g, v){
  // As on a pool page: the view at the verdict is recorded with it, and a verdict given with the panel hidden clears it.
  // With a second answer group the trial moves on when every group has an answer, whichever comes last; an answer
  // that leaves the trial incomplete changes nothing else, so the clip keeps playing.
  if(VIS) SH[tr]=LAY; else delete SH[tr];
  if(g) W[g][tr]=v; else V[tr]=v;
  emit();
  if(!complete(tr)){ lightButtons(); grid(); return; }
  const j=nextUnjudged(tr);
  if(j<0){ show(tr,false); document.getElementById('done').textContent=
    "Every "+__UNIT__.toLowerCase()+" has an answer. Copy the code below and send it."; return; }
  show(j,true);
}
__COPY__
__DRAW__
const vb=document.getElementById('vzb');
function paintAll(){
  if(!VIS) return;
  const ks=MULTI?T[tr].c.map((_,k)=>k):[focus];
  ks.forEach(k=>{ const c=MULTI?cvs[k]:cvs[0]; if(!c||!auds[k]) return;
    cur=T[tr].c[k]; au=auds[k]; cv=c; VZC.lo=CAL[cur][0]; VZC.hi=CAL[cur][1]; paintViz(); });
}
function loop(){ paintAll(); requestAnimationFrame(loop); }
function setViz(v){
  VIS=v; cvs.forEach(c=>c.hidden=!v); vb.classList.toggle('on',v);
  vb.textContent=v?"Hide pitch & energy":"Show pitch & energy";
  try{ localStorage.setItem(VSTORE, v?"1":"0"); }catch(e){}
}
function toggleViz(){ setViz(!VIS); }
function helpViz(){ document.getElementById('vzhelp').showModal(); }
document.addEventListener('keydown', e=>{
  if(e.target && e.target.tagName==='TEXTAREA') return;
  if(document.getElementById('vzhelp').open){ if(e.key!=="Escape") e.stopImmediatePropagation(); return; }
  if(e.ctrlKey||e.metaKey||e.altKey) return;
  if(e.key==="v"||e.key==="V"){ e.preventDefault(); toggleViz(); }
  else if(e.key==="?"){ e.preventDefault(); helpViz(); }
}, true);
document.addEventListener('keydown', e=>{
  if(e.target && e.target.tagName==='TEXTAREA') return;
  if(e.ctrlKey||e.metaKey||e.altKey) return;
  const k=e.key;
  if(KEYMAP[k]!==undefined){ e.preventDefault(); judge(...KEYMAP[k]); }
  else if(k===" "||e.code==="Space"){ e.preventDefault(); playFrom(0); }
  else if(k.length===1 && SLOTK[k.toLowerCase()]!==undefined){ e.preventDefault(); playSlot(SLOTK[k.toLowerCase()]); }
  else if(k==="ArrowDown"){ e.preventDefault(); playSlot(Math.min(auds.length-1,(playing<0?focus:playing)+1)); }
  else if(k==="ArrowUp"){ e.preventDefault(); playSlot(Math.max(0,(playing<0?focus:playing)-1)); }
  else if(k==="ArrowRight"){ e.preventDefault(); show(Math.min(N-1,tr+1),true); }
  else if(k==="ArrowLeft"){ e.preventDefault(); show(Math.max(0,tr-1),true); }
});
document.querySelectorAll('button.v[data-v]').forEach(b=>b.onclick=()=>judge(b.dataset.g||"",b.dataset.v));
restore(); emit();
show(Math.max(0,nextUnjudged(N-1)),false);
setViz(VIS); document.getElementById('vzl').classList.toggle('on',!!LAY); loop();
"""


MARK_HELP = ("<h2>Marking characters</h2><p>Under A and under B the sentence is shown again, one chip per character. "
             "Click a character you heard <b>mispronounced in that version</b>; it turns red. Click again to clear it. "
             "Punctuation is shown and cannot be marked. Marking is optional and never needed to move on, and a marked "
             "character means only that you heard it wrong there: an unmarked one is not a claim that it was right.</p>")


def help_html(scope: str, multi: bool, marks: bool = False) -> str:
    if marks:
        h = help_html(scope, multi)
        return h.replace("<h2>Keys</h2>", MARK_HELP + "<h2>Keys</h2>", 1)
    where = "each clip of the trial, A above B" if multi else "the clip last played"
    return ("<dialog id=vzhelp><button onclick='this.closest(\"dialog\").close()'>Close</button>"
            "<h2>What the panel shows</h2>"
            f"<p>One panel for {where}. The same drawing as the particle pages.</p>"
            "<p><span class=sw style='background:#c98a4b'></span><b>Top strip, spectrogram:</b> energy by frequency, low at "
            "the bottom. A sound after the words shows here as a smudge after a dark gap.</p>"
            "<p><span class=sw style='background:#7fb3d5'></span><b>Blue line, pitch,</b> on a 1-5 scale, where 1 is the low "
            f"and 5 the high of {html.escape(scope)}. Gaps are where no pitch was found.</p>"
            "<p><span class=sw style='background:rgba(201,138,75,.45)'></span><b>Shaded area, energy,</b> loudness over time.</p>"
            "<p><span class=sw style='background:#efe9e1'></span><b>White line:</b> where playback is.</p>"
            + bp.VIZ_HELP_LIVE +
            "<h2>Reading it with care</h2><p>The panel was measured when the page was built; it is there to look at. "
            "Your ear decides, and whether the panel was showing is saved with each answer.</p>"
            "<h2>Keys</h2><p><b>V</b> shows or hides the panel (hidden is the default). <b>?</b> opens this. "
            "<b>Esc</b> closes it.</p></dialog>")


def check_keys(options: list[tuple[str, str]], slot_labels: list[str], extra: list[dict] = ()) -> list[str]:
    """Why these answer keys cannot work on the page: a key used twice, or one the page already uses. `extra` are
    further answer groups, {"name": one capital letter, "options": [(key, char, label)]}."""
    keys = [str(i) for i in range(1, len(options) + 1)] + [o[0] for g in extra for o in g["options"]]
    out = []
    for name, opts in [("main", options)] + [(g["name"], [o[1:] for o in g["options"]]) for g in extra]:
        if len({o[0] for o in opts}) != len(opts):
            out.append(f"two {name} options share an answer character: {[o[0] for o in opts]}")
        if any(len(o[0]) != 1 or o[0] == "-" or o[0] == "|" for o in opts):
            out.append(f"a {name} answer character is not one character, or is '-' (unanswered) or '|'")
    if len(set(keys)) != len(keys):
        out.append(f"a key answers two things: {keys}")
    if any(not re.fullmatch(r"[A-LN-RT-Z]", g["name"]) for g in extra):
        out.append("a second group's name must be one capital letter other than S (the view) or M (the marks)")
    clash = set(keys) & ({l.lower() for l in slot_labels} | RESERVED_KEYS)
    if clash:
        out.append(f"answer keys {sorted(clash)} are also a play or panel key")
    if {l.lower() for l in slot_labels} & RESERVED_KEYS:
        out.append("a clip label is also a panel key")
    return out


def build_html(session: str, title: str, h1: str, intro: str, question: str, options: list[tuple[str, str]],
               trials: list[dict], audio: list[str], starts: list[float], viz: list[dict], cal: list[list[float]],
               multi: bool, unit: str, scope: str, extra: list[dict] = (), texts: list[str] | None = None,
               marks: bool = False, about: list[dict] | None = None, show_text: bool = True,
               preset: dict | None = None, notes: bool = False, ratings: bool = False) -> str:
    """trials[t] = {"c": [clip indices], "l": [slot labels]}; one main answer per trial, plus one per `extra` group
    ({"name", "question", "options": [(key, char, label)]}), each its own |<name>= field in the code; `texts`, one line
    shown per trial. The panel is always offered. `marks` (default off, and off is byte-identical to the page before
    marks existed) shows each trial's sentence as clickable chips under A and under B, and adds |M= (MARK_GRAMMAR).
    `about` (default off, and off is byte-identical): one {"title", "sections": [(heading, text)]} per trial, shown in
    a dialog from an "About this text" button on the card (key I): who wrote the material, why, and how it was judged.
    Text is escaped; paragraphs split on blank lines. It never changes the code or the answers.
    `show_text=False` hides the sentence line above the players (a long text with marks is already shown under each
    player, owner 2026-10-18: "no need to repeat the text 4 times"); default True is byte-identical.
    `preset` (default off, byte-identical): {"v": {trial: answer}, "mk": {trial: {side: [i]}}, "t": label}, shown
    when this browser has nothing saved, so a published demo opens with a listener's answers and marks.
    `notes` (default off, byte-identical): a free-form box per trial and one for the whole page (owner 2026-10-19: "We
    should give a free form feedback provider option"), saved with the answers and carried as |N= (NOTES_GRAMMAR).
    `ratings` (default off, byte-identical): every voice is rated 1-5 ("sounds natural"), keys 1-5 rating the voice
    playing or last played and starting the next; the trial's answer is DERIVED from the ratings (the top voice, or
    Same on a tie) and the buttons still override it. Carried as |R= (RATINGS_GRAMMAR). Owner 2026-10-19: marks are
    "boolean and not a spectrum", "we need to be more granular or a total tally, instead of pure preference", and
    then "refocus/tune for smooth and quick judgement while retaining the non boolean/tally nature"."""
    why = check_keys(options, trials[0]["l"], list(extra))
    if about is not None and len(about) != len(trials):
        raise SystemExit(f"{session}: {len(about)} about entries for {len(trials)} trials")
    if why:
        raise SystemExit(f"{session}: {why}")
    if not (len(audio) == len(starts) == len(viz) == len(cal)):
        raise SystemExit(f"{session}: {len(audio)} clips, {len(starts)} starts, {len(viz)} contours, {len(cal)} scales")
    if texts is not None and len(texts) != len(trials):
        raise SystemExit(f"{session}: {len(texts)} texts for {len(trials)} trials")
    if marks:
        if texts is None:
            raise SystemExit(f"{session}: marks need the sentence of every trial (texts=)")
        if any(t["l"] not in (["A", "B"], ["A", "B", "C"]) for t in trials):
            raise SystemExit(f"{session}: marks are per side of an A/B (or A/B/C) trial; a trial here is neither")
        if any(not cjk_chars(x) for x in texts):
            raise SystemExit(f"{session}: a trial's sentence has no CJK character to mark")
    opt = [[str(i), o[0], o[1]] for i, o in enumerate(options, 1)]
    buttons = "".join(f"<button class=v data-v=\"{html.escape(c)}\"><kbd>{k}</kbd>{html.escape(lab)}</button>"
                      for k, c, lab in opt)
    for g in extra:
        buttons += (f"</div><p class=qn2>{html.escape(g['question'])}</p><div class=btns>"
                    + "".join(f"<button class=v data-g=\"{g['name']}\" data-v=\"{html.escape(c)}\"><kbd>{html.escape(k)}"
                              f"</kbd>{html.escape(lab)}</button>" for k, c, lab in g["options"]))
    slots = trials[0]["l"]
    legend = ("<div class=legend><span>Keys:</span>"
              + "".join(f"<span><kbd>{k}</kbd> {html.escape(lab)}</span>" for k, _, lab in opt)
              + "".join(f"<span><kbd>{html.escape(k)}</kbd> {html.escape(g.get('short', g['name']))}: "
                        f"{html.escape(lab)}</span>" for g in extra for k, _, lab in g["options"])
              + f"<span><kbd>Space</kbd> play {' then '.join(slots) if len(slots) <= 2 else 'all in order'}</span>"
              + "<span>" + " ".join(f"<kbd>{l.lower()}</kbd>" for l in slots) + " play one</span>"
              + ("<span><kbd>&uarr;</kbd><kbd>&darr;</kbd> previous / next clip</span>" if len(slots) > 2 else "")
              + f"<span><kbd>&larr;</kbd><kbd>&rarr;</kbd> previous / next {unit.lower()}</span>"
              + "<span><kbd>V</kbd> panel</span><span><kbd>?</kbd> help</span>"
              + ("<span>click a character under A or B: mispronounced there</span>" if marks else "") + "</div>")
    vcal = {"lo": -12.0, "hi": 12.0, "bins": bp.SPEC_BINS, "targets": bp.TONE_TARGETS, "names": bp.TONE_NAMES,
            "version": bp.VIZ_VERSION, "layers": bp.LIVE_LAYERS}
    js = (AB_JS.replace("__COPY__", bp.COPY_JS).replace("__DRAW__", bp.VIZ_DRAW_JS)
          .replace("__SESS__", json.dumps(session)).replace("__N__", str(len(trials)))
          .replace("__A__", json.dumps(audio)).replace("__S__", json.dumps([round(s, 3) for s in starts]))
          .replace("__T__", json.dumps(trials, ensure_ascii=False)).replace("__OPT__", json.dumps(
              [[k, c] for k, c, _ in opt], ensure_ascii=False))
          .replace("__MULTI__", "true" if multi else "false").replace("__GAP__", str(GAP_MS))
          .replace("__UNIT__", json.dumps(unit))
          .replace("__XG__", json.dumps([{"n": g["name"], "o": [[k, c] for k, c, _ in g["options"]]} for g in extra],
                                        ensure_ascii=False))
          .replace("__TX__", json.dumps(texts, ensure_ascii=False) if texts is not None else "null")
          .replace("__VZ__", json.dumps(viz, separators=(",", ":")))
          .replace("__VZC__", json.dumps(vcal, ensure_ascii=False, separators=(",", ":")))
          .replace("__CAL__", json.dumps(cal, separators=(",", ":"))))
    for marker, code in AB_MARK_SPLICE.items():
        if js.count(marker) != 1:
            raise SystemExit(f"the A/B script no longer has {marker} exactly once")
        js = js.replace(marker, code.replace("__CJKR__", json.dumps(CJK_RANGES)) if marks else "")
    if marks and any(t["l"] == ["A", "B", "C"] for t in trials):    # three players: C marks too (A/B pages unchanged)
        if js.count('["A","B"]') != 2:
            raise SystemExit('the marks script no longer names its sides ["A","B"] exactly twice')
        js = js.replace('["A","B"]', '["A","B","C"]')
    if preset is not None:
        old = '  if(!d) return;\n  Object.assign(V, d.v||{});'
        if js.count(old) != 1:
            raise SystemExit("the A/B script's restore() no longer has its empty-store return exactly once")
        js = js.replace(old, '  if(!d) d=PRESET;\n  if(!d) return;\n  Object.assign(V, d.v||{});')
        js = "const PRESET=" + json.dumps(preset, ensure_ascii=False) + ";\n" + js
    if marks:
        for old, new in AB_MARK_STORES.items():
            if js.count(old) != 1:
                raise SystemExit(f"the A/B script no longer has {old!r} exactly once")
            js = js.replace(old, new)
    if notes:
        for old, new in NOTES_SPLICE.items():
            if js.count(old) != 1:
                raise SystemExit(f"the A/B script no longer has {old!r} exactly once")
            js = js.replace(old, new)
        js = NOTES_JS + js + NOTES_WIRE
    if ratings:
        if any(o[0] not in RATING_DERIVED for o in options):
            raise SystemExit(f"{session}: a ratings page derives its answer, so its options must be among {RATING_DERIVED}")
        for old, new in RATINGS_SPLICE.items():
            if js.count(old) != 1:
                raise SystemExit(f"the A/B script no longer has {old!r} exactly once")
            js = js.replace(old, new)
        js = RATINGS_JS + js + RATINGS_WIRE
    bar = ("<div class=vizbar><button id=vzb onclick='toggleViz()' title='V'>Show pitch &amp; energy</button>"
           "<button id=vzl onclick='layersViz()' title='Live layers'>Layers</button>"
           "<button class=q onclick='helpViz()' title='What am I looking at? (?)'>?</button></div>"
           "<div id=vzlay hidden></div>")
    parts = [
        "<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>",
        f"<title>{html.escape(title)}</title>",
        f"<style>{bp.CSS}</style>", f"<style>{bp.VIZ_CSS}</style>", f"<style>{AB_CSS}</style>"
        + (f"<style>{AB_MARK_CSS}</style>" if marks else "") + (f"<style>{NOTES_CSS}</style>" if notes else "")
        + (f"<style>{RATINGS_CSS}</style>" if ratings else ""),
        "<div class=w>",
        f"<h1>{html.escape(h1)}</h1>",
        f"<p class=sub>{intro}</p>",
        legend,
        f"<div class=card><div class=pos>{html.escape(unit)} <span id=n></span></div>",
        f"<p class=qn>{html.escape(question)}</p>",
        ("<p><button class=q onclick='aboutOpen()' title='I'>About this text</button></p>" if about is not None else ""),
        ("<p class=say id=say></p>" if show_text else "<p class=say id=say hidden></p>") if texts is not None else "",
        bar,
        "<div id=slots></div>",
        f"<div class=btns>{buttons}</div>",
        (f"<p class=qn2>Notes on this {html.escape(unit.lower())} (optional, free form): anything the buttons and "
         "marks cannot say</p><textarea id=note class=note rows=3></textarea>" if notes else ""),
        "<p class=sub id=done></p></div>",
        "<div class=grid id=grid></div><p class=sub id=saved></p>"
        + ("<p class=qn2>Notes on the whole page (optional, free form)</p>"
           "<textarea id=noteall class=note rows=3></textarea>" if notes else "") + "</div>",
        help_html(scope, multi, marks),
        "<div class=dock><div class=dockin><span class=count id=cnt></span>"
        "<textarea id=code readonly></textarea>"
        "<button class=big onclick='copyCode(this)'>Copy code</button></div></div>",
        f"<script>{bp.vendored_bundle()}</script>",
        f"<script>{js}</script>",
    ]
    if about is not None:
        ab = about_html(about)
        if notes:       # typing an "i" in a notes box must not open the dialog
            old = "  if(d.open){"
            if ab.count(old) != 1:
                raise SystemExit("the About script no longer has its open-dialog test exactly once")
            ab = ab.replace(old, "  if(e.target && e.target.tagName==='TEXTAREA') return;\n" + old)
        parts.append(ab)
    return "\n".join(parts)


# Free-form notes: NT[trial] and NT.all, saved with the answers, emitted as |N=<encodeURIComponent(JSON)> (empty when
# none). encodeURIComponent escapes "|" and ",", so the field cannot break the code's other fields.
NOTES_GRAMMAR = ("|N=<encodeURIComponent of a JSON object {\"<trial position>\": text, \"all\": text}>, empty when no "
                 "note was written; always present on a page that offers notes")
NOTES_CSS = "textarea.note{width:100%;box-sizing:border-box;font:inherit;font-size:15px;line-height:1.4}"
NOTES_JS = """const NT={};
function noteSet(k,v){ if(v.trim()) NT[k]=v; else delete NT[k]; emit(); }
function noteShow(){ const e=document.getElementById('note'); if(e) e.value=NT[tr]||""; }
"""
NOTES_SPLICE = {"JSON.stringify({v:V,w:W": "JSON.stringify({v:V,nt:NT,w:W",
                "  Object.assign(V, d.v||{});": "  Object.assign(V, d.v||{}); Object.assign(NT, d.nt||{});",
                "  let n=0; for(let i=0;i<N;i++) n+=complete(i)?1:0;":
                    "  document.getElementById('code').value+=\"|N=\"+(Object.keys(NT).length?"
                    "encodeURIComponent(JSON.stringify(NT)):\"\");\n  let n=0; for(let i=0;i<N;i++) n+=complete(i)?1:0;",
                "  document.getElementById('done').textContent=\"\";":
                    "  document.getElementById('done').textContent=\"\"; noteShow();"}
NOTES_WIRE = """
document.getElementById('note').addEventListener('input',e=>noteSet(String(tr),e.target.value));
document.getElementById('noteall').addEventListener('input',e=>noteSet("all",e.target.value));
document.getElementById('noteall').value=NT.all||""; noteShow();
"""


# Ratings: RT[trial][side] = 1..5. Keys 1-5 rate the voice playing (or last played) and start the next unrated one;
# when every voice of the trial is rated, the answer is derived -- the top voice, or Same on a tie -- through judge(),
# so the page moves on exactly as if that answer had been pressed. A later change re-derives without moving on.
RATINGS_GRAMMAR = ("|R=<token>,... one per trial: '-' or side letters each followed by a rating 1-5 "
                   "(1 not Cantonese-like, 3 acceptable, 5 native), e.g. A4B3C5; the main answer is derived from it")
RATING_DERIVED = ("a", "b", "c", "s", "?")
RATINGS_CSS = (".rate{margin:2px 0 10px;font-size:14px;color:var(--dim)}"
               ".rate button.rv{font:600 14px ui-monospace,Menlo,monospace;width:30px;height:28px;margin:0 2px;"
               "border-radius:6px;border:1px solid #322a22;background:#221d18;color:var(--fg);cursor:pointer}"
               ".rate button.rv.on{background:var(--acc);border-color:var(--acc);color:#000}")
RATINGS_JS = """const RT={}, RTB={};
function rated(t){ return T[t].l.every(sd=>(RT[t]||{})[sd]); }
function derived(t){ const o=RT[t]||{}, top=Math.max(...T[t].l.map(sd=>o[sd]||0));
  const w=T[t].l.filter(sd=>o[sd]===top); return w.length===1 ? w[0].toLowerCase() : "s"; }
function paintRate(t){ T[t].l.forEach(sd=>(RTB[sd]||[]).forEach((b,j)=>b.classList.toggle('on',(RT[t]||{})[sd]===j+1))); }
function setRate(t,side,v,auto){
  const o=RT[t]=RT[t]||{}; if(o[side]===v && !auto) delete o[side]; else o[side]=v;
  if(!Object.keys(o).length) delete RT[t];
  paintRate(t);
  if(!rated(t)){ emit(); grid();
    if(auto){ const k=T[t].l.findIndex(sd=>!(RT[t]||{})[sd]); if(k>=0) playSlot(k); } return; }
  if(V[t]){ V[t]=derived(t); emit(); lightButtons(); grid(); return; }
  judge("", derived(t));
}
function rCode(){ const a=[]; for(let i=0;i<N;i++){ const o=RT[i]||{};
  const s=T[i].l.filter(sd=>o[sd]).map(sd=>sd+o[sd]).join(""); a.push(s||"-"); } return a.join(","); }
function rateRow(t,k){
  const side=T[t].l[k], d=document.createElement('div'); d.className='rate';
  const lab=document.createElement('span'); lab.textContent=side+" sounds natural: "; d.appendChild(lab);
  RTB[side]=[];
  for(let v=1;v<=5;v++){ const b=document.createElement('button'); b.className='rv'; b.textContent=String(v);
    if((RT[t]||{})[side]===v) b.classList.add('on');
    b.onclick=()=>setRate(tr,side,v,false); RTB[side].push(b); d.appendChild(b); }
  const h=document.createElement('span'); h.textContent="  1 not Cantonese-like, 3 acceptable, 5 native"; d.appendChild(h);
  return d; }
"""
RATINGS_SPLICE = {"JSON.stringify({v:V,": "JSON.stringify({v:V,rt:RT,",
                  "  Object.assign(V, d.v||{});": "  Object.assign(V, d.v||{}); Object.assign(RT, d.rt||{});",
                  "  let n=0; for(let i=0;i<N;i++) n+=complete(i)?1:0;":
                      "  document.getElementById('code').value+=\"|R=\"+rCode();\n"
                      "  let n=0; for(let i=0;i<N;i++) n+=complete(i)?1:0;",
                  "box.appendChild(row); auds.push(a);": "box.appendChild(row); auds.push(a); box.appendChild(rateRow(t,k));"}
RATINGS_WIRE = """
document.addEventListener('keydown', e=>{
  if(e.target && e.target.tagName==='TEXTAREA') return;
  if(e.ctrlKey||e.metaKey||e.altKey) return;
  if(!/^[1-5]$/.test(e.key)) return;
  e.preventDefault(); e.stopImmediatePropagation();
  const k=playing>=0?playing:focus; setRate(tr, T[tr].l[k], Number(e.key), true);
}, true);
"""


def parse_ratings(field: str, trials: list) -> list[dict]:
    """|R= back to one {side: rating} per trial; refuses anything the page could not have written."""
    toks = field.split(",")
    if len(toks) != len(trials):
        raise SystemExit(f"the code's R has {len(toks)} tokens; the key has {len(trials)} trials")
    out = []
    for i, (tok, t) in enumerate(zip(toks, trials)):
        if tok == "-":
            out.append({})
            continue
        m = re.fullmatch(r"(?:A(?P<A>[1-5]))?(?:B(?P<B>[1-5]))?(?:C(?P<C>[1-5]))?", tok)
        if not m:
            raise SystemExit(f"R token {i} {tok!r} is not this page's rating grammar")
        d = {k: int(v) for k, v in m.groupdict().items() if v}
        if set(d) - {k for k in "ABC" if k in t}:
            raise SystemExit(f"R token {i} rates a side the trial does not have")
        out.append(d)
    return out


def parse_notes(field: str, n: int) -> dict:
    """|N= back to {position or "all": text}; refuses a key the page could not have written."""
    import urllib.parse
    if not field:
        return {}
    try:
        d = json.loads(urllib.parse.unquote(field))
    except ValueError as e:
        raise SystemExit(f"the code's N is not the page's encoding: {e}")
    if not isinstance(d, dict) or any(not isinstance(v, str) for v in d.values()):
        raise SystemExit("the code's N is not an object of texts")
    bad = [k for k in d if k != "all" and not (k.isdigit() and int(k) < n)]
    if bad:
        raise SystemExit(f"the code's N names {bad}, which are not this page's trials")
    return d


ABOUT_JS = """const ABOUT=__ABOUT__;
function aboutOpen(){ const d=document.getElementById('about'); document.getElementById('aboutbody').innerHTML=ABOUT[tr]; d.showModal(); }
document.addEventListener('keydown',e=>{
  const d=document.getElementById('about');
  if(d.open){ if(e.key!=="Escape") e.stopImmediatePropagation(); return; }
  if((e.key==='i'||e.key==='I') && !e.ctrlKey && !e.metaKey && !e.altKey){ e.stopImmediatePropagation(); aboutOpen(); }
}, true);"""


def about_html(about: list[dict]) -> str:
    """The "About this text" dialog: per trial, a title and (heading, text) sections, escaped; blank lines split
    paragraphs."""
    def one(a: dict) -> str:
        body = f"<h2>{html.escape(a['title'])}</h2>"
        for head, text in a.get("sections", []):
            paras = "".join(f"<p>{html.escape(x.strip())}</p>" for x in str(text).split("\n\n") if x.strip())
            body += f"<h3>{html.escape(head)}</h3>{paras}"
        return body
    return ("<dialog id=about style='max-width:46em'><button onclick='this.closest(\"dialog\").close()'>Close</button>"
            "<div id=aboutbody></div></dialog>"
            f"<script>{ABOUT_JS.replace('__ABOUT__', json.dumps([one(a) for a in about], ensure_ascii=False))}</script>")


def contour_of(b: bytes) -> dict:
    return bp.contour(io.BytesIO(b))


def scale(contours: list[dict]) -> list[float]:
    if not any(x is not None for c in contours for x in c["f"]):
        return [-12.0, 12.0]
    c = bp.calibration(contours)
    return [c["lo"], c["hi"]] if c["hi"] > c["lo"] else [c["lo"] - 1, c["lo"] + 1]


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def uri(b: bytes) -> str:
    return "data:audio/flac;base64," + base64.b64encode(b).decode("ascii")


SHIP_MARK_INTRO = (" <b>Marks:</b> under A and under B the sentence is shown again; click any character you hear "
                   "mispronounced in that version, and click again to clear it. Marking is optional and never needed to "
                   "move on; the keys work as before.")

# ----------------------------------------------------------------- harvest
CODE = re.compile(r"^(?P<id>[A-Za-z0-9_-]+):(?P<v>[^|]*)(?P<rest>(?:\|[A-Z]=[^|]*)*)$")


def groups_of(key: dict) -> list[dict]:
    """The page's answer groups, main first: {"name" ("" for the main), "answers", "sides" (answer -> the sides it
    names, None for no claim; absent on a rating), "control_hit" ("planted": naming the planted side detects the
    control; "unplanted": preferring the untouched side does)}. A key written before groups existed is read through
    its options, or through V1 for a v1 key."""
    if key.get("groups"):
        return key["groups"]
    if "options" in key:
        sides = {o["answer"]: o["sides"] for o in key["options"] if "sides" in o}
        return [{"name": "", "answers": [o["answer"] for o in key["options"]], "sides": sides or None,
                 "control_hit": "planted"}]
    spec = V1[key["bm"]]
    return [{"name": "", "answers": [c for c, _ in spec["options"]], "sides": spec["sides"], "control_hit": "planted"}]


MARK_TOKEN = re.compile(r"(?:A(?P<A>[0-9]+(?:\.[0-9]+)*))?(?:B(?P<B>[0-9]+(?:\.[0-9]+)*))?(?:C(?P<C>[0-9]+(?:\.[0-9]+)*))?")


def trial_cjk_len(t: dict, pos: int) -> int:
    """How many markable characters trial `pos` shows: from its text when the key stores one, else its cjk_len."""
    n = t.get("cjk_len")
    if "text" in t:
        m = len(cjk_chars(t["text"]))
        if n is not None and n != m:
            raise SystemExit(f"trial {pos}: the key's cjk_len {n} is not its text's {m}")
        return m
    if not isinstance(n, int) or isinstance(n, bool) or n < 1:
        raise SystemExit(f"trial {pos}: a marking key must give every trial its cjk_len (got {n!r})")
    return n


def parse_mark_token(tok: str, n: int, pos: int, sides: str = "AB") -> dict:
    """One trial's |M= token -> {"A": [...], "B": [...]}, refusing anything the page could not have emitted."""
    if tok == "-":
        return {"A": [], "B": []}
    other = sorted({c for c in tok if c.isalpha() and c not in sides})
    if other:
        raise SystemExit(f"M token {tok!r} at trial {pos}: side {other} is not one of this trial's {list(sides)}")
    m = MARK_TOKEN.fullmatch(tok)
    if not tok or not m:
        raise SystemExit(f"M token {tok!r} at trial {pos} is malformed (MARK_GRAMMAR)")
    out = {}
    for side in ("A", "B") + (("C",) if m["C"] is not None else ()):    # C only on three-player pages
        parts = m[side].split(".") if m[side] is not None else []
        if any(len(p) > 1 and p[0] == "0" for p in parts):
            raise SystemExit(f"M token {tok!r} at trial {pos}: an index has a leading zero")
        idx = [int(p) for p in parts]
        if len(set(idx)) != len(idx):
            raise SystemExit(f"M token {tok!r} at trial {pos}: {side} marks an index twice")
        if idx != sorted(idx):
            raise SystemExit(f"M token {tok!r} at trial {pos}: {side}'s indices are not ascending")
        if idx and idx[-1] >= n:
            raise SystemExit(f"M token {tok!r} at trial {pos}: index {idx[-1]} is at or past its {n} CJK characters")
        out[side] = idx
    return out


def parse_code(code: str, key: dict) -> dict:
    """A code from v1 or v2 of a page, joined to its key by position. Refuses anything the page could not have made.

    Grammar: <ID>:<main answers>[|<G>=<answers of group G>]...[|S=<view>][|M=<marks>]. Every group the key names must
    be there, once, with one character per trial; |M= must be there exactly when the key says "marks": true; nothing
    else may be."""
    m = CODE.match(code.strip())
    if not m:
        raise SystemExit(f"not an A/B or audition code: {code[:40]!r}")
    if m["id"] != key["bm"]:
        raise SystemExit(f"the code is for {m['id']!r}, not {key['bm']}")
    trials = key["trials"]
    segs = [s.split("=", 1) for s in m["rest"].split("|")[1:]]
    names = [n for n, _ in segs]
    if len(set(names)) != len(names):
        raise SystemExit(f"a field appears twice: {names}")
    fields = dict(segs)
    marking = key.get("marks") is True
    if marking and "M" not in fields:
        raise SystemExit("this page's key says marks: true, and the code carries no |M=")
    if not marking and "M" in fields:
        raise SystemExit("the code carries |M=, and this page's key does not offer marks")
    noting = key.get("notes") is True
    if noting != ("N" in fields):
        raise SystemExit("this page's key offers notes and the code carries no |N=" if noting
                         else "the code carries |N=, and this page's key does not offer notes")
    notes = parse_notes(fields.get("N", ""), len(trials)) if noting else {}
    rating = key.get("ratings") is True
    if rating != ("R" in fields):
        raise SystemExit("this page's key offers ratings and the code carries no |R=" if rating
                         else "the code carries |R=, and this page's key does not offer ratings")
    rates = parse_ratings(fields["R"], trials) if rating else None
    groups = groups_of(key)
    want = {g["name"] for g in groups[1:]}
    if set(fields) - {"S", "M", "N", "R"} != want:
        raise SystemExit(f"the code carries fields {sorted(set(fields) - {'S', 'M', 'N', 'R'})}; this page's are {sorted(want)}")
    marks = None
    if marking:
        toks = fields["M"].split(",")
        if len(toks) != len(trials):
            raise SystemExit(f"the code's M has {len(toks)} tokens; the key has {len(trials)} trials")
        marks = [parse_mark_token(tok, trial_cjk_len(t, i), i, "ABC" if "C" in t else "AB")
                 for i, (tok, t) in enumerate(zip(toks, trials))]
    ans = {"": m["v"], **{g["name"]: fields[g["name"]] for g in groups[1:]}}
    for g in groups:
        a = ans[g["name"]]
        label = g["name"] or "main"
        if len(a) != len(trials):
            raise SystemExit(f"the code's {label} answers {len(a)} trials; the key has {len(trials)}")
        bad = sorted({c for c in a if c not in set(g["answers"]) | {"-"}})
        if bad:
            raise SystemExit(f"{label} answers {bad} are not this page's ({sorted(g['answers'])})")
    answered_any = [any(ans[g["name"]][i] != "-" for g in groups) for i in range(len(trials))]
    # A trial with marks and no answer is kept: the page emits marks whether or not the trial is answered.
    touched = [answered_any[i] or bool(marks and any(marks[i].values())) or str(i) in notes
               or bool(rates and rates[i]) for i in range(len(trials))]
    shown = {}
    if "S" in fields:
        for item in filter(None, fields["S"].split(",")):
            pos, _, lay = item.partition(":")
            if not pos.isdigit() or int(pos) >= len(trials) or not answered_any[int(pos)]:
                raise SystemExit(f"S names {item!r}, which is not an answered trial")
            if set(lay) - set(bp.LIVE_LAYERS) or len(set(lay)) != len(lay):
                raise SystemExit(f"S names layers {lay!r}; the page has {sorted(bp.LIVE_LAYERS)}")
            shown[int(pos)] = lay
    version = 2 if "S" in fields else 1
    rows = []
    for i, t in enumerate(trials):
        if not touched[i]:
            continue
        row = {"pos": i, "panel_offered": version == 2,
               "viz_shown": (i in shown) if version == 2 else None, "viz_layers": shown.get(i) if version == 2 else None,
               **{k: t[k] for k in t if k not in ("clips",)}}
        for g in groups:
            sfx = f"_{g['name']}" if g["name"] else ""
            c = ans[g["name"]][i]
            row["answer" + sfx] = None if c == "-" else c
            if c == "-" or not g.get("sides"):
                continue
            sides = g["sides"][c]
            row["sides" + sfx] = sides
            if t.get("kind") == "control":
                hit = [t["planted_on"]] if g.get("control_hit", "planted") == "planted" else \
                    [s for s in ("A", "B") if s != t["planted_on"]]
                row["control_detected" + sfx] = sides == hit if sides is not None else None
            elif sides is not None:
                row["arms" + sfx] = sorted(t[s] for s in sides)
        if noting:
            row["note"] = notes.get(str(i))
        if rating:
            row["ratings"] = rates[i]
            row["ratings_by_arm"] = {t[sd]: rates[i].get(sd) for sd in "ABC" if sd in t}
        if marking:
            mk = marks[i]
            row["marks"] = mk
            row["cjk_len"] = trial_cjk_len(t, i)
            if "A" in t and "B" in t:
                by = {}
                for side in ("A", "B") + (("C",) if "C" in t else ()):
                    by.setdefault(t[side], []).extend(mk.get(side, []))
                row["marks_by_arm"] = {a: sorted(v) for a, v in by.items()}
            elif t.get("kind") == "control":
                # one arm on both sides; what separates them is the planted tail
                un = "B" if t["planted_on"] == "A" else "A"
                row["marks_by_role"] = {"planted": mk[t["planted_on"]], "untouched": mk[un]}
            if "text" in t:
                cs = cjk_chars(t["text"])
                row["marked_chars"] = {side: [cs[j] for j in mk.get(side, [])] for side in ("A", "B") + (("C",) if "C" in t else ())}
        rows.append(row)
    complete = sum(all(r.get("answer" + (f"_{g['name']}" if g["name"] else "")) for g in groups) for r in rows)
    return {"bm": key["bm"], "version": version, "answered": complete, "touched": len(rows), "trials": len(trials),
            "groups": [g["name"] for g in groups], "marks_offered": marking, "rows": rows,
            **({"notes_offered": True, "note_all": notes.get("all")} if noting else {})}
