"""대시보드 2단계(R-5777) — 결정 하나 단위로 들여다보는 두 쪽. 표준 라이브러리만.

- `samples.html`(F-001) — 채점 평가 `eval-scored-{저장본}.json` 의 `per_sample`(표본마다 칸별
  정답 `truth` · 예측 `pred`)을 뇌 넷(진짜 · 재배선 셋) 나란히. 거르기(칸 · 진짜만 맞음 ·
  진짜만 틀림 · 에피소드)는 페이지 안 자바스크립트 — 자료는 페이지에 JSON 으로 박는다
  (외부 파일 없음).
- `episodes.html`(F-003) — 최신 성적표 `scorecard-*/<구성>.jsonl`(에피소드 요약)의 씨앗 × 구성 표 ·
  칸을 누르면 `<구성>.decisions.jsonl`(scorecard-v3 부터 · R-5777 3.3 · 쓰는 쪽
  `loop/scorecard.py` `decision_row`)의 결정 표. 없으면
  "결정 기록 전(scorecard-v2)". 교사가 돌라는 쪽(`teacher.turn`)과 Qwen 이 돈 쪽(`action.turn`)이
  다르면 표시 — 냄새 좌우 병목을 눈으로 보는 자리.
  층 구조 기록(scorecard-v4 · R-1027 — `tier` · `trigger`)이 있으면 결정 표에 층 · 계기 칸과
  LLM 호출 수 · 호출률을 더한다(없는 옛 기록은 그대로).

가벼움(R-7347): numpy 를 올리지 않고, 원본이 바뀌지 않았으면(경로 · 크기 · 수정 시각 지문)
다시 쓰지 않는다. 결정 줄은 한 줄씩 읽어 필요한 값만 남긴다(`max_decisions` 에서 멈춤).
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

BRAIN_ORDER = ("real", "rewired-s0", "rewired-s1", "rewired-s2")
TURN_KO = {"left": "왼쪽", "right": "오른쪽", "none": "안 돎"}
TIER_KO = {"llm": "Qwen", "fast": "빠른 정책", "reflex": "반사"}
TRIGGER_KO = {"goal": "목표", "gap": "간격", "conflict": "충돌", "ambiguous": "애매함",
              "surprise": "놀람", "random": "무작위", "periodic": "주기"}
_SIGS: dict[Path, tuple] = {}          # 쪽 → (원본 지문 · 요약) — 지문이 같으면 읽지도 쓰지도 않음


def _sig(paths_: list[Path]) -> tuple:
    out = []
    for p in sorted(paths_):
        try:
            st = p.stat()
            out.append((str(p), st.st_size, st.st_mtime_ns))
        except OSError:
            out.append((str(p), None, None))
    return tuple(out)


def _cached(out: Path, sig: tuple) -> Any:
    """원본 지문이 지난번과 같고 쪽이 있으면 그때의 요약 — 30 초마다 다시 읽지 않으려고."""
    hit = _SIGS.get(out)
    return hit[1] if hit and hit[0] == sig and out.exists() else None


def _write(out: Path, chunks, sig: tuple, summary: Any) -> None:
    """조각을 차례로 파일에 흘려 쓴다(큰 문자열을 한 번에 만들지 않음) · 다 쓴 뒤 바꿔 끼움."""
    tmp = out.with_suffix(out.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(c)
    tmp.replace(out)
    _SIGS[out] = (sig, summary)


def _json_blob(obj: Any) -> str:
    """<script type="application/json"> 안에 안전하게 — `</` 로 스크립트가 닫히지 않게."""
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")


# ---------------------------------------------------------------- F-001 캡션 표본

def read_samples(runs: Path, version: str, checkpoints: list[str],
                 field_order: list[str], max_samples: int) -> dict[str, Any]:
    """뇌 넷 × 저장본의 per_sample 을 한 표로. 표본 순서 · 정답이 뇌마다 같은지 확인한다."""
    files: list[Path] = []
    cps: dict[str, Any] = {}
    for cp in checkpoints:
        brains: dict[str, dict[str, Any]] = {}
        for b in BRAIN_ORDER:
            f = runs / f"projector-{b}-{version}" / f"eval-scored-{cp}.json"
            if f.exists():
                files.append(f)
                try:
                    brains[b] = json.loads(f.read_text(encoding="utf-8"))
                except (ValueError, OSError):
                    continue
        if "real" not in brains:
            continue
        real = brains["real"]
        ps = (real.get("per_sample") or [])[:max_samples]
        fields = [f for f in field_order if any(f in s.get("truth", {}) for s in ps)]
        names = [b for b in BRAIN_ORDER if b in brains]
        preds = {b: {(s.get("i"), s.get("seed")): s.get("pred") or {}
                     for s in (brains[b].get("per_sample") or [])} for b in names}
        rows, d_of, mismatched = [], {}, 0
        for s in ps:
            key = (s.get("i"), s.get("seed"))
            d = d_of.get(s.get("seed"), 0)          # 에피소드 안 결정 번호(파일 순서)
            d_of[s.get("seed")] = d + 1
            truth = s.get("truth") or {}
            row_p = []
            for b in names:
                if key not in preds[b]:
                    mismatched += 1
                row_p.append([preds[b].get(key, {}).get(f) for f in fields])
            rows.append({"i": s.get("i"), "seed": s.get("seed"), "d": d,
                         "t": [truth.get(f) for f in fields], "p": row_p})
        cps[cp] = {"brains": names, "fields": fields, "rows": rows,
                   "summary": summarize(rows, names, fields),
                   "file_accuracy": {b: (brains[b].get("accuracy") or {}) for b in names},
                   "baseline": real.get("majority_baseline") or {},
                   "missing_pairs": mismatched}
    return {"version": version, "checkpoints": cps, "_files": files}


def summarize(rows: list[dict[str, Any]], names: list[str],
              fields: list[str]) -> dict[str, dict[str, Any]]:
    """칸마다 맞음 수 · 진짜만 맞음 · 진짜만 틀림 · 재배선 뇌별 (진짜✓ 그뇌✗) − (진짜✗ 그뇌✓).

    마지막 값은 정의상 (진짜 맞음 수 − 그 뇌 맞음 수)와 같다 — 판정 표의 차이와 맞춰 보는 자리.
    """
    out: dict[str, dict[str, Any]] = {}
    for j, f in enumerate(fields):
        n, correct = 0, [0] * len(names)
        only_right = only_wrong = 0
        pair = {b: [0, 0] for b in names[1:]}        # [진짜만 이 뇌보다 맞음, 이 뇌만 맞음]
        for r in rows:
            t = r["t"][j]
            if t is None:
                continue
            n += 1
            ok = [r["p"][k][j] == t for k in range(len(names))]
            for k, o in enumerate(ok):
                correct[k] += o
            if len(ok) > 1:
                only_right += ok[0] and not any(ok[1:])
                only_wrong += (not ok[0]) and all(ok[1:])
            for k, b in enumerate(names[1:], start=1):
                pair[b][0] += ok[0] and not ok[k]
                pair[b][1] += ok[k] and not ok[0]
        out[f] = {"n": n, "correct": dict(zip(names, correct)),
                  "accuracy": {b: (c / n if n else None) for b, c in zip(names, correct)},
                  "only_real_right": only_right, "only_real_wrong": only_wrong,
                  "pair": {b: {"real_only": a, "other_only": o, "diff": a - o}
                           for b, (a, o) in pair.items()}}
    return out


# ---------------------------------------------------------------- F-003 성적표 에피소드

def _latest_scorecard(runs: Path, pattern: str) -> Path | None:
    cands = [p for p in runs.glob(pattern) if (p / "scorecard.json").exists()]
    return max(cands, key=lambda p: (p / "scorecard.json").stat().st_mtime) if cands else None


def _turn(a: Any) -> str | None:
    return a.get("turn") if isinstance(a, dict) else None


def _act(a: Any) -> str:
    """행동 JSON → 짧은 글(읽을 때 한 번 · 페이지 자료를 작게)."""
    if not isinstance(a, dict):
        return "—"
    ts = a.get("turn_strength")
    walk = a.get("walk")
    return (TURN_KO.get(str(a.get("turn")), str(a.get("turn")))
            + (f" {float(ts):.2f}" if ts else "")
            + (f" · 걷기 {float(walk):.2f}" if walk is not None else " · 걷기 ?")
            + (" · 멈춤" if a.get("stop") else ""))


def read_episodes(runs: Path, pattern: str, max_decisions: int,
                  raw_chars: int) -> dict[str, Any]:
    """씨앗 × 구성 에피소드 요약 + (있으면) 결정 기록. 도착 수는 scorecard.json 과 맞춰 본다."""
    sc = _latest_scorecard(runs, pattern)
    if sc is None:
        return {"path": None, "_files": []}
    files = [sc / "scorecard.json"]
    try:
        card = json.loads((sc / "scorecard.json").read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return {"path": sc.name, "error": "읽기 실패", "_files": files}
    configs = list((card.get("configs") or {}).keys())
    seeds = list(card.get("seeds") or [])
    eps: dict[str, dict[str, Any]] = {}
    check: dict[str, Any] = {}
    decisions: dict[str, dict[str, list]] = {}
    has_dec: dict[str, bool] = {}
    budget = max_decisions
    caps: dict[str, int] = {}          # 같은 캡션 글은 한 번만(번호로 가리킴)
    for c in configs:
        eps[c] = {}
        f = sc / f"{c}.jsonl"
        if f.exists():
            files.append(f)
            for ln in f.read_text(encoding="utf-8").splitlines():
                if ln.strip():
                    try:
                        e = json.loads(ln)
                    except ValueError:
                        continue
                    eps[c][str(e.get("seed"))] = {
                        "reached": bool(e.get("reached")), "reach_ms": e.get("reach_ms"),
                        "final_food_mm": e.get("final_food_mm"), "decisions": e.get("decisions"),
                        "violations": e.get("violations"), "dead": e.get("dead")}
        reached = sum(1 for e in eps[c].values() if e["reached"])
        cfg = card["configs"][c] or {}
        expect = (round(cfg["reach_rate"] * cfg["episodes"])
                  if cfg.get("reach_rate") is not None and cfg.get("episodes") else None)
        check[c] = {"episodes": len(eps[c]), "reached": reached, "scorecard_reached": expect,
                    "match": expect is not None and expect == reached}
        dfile = sc / f"{c}.decisions.jsonl"
        has_dec[c] = dfile.exists()
        if not has_dec[c]:
            continue
        files.append(dfile)
        by_seed: dict[str, list] = {}
        n = mism = tiered = calls = 0
        calls_by_seed: dict[str, list[int]] = {}       # 씨앗 → [LLM 호출, 층 기록 결정]
        with dfile.open(encoding="utf-8") as fh:
            for ln in fh:
                if budget <= 0:
                    break
                try:
                    r = json.loads(ln)
                except ValueError:
                    continue
                t, a = _turn(r.get("teacher")), _turn(r.get("action"))
                diff = t is not None and t != a
                n += 1
                mism += diff
                budget -= 1
                tr = r.get("truth") if isinstance(r.get("truth"), dict) else {}
                loom = (tr.get("looming_azimuth_deg") if tr.get("looming_present") else None)
                cap = r.get("caption")
                cap_i = caps.setdefault(cap, len(caps)) if isinstance(cap, str) else None
                tier = r.get("tier")                     # scorecard-v4 층 구조(없으면 None)
                if tier is not None:
                    tiered += 1
                    calls += tier == "llm"
                    cs = calls_by_seed.setdefault(str(r.get("seed")), [0, 0])
                    cs[0] += tier == "llm"
                    cs[1] += 1
                by_seed.setdefault(str(r.get("seed")), []).append([
                    r.get("d"), r.get("t_ms"), cap_i, loom, _act(r.get("teacher")),
                    _act(r.get("action")), r.get("source"), (r.get("raw") or "")[:raw_chars],
                    r.get("valid"), r.get("why"), r.get("odor_L"), r.get("odor_R"),
                    r.get("food_bearing_deg"), r.get("food_mm"), r.get("drive"),
                    r.get("reflex_bins"), TIER_KO.get(tier, tier),
                    TRIGGER_KO.get(r.get("trigger"), r.get("trigger")), diff])
        decisions[c] = by_seed
        check[c].update({"decision_rows": n, "turn_mismatch": mism})
        if tiered:
            check[c].update({"tier_rows": tiered, "llm_calls": calls,
                             "llm_rate": round(calls / tiered, 3), "calls_by_seed": calls_by_seed})
    return {"path": sc.name, "rule": (card.get("judgement") or {}).get("rule"),
            "configs": configs, "seeds": seeds, "episodes": eps, "check": check,
            "has_decisions": has_dec, "decisions": decisions, "captions": list(caps),
            "truncated": budget <= 0, "_files": files}


# ---------------------------------------------------------------- 그리기

EXTRA_CSS = """
.hit{background:var(--ok-soft);color:var(--ok)}
.miss{background:var(--b-soft);color:var(--b)}
.dim{color:var(--ink-3)}
.ctl{display:flex;flex-wrap:wrap;gap:10px 16px;align-items:center;margin:10px 0}
.ctl label{font-size:.85rem}
select{font:inherit;font-size:.85rem;background:var(--panel);color:var(--ink);
 border:1px solid var(--rule);border-radius:3px;padding:2px 6px}
td.c{text-align:center;cursor:pointer}
td.sel{outline:2px solid var(--a)}
.raw{font-family:var(--f-mono);font-size:.75rem;max-width:36ch;overflow-wrap:anywhere}
"""


def _page(title: str, css: str, body: str, blob, script: str) -> list[str]:
    """조각 목록 — `blob` 은 문자열 하나 또는 조각 목록(큰 자료를 한 덩어리로 만들지 않으려고)."""
    head = f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>{css}{EXTRA_CSS}</style>
</head>
<body>
{body}
<script type="application/json" id="data">"""
    tail = f"""</script>
<script>
const D = JSON.parse(document.getElementById('data').textContent);
{script}
</script>
</body>
</html>
"""
    return [head, *([blob] if isinstance(blob, str) else blob), tail]


SAMPLES_JS = r"""
const P = new URLSearchParams(location.hash.slice(1));
const $ = id => document.getElementById(id);
function el(tag, cls, text){ const e=document.createElement(tag); if(cls) e.className=cls;
  if(text!==undefined && text!==null) e.textContent=String(text); return e; }
function opt(sel, v, t){ const o=el('option',null,t); o.value=v; sel.appendChild(o); }
const cps = Object.keys(D.checkpoints);
cps.forEach(c => opt($('cp'), c, c==='final'?'최종':c==='stage1-caption'?'캡션 단계':c));
if (P.get('cp') && cps.includes(P.get('cp'))) $('cp').value = P.get('cp');
function fill(){
  const C = D.checkpoints[$('cp').value];
  const f = $('field'), keep = P.get('f') || f.value;
  f.innerHTML=''; opt(f,'','전부');
  C.fields.forEach(k => opt(f, k, D.names[k]||k));
  if (C.fields.includes(keep)) f.value = keep;
  const ep = $('ep'); const ek = ep.value; ep.innerHTML=''; opt(ep,'','전부');
  [...new Set(C.rows.map(r=>r.seed))].forEach(s => opt(ep, s, s));
  ep.value = ek; if (P.get('m')) $('mode').value = P.get('m');
  P.delete('f'); P.delete('m');
  draw();
}
function draw(){
  const C = D.checkpoints[$('cp').value], nb = C.brains.length;
  const fsel = $('field').value, mode = $('mode').value, ep = $('ep').value;
  const idx = fsel ? [C.fields.indexOf(fsel)] : C.fields.map((_,j)=>j);
  const head = $('head'); head.innerHTML='';
  const tr = el('tr'); ['에피소드(씨앗)','결정'].forEach(h=>tr.appendChild(el('th',null,h)));
  idx.forEach(j => { const th=el('th',null,(D.names[C.fields[j]]||C.fields[j])+' — 정답 · '
    + C.brains.join(' · ')); th.colSpan = 1 + nb; tr.appendChild(th); });
  head.appendChild(tr);
  const body = $('body'); body.innerHTML=''; let shown=0;
  for (const r of C.rows){
    if (ep && String(r.seed)!==ep) continue;
    let keep = false, any = false;
    for (const j of idx){
      const t = r.t[j]; if (t===null || t===undefined) continue; any = true;
      const ok = r.p.map(p=>p[j]===t);
      if (mode==='all') keep = true;
      else if (mode==='only_right' && ok[0] && !ok.slice(1).some(x=>x)) keep = true;
      else if (mode==='only_wrong' && !ok[0] && ok.slice(1).every(x=>x)) keep = true;
      else if (mode==='differ' && ok.some(x=>x!==ok[0])) keep = true;
    }
    if (!any || !keep) continue;
    shown++;
    const row = el('tr'); row.appendChild(el('td',null,r.seed)); row.appendChild(el('td',null,r.d));
    for (const j of idx){
      const t = r.t[j];
      row.appendChild(el('td', t===null?'dim':'', t===null?'—':t));
      for (let k=0;k<nb;k++){
        const p = r.p[k][j];
        row.appendChild(el('td', t===null?'dim':(p===t?'hit':'miss'), p===null?'—':p));
      }
    }
    body.appendChild(row);
  }
  $('count').textContent = shown + ' / ' + C.rows.length + ' 표본';
}
['cp'].forEach(id => $(id).addEventListener('change', fill));
['field','mode','ep'].forEach(id => $(id).addEventListener('change', draw));
fill();
"""


def render_samples(s: dict[str, Any], field_ko: dict[str, str], judged: list[str],
                   css: str = "") -> list[str]:
    cps = s["checkpoints"]
    if not cps:
        body = ("<h1>캡션 표본</h1><p class=\"muted\">아직 없음 — 판 "
                f"<code>{html.escape(str(s['version']))}</code> 의 진짜 배선 "
                "<code>eval-scored-*.json</code> 이 없다.</p>")
        return _page("캡션 표본", css, body, "{}", "")
    # 요약 표(최종 저장본 · 없으면 첫 저장본) — 판정 표의 차이와 맞춰 보는 자리
    cp0 = "final" if "final" in cps else next(iter(cps))
    c0 = cps[cp0]
    rew = c0["brains"][1:]
    rows = []
    for f, sm in c0["summary"].items():
        mark = " ★" if f in judged else ""
        acc_ok = all(abs((sm["accuracy"][b] or 0) - (c0["file_accuracy"][b].get(f) or 0)) < 1e-9
                     for b in c0["brains"] if f in c0["file_accuracy"][b])
        cells = [f"{html.escape(field_ko.get(f, f))}{mark}", str(sm["n"]),
                 " · ".join(f'{sm["correct"][b]}' for b in c0["brains"]),
                 '<span class="ok">같음</span>' if acc_ok else '<span class="bad">다름</span>',
                 str(sm["only_real_right"]), str(sm["only_real_wrong"]),
                 " · ".join(f'{p["real_only"]}−{p["other_only"]}={p["diff"]:+d}'
                            for p in sm["pair"].values())]
        rows.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
    head = ["칸(★ 판정 칸)", "표본", "맞음 수 " + " · ".join(c0["brains"]),
            "파일 정확도와", "진짜만 맞음", "진짜만 틀림",
            "재배선별 (진짜✓그뇌✗)−(진짜✗그뇌✓) " + " · ".join(rew)]
    th = "".join(f"<th>{html.escape(h)}</th>" for h in head)
    summary = (f'<div class="tbl"><table><thead><tr>{th}</tr></thead><tbody>{"".join(rows)}'
               "</tbody></table></div>")
    miss = sum(c["missing_pairs"] for c in cps.values())
    warn = (f'<p class="bad">재배선 파일에 진짜와 짝이 안 맞는 표본 {miss} 개</p>' if miss else "")
    body = f"""<h1>캡션 표본 — 뇌 넷 나란히</h1>
<p class="sub">판 {html.escape(str(s['version']))} · 채점 평가 per_sample · 읽기 전용 ·
 <a href="index.html">대시보드로</a> · <a href="episodes.html">성적표 에피소드</a></p>
<p>맞으면 <span class="hit">초록</span> · 틀리면 <span class="miss">빨강</span>.
 정답이 없는 칸(예: 물체가 없을 때 물체 쪽)은 세지 않는다.
 <b>진짜만 맞음</b> = 진짜는 맞고 재배선 셋 모두 틀림 ·
 <b>진짜만 틀림</b> = 그 반대. 마지막 열의 차이는 정의상 (진짜 맞음 수 − 그 뇌 맞음 수)다.</p>
<h2>칸별 요약 — {html.escape('최종' if cp0 == 'final' else cp0)}</h2>
{summary}{warn}
<h2>표본</h2>
<div class="ctl">
 <label>저장본 <select id="cp"></select></label>
 <label>칸 <select id="field"></select></label>
 <label>보기 <select id="mode">
  <option value="all">전부</option><option value="only_right">진짜만 맞음</option>
  <option value="only_wrong">진짜만 틀림</option><option value="differ">뇌끼리 갈림</option>
 </select></label>
 <label>에피소드 <select id="ep"></select></label>
 <span class="muted" id="count"></span>
</div>
<div class="tbl"><table><thead id="head"></thead><tbody id="body"></tbody></table></div>
<footer>R-5777 F-001 · 자료는 이 페이지 안에 있다(외부 파일 없음)</footer>"""
    blob = _json_blob({"names": field_ko,
                       "checkpoints": {k: {"brains": v["brains"], "fields": v["fields"],
                                           "rows": v["rows"]} for k, v in cps.items()}})
    return _page("캡션 표본", css, body, blob, SAMPLES_JS)


EPISODES_JS = r"""
const $ = id => document.getElementById(id);
function el(tag, cls, text){ const e=document.createElement(tag); if(cls) e.className=cls;
  if(text!==undefined && text!==null) e.textContent=String(text); return e; }
function num(x, nd){ return (x===null||x===undefined)?'—':Number(x).toFixed(nd); }
const COLS = ['결정','시각 ms','캡션(정답)','물체 방위°','교사','Qwen','좌우','출처','유효',
  '냄새 L','냄새 R','L−R','먹이 방위°','먹이 mm','drive','반사 칸','원문'];
const TIER_COLS = ['층','계기'];
function show(c, seed, td){
  document.querySelectorAll('td.sel').forEach(x=>x.classList.remove('sel'));
  td.classList.add('sel');
  const box = $('dec'); box.innerHTML='';
  box.appendChild(el('h2',null,c+' · 씨앗 '+seed));
  if (!D.has[c]){ box.appendChild(el('p','muted','이 성적표는 결정 기록 전(scorecard-v2) — '
     + '결정 단위 기록은 scorecard-v3 부터 쌓인다.')); return; }
  const rows = (D.dec[c]||{})[seed] || [];
  if (!rows.length){ box.appendChild(el('p','muted','이 에피소드의 결정 줄이 없다'
     + (D.truncated?' (상한 max_decisions 에서 잘림)':''))); return; }
  const mis = rows.filter(r=>r[18]).length;
  const tiered = rows.some(r=>r[16]!==null && r[16]!==undefined);
  const llm = rows.filter(r=>r[16]==='Qwen').length;
  box.appendChild(el('p',null,'결정 '+rows.length+' · 교사와 Qwen 의 도는 쪽이 다른 결정 '+mis
     + (tiered ? ' · LLM 호출 '+llm+' · 호출률 '+(llm/rows.length).toFixed(3) : '')));
  const wrap = el('div','tbl'), tb = el('table'), th = el('tr');
  (tiered ? COLS.slice(0,2).concat(TIER_COLS, COLS.slice(2)) : COLS)
    .forEach(h=>th.appendChild(el('th',null,h)));
  const thead = el('thead'); thead.appendChild(th); tb.appendChild(thead);
  const body = el('tbody');
  for (const r of rows){
    const [d,t,cap,loom,te,ac,src,raw,valid,why,oL,oR,fb,fm,drv,rb,tier,trig,diff] = r;
    const row = el('tr');
    row.appendChild(el('td',null,d)); row.appendChild(el('td',null,t));
    if (tiered){
      row.appendChild(el('td', tier==='Qwen'?'hit':'', tier===null||tier===undefined?'—':tier));
      row.appendChild(el('td',null,trig===null||trig===undefined?'—':trig));
    }
    [cap===null||cap===undefined?'—':D.caps[cap], loom===null?'없음':num(loom,0), te, ac]
      .forEach(v=>row.appendChild(el('td',null,v)));
    row.appendChild(el('td', diff?'miss':'hit', diff?'다름':'같음'));
    row.appendChild(el('td',null,src===null||src===undefined?'—':src));
    const vt = valid===false ? '아니오'+(why?' · '+why:'') : (valid ? '예' : '—');
    row.appendChild(el('td', valid===false?'miss':'', vt));
    row.appendChild(el('td',null,num(oL,2))); row.appendChild(el('td',null,num(oR,2)));
    const lr = (oL===null||oR===null||oL===undefined||oR===undefined) ? '—' : num(oL-oR,2);
    row.appendChild(el('td',null,lr));
    row.appendChild(el('td',null,num(fb,1))); row.appendChild(el('td',null,num(fm,1)));
    row.appendChild(el('td',null,Array.isArray(drv)?drv.map(x=>num(x,2)).join(' · '):num(drv,2)));
    row.appendChild(el('td',null,rb===null||rb===undefined?'—':rb));
    row.appendChild(el('td','raw',raw));
    body.appendChild(row);
  }
  tb.appendChild(body); wrap.appendChild(tb); box.appendChild(wrap);
  box.scrollIntoView({behavior:'smooth'});
}
document.querySelectorAll('td.c').forEach(td =>
  td.addEventListener('click', () => show(td.dataset.c, td.dataset.s, td)));
"""


def render_episodes(e: dict[str, Any], field_ko: dict[str, str], css: str = "") -> list[str]:
    if e.get("path") is None or e.get("error"):
        msg = e.get("error") or "성적표가 아직 없다"
        return _page("성적표 에피소드", css,
                     '<h1>성적표 에피소드</h1>'
                     f'<p class="muted">{html.escape(str(msg))}</p>', "{}", "")
    configs, seeds = e["configs"], e["seeds"]
    th = "<th>씨앗</th>" + "".join(f"<th>{html.escape(c)}</th>" for c in configs)
    body_rows = []
    for s in seeds:
        tds = [f"<td>{s}</td>"]
        for c in configs:
            ep = e["episodes"][c].get(str(s))
            if ep is None:
                tds.append('<td class="dim">—</td>')
                continue
            txt = ("도착" + (f" {ep['reach_ms'] / 1000:.1f} s" if ep.get("reach_ms") else "")
                   if ep["reached"] else
                   (f"{ep['final_food_mm']:.1f} mm" if ep.get("final_food_mm") is not None
                    else "—"))
            cls = "c hit" if ep["reached"] else "c"
            tds.append(f'<td class="{cls}" data-c="{html.escape(c)}" data-s="{s}">'
                       f"{html.escape(txt)}</td>")
        body_rows.append("<tr>" + "".join(tds) + "</tr>")
    tot = ["<td><b>도착 수</b></td>"]
    for c in configs:
        k = e["check"][c]
        mark = ('<span class="ok">= 성적표</span>' if k["match"]
                else f'<span class="bad">성적표 {k["scorecard_reached"]}</span>')
        tot.append(f"<td><b>{k['reached']}</b> / {k['episodes']} {mark}</td>")
    mis = ["<td>도는 쪽 다름</td>"]
    for c in configs:
        k = e["check"][c]
        mis.append(f"<td>{k['turn_mismatch']} / {k['decision_rows']}</td>"
                   if "decision_rows" in k else '<td class="dim">기록 전</td>')
    rows_extra = "<tr>" + "".join(mis) + "</tr>"
    if any("llm_calls" in e["check"][c] for c in configs):       # 층 구조 기록(scorecard-v4)
        llm = ["<td>LLM 호출 · 호출률</td>"]
        for c in configs:
            k = e["check"][c]
            llm.append(f"<td>{k['llm_calls']} / {k['tier_rows']} · {k['llm_rate']:.3f}</td>"
                       if "llm_calls" in k else '<td class="dim">—</td>')
        rows_extra += "<tr>" + "".join(llm) + "</tr>"
    table = (f'<div class="tbl"><table><thead><tr>{th}</tr></thead><tbody>'
             + "".join(body_rows) + "<tr>" + "".join(tot) + "</tr>" + rows_extra
             + "</tbody></table></div>")
    any_dec = any(e["has_decisions"].values())
    note = ("" if any_dec else
            '<p class="warn">이 성적표는 결정 기록 전(scorecard-v2) — 칸을 눌러도 결정 표가 없다. '
            "결정 단위 기록은 다음 성적표(scorecard-v3)부터 쌓인다.</p>")
    trunc = ('<p class="bad">결정 줄이 상한(max_decisions)에서 잘렸다</p>'
             if e.get("truncated") else "")
    body = f"""<h1>성적표 에피소드 — 씨앗 × 구성</h1>
<p class="sub"><code>{html.escape(str(e['path']))}</code> · 규칙 {html.escape(str(e.get('rule')))} ·
 읽기 전용 · <a href="index.html">대시보드로</a> · <a href="samples.html">캡션 표본</a></p>
<p>칸 = 도착(초록 · 도착 시각) 또는 끝났을 때 먹이까지 거리.
 <b>칸을 누르면</b> 그 에피소드의 결정 표 — 교사가 돌라는 쪽과 Qwen 이 돈 쪽이 다르면
 <span class="miss">다름</span>.
 냄새 L · R 은 두 더듬이 값(좌우 차가 신호에 남는지 보는 자리).</p>
{note}{trunc}{table}
<div id="dec"></div>
<footer>R-5777 F-003 · 자료는 이 페이지 안에 있다(외부 파일 없음)</footer>"""
    head = _json_blob({"names": field_ko, "has": e["has_decisions"], "caps": e["captions"],
                       "truncated": e.get("truncated", False)})[:-1]
    parts = [head, ',"dec":{']
    for k, (c, by_seed) in enumerate(e["decisions"].items()):   # 구성마다 따로 — 한 덩어리 금지
        parts += ["," if k else "", _json_blob(c), ":", _json_blob(by_seed)]
    parts.append("}}")
    return _page("성적표 에피소드", css, body, parts, EPISODES_JS)


def _sample_files(runs: Path, version: str, checkpoints: list[str]) -> list[Path]:
    return [runs / f"projector-{b}-{version}" / f"eval-scored-{cp}.json"
            for cp in checkpoints for b in BRAIN_ORDER]


def _episode_files(runs: Path, pattern: str) -> list[Path]:
    sc = _latest_scorecard(runs, pattern)
    return sorted(sc.glob("*.json*")) if sc else []


def build_pages(runs: Path, out_dir: Path, cfg: dict[str, Any], css: str,
                field_ko: dict[str, str], judged: list[str]) -> dict[str, Any]:
    """두 쪽을 만든다 · state.json 에 넣을 작은 요약을 돌려준다.

    원본 지문(경로 · 크기 · 수정 시각)을 **읽기 전에** 보고, 같으면 읽지도 쓰지도 않는다.
    """
    dbg = cfg.get("debug") or {}
    version = str(cfg.get("train_rule"))
    cps = list(dbg.get("checkpoints") or ["final"])
    sp = out_dir / dbg.get("samples_page", "samples.html")
    sig = ("samples", version, _sig(_sample_files(runs, version, cps)), tuple(judged))
    s_sum = _cached(sp, sig)
    if s_sum is None:
        samples = read_samples(runs, version, cps, list(field_ko),
                               int(dbg.get("max_samples", 5000)))
        s_sum = {cp: {"brains": c["brains"], "summary": c["summary"],
                      "missing_pairs": c["missing_pairs"]}
                 for cp, c in samples["checkpoints"].items()}
        _write(sp, render_samples(samples, field_ko, judged, css), sig, s_sum)
        del samples
    ep = out_dir / dbg.get("episodes_page", "episodes.html")
    pattern = cfg["globs"]["scorecard"]
    sig = ("episodes", _sig(_episode_files(runs, pattern)))
    e_sum = _cached(ep, sig)
    if e_sum is None:
        eps = read_episodes(runs, pattern, int(dbg.get("max_decisions", 20000)),
                            int(dbg.get("raw_chars", 200)))
        e_sum = {"path": eps.get("path"), "check": eps.get("check"),
                 "has_decisions": eps.get("has_decisions"), "truncated": eps.get("truncated")}
        _write(ep, render_episodes(eps, field_ko, css), sig, e_sum)
        del eps
    return {"samples_page": sp.name, "episodes_page": ep.name, "samples": s_sum,
            "episodes": e_sum}
