#!/usr/bin/env python3
"""
초파리 뇌 학습 안내서 — 마크다운을 읽기용 HTML 로 변환한다.

정본은 같은 폴더의 *.md 다. 같은 이름의 .html 을 만든다(README.md → index.html).
왼쪽 고정 메뉴는 1부(초파리 뇌) · 2부(Qwen 연결) · 부록으로 나눈다.
문서 사이 링크(.md)는 .html 로 바꾸고, 이 폴더 밖 문서는 **HTML 사본이 있을 때만** 바꾼다
(연구 노트는 사본이 있고, 요구사항 문서는 마크다운 그대로 연다).
모양(CSS)은 연구 노트 build.py 와 같다.

사용법:
    uv run --no-project --with markdown==3.11.0 python docs/training/build.py

python-markdown 판은 3.11.0 으로 고정한다 — 연구 노트 build.py 와 같은 까닭(3.11.1 부터 닫는 ** 앞이
괄호·낫표이고 뒤에 한글이 붙은 굵게를 풀지 못함). 다른 판이면 HTML 을 만들지 않고 멈춘다.

문서를 더하려면 아래 GROUPS 에 (파일 이름, 메뉴 표시명) 을 넣는다.
"""
from __future__ import annotations

import html as H
import re
import sys
import unicodedata
from pathlib import Path

MARKDOWN_VERSION = "3.11.0"  # 연구 노트 build.py 와 같은 판(위 독스트링)
RUN = f"uv run --no-project --with markdown=={MARKDOWN_VERSION} python docs/training/build.py"

try:
    import markdown
except ImportError:  # 안내만 하고 끝낸다
    sys.exit(f"python-markdown 이 없습니다. `{RUN}` 로 실행하십시오.")
if markdown.__version_info__[:3] != tuple(map(int, MARKDOWN_VERSION.split("."))):  # 조용히 깨진 HTML 을 만들지 않는다
    sys.exit(f"python-markdown {markdown.__version__} 입니다 — 이 빌드는 {MARKDOWN_VERSION} 고정입니다. `{RUN}` 로 실행하십시오.")

ROOT = Path(__file__).resolve().parent

# 메뉴 = 읽는 순서 (묶음 이름, [(파일, 표시명)])
GROUPS = [
    ("시작", [("README.md", "안내서 시작 — 지도와 상태"),
              ("00-초파리-뇌-쉽게-보기.md", "00 · 초파리 뇌 쉽게 보기")]),
    ("1부 · 초파리 뇌", [
        ("01-뇌-재료-공식-데이터와-그래프.md", "01 · 뇌 재료 — 데이터와 그래프"),
        ("02-뇌-엔진-LIF.md", "02 · 뇌 엔진 — LIF"),
        ("03-뇌-맞추기와-검산.md", "03 · 뇌 맞추기와 검산"),
        ("04-감각-넣기와-신호-읽기.md", "04 · 감각 넣기와 신호 읽기"),
        ("05-회로를-학습시키려면.md", "05 · 회로를 학습시키려면(후보)"),
    ]),
    ("2부 · Qwen 연결", [
        ("06-학습-자료-모으기.md", "06 · 학습 자료 모으기"),
        ("07-투영기와-Qwen-잇기.md", "07 · 투영기와 Qwen 잇기"),
        ("08-투영기-학습-절차.md", "08 · 투영기 학습 절차"),
        ("09-평가와-대조군.md", "09 · 평가와 대조군"),
        ("10-학습-기록과-다음.md", "10 · 학습 기록 · 결과 · 다음"),
    ]),
    ("부록", [
        ("11-명령-모음과-용어.md", "11 · 명령 모음과 용어"),
        ("12-재현-절차와-다음-작업.md", "12 · 재현 절차와 다음 작업"),
    ]),
]
ORDER = [name for _, items in GROUPS for name, _ in items]
SHORT = {name: label for _, items in GROUPS for name, label in items}
RELATED = [
    ("../research/fly-connectome/index.html", "연구 노트(설계 · 배경)"),
    ("../research/fly-connectome/11-구현-초파리-신호를-Qwen-에.html", "연구 노트 11 · 구현"),
    ("../datasets.md", "학습 자료 형식"),
    ("../requirements/echomind-fly-connectome-handoff.md", "인수인계(진행 상황)"),
]


def html_name(md_name: str) -> str:
    return "index.html" if md_name == "README.md" else md_name[:-3] + ".html"


CSS = """
  :root {
    --ground:#F1F4F5; --panel:#FAFBFB; --side:#E9EEF0; --plate:#E7ECEE; --ink:#12171A; --ink-2:#414A4F; --ink-3:#6B757A;
    --rule:#D3DADD; --rule-2:#B6C0C4; --a:#0E7C86; --a-soft:#DCEFF1; --b:#AB2A79; --b-soft:#F7E3EF;
    --code-bg:#E9EEF0; --code-ink:#1B2225;
    --f-display:"Gowun Batang","AppleMyungjo","Nanum Myeongjo",serif;
    --f-body:"Gothic A1","Apple SD Gothic Neo","Malgun Gothic",system-ui,sans-serif;
    --f-mono:"JetBrains Mono","Gothic A1",ui-monospace,Menlo,monospace;
    --side-w: 272px;
  }
  @media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
    --ground:#0E1214; --panel:#161B1E; --side:#121719; --plate:#1C2327; --ink:#E6EBEC; --ink-2:#B2BCC0; --ink-3:#859096;
    --rule:#263036; --rule-2:#3A464C; --a:#4ECDD6; --a-soft:#102E33; --b:#F07AC0; --b-soft:#33122A;
    --code-bg:#171E21; --code-ink:#D7DEE1; } }
  :root[data-theme="dark"] {
    --ground:#0E1214; --panel:#161B1E; --side:#121719; --plate:#1C2327; --ink:#E6EBEC; --ink-2:#B2BCC0; --ink-3:#859096;
    --rule:#263036; --rule-2:#3A464C; --a:#4ECDD6; --a-soft:#102E33; --b:#F07AC0; --b-soft:#33122A;
    --code-bg:#171E21; --code-ink:#D7DEE1; }
  *,*::before,*::after{box-sizing:border-box}
  html,body{height:100%}
  body{margin:0;background:var(--ground);color:var(--ink);font-family:var(--f-body);font-size:1.0625rem;line-height:1.8;
       word-break:keep-all;overflow-wrap:break-word}
  a{color:var(--a);text-underline-offset:3px;text-decoration-thickness:1px}
  a:focus-visible{outline:2px solid var(--b);outline-offset:2px}

  /* 왼쪽 고정 메뉴 + 오른쪽 본문, 브라우저 전체 */
  .shell{display:grid;grid-template-columns:var(--side-w) minmax(0,1fr);min-height:100%}
  nav.side{position:sticky;top:0;height:100vh;overflow-y:auto;background:var(--side);border-right:1px solid var(--rule);
           padding:28px 22px 32px;font-size:.9rem;line-height:1.5}
  nav.side .brand{font-family:var(--f-display);font-weight:700;font-size:1.05rem;line-height:1.35;margin:0 0 4px}
  nav.side .sub{font-family:var(--f-mono);font-size:.7rem;letter-spacing:.08em;color:var(--ink-3);margin:0 0 22px}
  nav.side ol{list-style:none;margin:0;padding:0;display:grid;gap:2px}
  nav.side li a{display:block;padding:8px 10px;border-radius:4px;text-decoration:none;color:var(--ink-2)}
  nav.side li a:hover{background:var(--plate);color:var(--ink)}
  nav.side li a.cur{background:var(--b-soft);color:var(--b);font-weight:700}
  nav.side .grp{font-family:var(--f-mono);font-size:.68rem;letter-spacing:.1em;color:var(--ink-3);margin:18px 0 6px 10px}
  nav.side .foot{margin-top:24px;padding-top:14px;border-top:1px solid var(--rule);font-size:.78rem;color:var(--ink-3);line-height:1.6}
  nav.side .foot a{color:var(--ink-2)}
  main{min-width:0;padding:clamp(28px,4vw,56px) clamp(20px,5vw,72px) 96px}
  .page{max-width:none}   /* 본문이 브라우저 전체 폭을 쓴다 */
  @media (max-width:860px){
    .shell{grid-template-columns:minmax(0,1fr)}
    nav.side{position:static;height:auto;border-right:0;border-bottom:1px solid var(--rule);padding:18px 16px}
    nav.side ol{display:flex;flex-wrap:wrap;gap:4px}
    nav.side li a{padding:6px 10px;font-size:.82rem}
    nav.side .grp{display:none}
  }

  h1{font-family:var(--f-display);font-weight:700;font-size:clamp(1.8rem,1.2rem+2.4vw,2.6rem);line-height:1.25;margin:0 0 18px;text-wrap:balance}
  h2{font-family:var(--f-display);font-weight:700;font-size:1.6rem;line-height:1.35;margin:52px 0 12px;text-wrap:balance}
  h3{font-size:1.2rem;font-weight:700;margin:34px 0 10px;line-height:1.45}
  h4{font-size:1.05rem;font-weight:700;margin:24px 0 8px}
  p{margin:0 0 16px} ul,ol{margin:0 0 16px;padding-left:1.2em} li{margin:6px 0}
  blockquote{border-left:3px solid var(--b);background:var(--b-soft);margin:22px 0;padding:14px 20px;font-size:.92rem;line-height:1.75}
  blockquote p:last-child{margin-bottom:0}
  hr{border:0;border-top:1px solid var(--rule);margin:40px 0}
  code,pre{font-variant-ligatures:none}   /* <|vision_pad|> 같은 글자가 합자로 바뀌지 않게 */
  code{font-family:var(--f-mono);font-size:.86em;background:var(--code-bg);padding:.1em .36em;border-radius:2px}
  pre{margin:16px 0 20px;padding:14px 16px;overflow-x:auto;background:var(--code-bg);border:1px solid var(--rule);border-radius:3px;
      font-family:var(--f-mono);font-size:.8rem;line-height:1.7;color:var(--code-ink)}
  pre code{background:none;padding:0;font-size:inherit}
  .tbl{overflow-x:auto;margin:20px 0;border:1px solid var(--rule);border-radius:3px;background:var(--panel)}
  table{border-collapse:collapse;width:100%;font-size:.9rem;line-height:1.6;min-width:560px}
  th,td{text-align:left;vertical-align:top;padding:11px 16px;border-bottom:1px solid var(--rule)}
  tr:last-child td{border-bottom:0}
  th{font-family:var(--f-mono);font-weight:400;font-size:.72rem;letter-spacing:.08em;color:var(--ink-3);white-space:nowrap;background:var(--plate)}
  img{max-width:100%;height:auto;display:block;margin:22px 0;border:1px solid var(--rule);border-radius:3px;background:#fff}
  .pager{display:flex;justify-content:space-between;gap:16px;margin-top:56px;padding-top:18px;border-top:1px solid var(--rule);font-size:.9rem}
  .pager a{text-decoration:none}
  footer{margin-top:24px;font-size:.8rem;color:var(--ink-3);line-height:1.7}
"""

PAGE = """<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Gowun+Batang:wght@400;700&family=Gothic+A1:wght@400;500;700&family=JetBrains+Mono:wght@400;500&display=swap">
<style>{css}</style>
</head>
<body>
<div class="shell">
<nav class="side" aria-label="문서 메뉴">
  <p class="brand">초파리 뇌 학습 안내서</p>
  <p class="sub">R-3340 · 2026-09-23</p>
  {toc}
  <div class="grp">관련 문서</div>
  <ol>{related}</ol>
  <div class="foot">정본은 마크다운 — <a href="{md}">{md}</a><br>HTML 은 읽기용 사본</div>
</nav>
<main>
<div class="page">
{body}
<div class="pager">{prev}{next}</div>
<footer>이 HTML 은 <code>{md}</code> 에서 <code>build.py</code> 로 만든 읽기용 사본입니다. 고칠 때는 마크다운을 고치고 다시 변환합니다.</footer>
</div>
</main>
</div>
</body>
</html>
"""



def _link(m: re.Match) -> str:
    """`](경로.md#앵커)` → HTML 사본이 있으면 .html 로, 없으면 그대로."""
    path, frag = m.group(1), m.group(2) or ""
    target = (ROOT / path).resolve()
    html = target.with_name("index.html" if target.name == "README.md" else target.stem + ".html")
    if target.parent == ROOT or html.exists():
        new = Path(path).with_name(html.name).as_posix()
        return f"]({new}{frag})"
    return m.group(0)


def convert(name: str, md: "markdown.Markdown") -> str:
    raw = (ROOT / name).read_text(encoding="utf-8")
    text = re.sub(r"\]\(([^)#\s]+\.md)(#[^)]*)?\)", _link, raw)
    md.reset()
    body = md.convert(text).replace("<table>", '<div class="tbl"><table>').replace("</table>", "</table></div>")

    m = re.search(r"^#\s+(.+)$", raw, re.M)
    title = re.sub(r"^\d+\.\s*", "", m.group(1).strip()) if m else name

    toc = []
    for group, items in GROUPS:
        lis = []
        for n, label in items:
            cls = ' class="cur"' if n == name else ""
            lis.append(f'<li><a href="{html_name(n)}"{cls}>{H.escape(label)}</a></li>')
        toc.append(f'<div class="grp">{H.escape(group)}</div>\n  <ol>{"".join(lis)}</ol>')
    related = "\n".join(f'<li><a href="{p}">{H.escape(t)}</a></li>' for p, t in RELATED)

    i = ORDER.index(name)
    prev_l = f'<a href="{html_name(ORDER[i-1])}">← {H.escape(SHORT[ORDER[i-1]])}</a>' if i > 0 else "<span></span>"
    next_l = f'<a href="{html_name(ORDER[i+1])}">{H.escape(SHORT[ORDER[i+1]])} →</a>' if i < len(ORDER) - 1 else "<span></span>"

    return PAGE.format(title=H.escape(title), css=CSS, toc="\n  ".join(toc), related=related,
                       md=H.escape(name), body=body, prev=prev_l, next=next_l)


def gh_slug(value: str, separator: str) -> str:
    """GitHub 식 제목 id — 소문자, 문장부호 제거, 빈칸 하나가 하이픈 하나. 마크다운의 `…md#…` 링크와 맞춘다."""
    value = unicodedata.normalize("NFC", value)
    value = re.sub(r"[^\w\s-]", "", value).strip().lower()
    return re.sub(r"\s", separator, value)


def check_links() -> int:
    """만든 HTML 안의 상대 링크·그림이 실제로 있는지 센다. 깨진 수를 돌려준다."""
    import urllib.parse
    bad = 0
    for f in [ROOT / html_name(n) for n in ORDER]:
        for m in re.finditer(r'(?:href|src)="([^"#]+)(#[^"]*)?"', f.read_text(encoding="utf-8")):
            u = m.group(1)
            if u.startswith("http"):
                continue
            target = ROOT / urllib.parse.unquote(u)
            if not target.exists():
                bad += 1
                print(f"  깨진 링크: {f.name} → {u}")
                continue
            frag = m.group(2)
            if frag and target.suffix == ".html":
                anchor = urllib.parse.unquote(frag[1:])
                if f'id="{anchor}"' not in target.read_text(encoding="utf-8"):
                    bad += 1
                    print(f"  깨진 단락 링크: {f.name} → {u}{frag}")
    return bad


def main() -> int:
    md = markdown.Markdown(extensions=["tables", "fenced_code", "sane_lists", "toc"],
                           extension_configs={"toc": {"slugify": gh_slug, "toc_depth": "1-4"}})
    missing = [n for n in ORDER if not (ROOT / n).exists()]
    if missing:
        sys.exit(f"없는 마크다운: {missing}")
    for n in ORDER:
        out = ROOT / html_name(n)
        out.write_text(convert(n, md), encoding="utf-8")
        print(f"만듦  {out.name}  ({out.stat().st_size:,} bytes)")
    bad = check_links()
    print("깨진 링크 없음" if bad == 0 else f"깨진 링크 {bad}개")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
