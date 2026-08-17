# -*- coding: utf-8 -*-
"""단일 HTML 빌드 — 소스 조각 + 라이브러리 + 데이터를 하나로 합친다.

  python3 viz/make_html.py            → viz/천안_배치시뮬레이터.html

조립 순서가 중요하다:
  1) app_head.html  : CSS(MapLibre CSS 를 앞에 주입) + DOM 뼈대 + <script> 시작
  2) MapLibre GL JS : 내장 (CDN 없이 오프라인 동작)
  3) 데이터 상수     : D(대시보드) G(간트) M(지도·시뮬) GW(GNN 가중치)
  4) app_js.js      : 공통 유틸(el/svg/css/tip) + 탭B·탭C 패널 + render()
  5) map_js.js      : 지도 · 유형 재분류 · 하루 시뮬 · 애니메이션
  6) gnn_js.js      : GNN 추론 (map_js 의 SIM/RG/DAILY 를 참조하므로 뒤에 와야 한다)
  7) render() 호출  : 모든 함수가 정의된 뒤 마지막에 한 번
"""
import json, os, sys

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)
def rd(p):
    with open(os.path.join(ROOT, p), encoding="utf-8") as f: return f.read()

VARIANT = sys.argv[1] if len(sys.argv) > 1 else "hosp"   # base | sev | hosp

head  = rd("viz/app_head.html")
mlcss = rd("lib_maplibre.css")          # 이전 관제 시각화에서 추출한 MapLibre GL CSS
mljs  = rd("lib_maplibre.js")           # 〃 JS (deck.gl 은 쓰지 않는다 — 캔버스 오버레이로 대체)
app   = rd("viz/app_js.js").replace("render();\n", "")   # render() 는 마지막에 따로 호출
mp    = rd("viz/map_js.js")
gn    = rd("viz/gnn_js.js")

D  = rd("out/viz_dash.json")            # export_viz.py 산출
G  = rd("out/viz_graph.json")           # 〃 (간트용 하루치 이벤트)
M  = rd("out/viz_map.json")             # export_map.py 산출 (경계·ETA 행렬·시뮬 파라미터)
GW = rd(f"out/gnn_weights_{VARIANT}.json")   # train_gnn.py 산출 (가중치 + 정규화 통계)

# 탭2의 GNN 수치를 이번 가중치의 test 성적으로 갱신
w = json.loads(GW); d = json.loads(D)
d["model"]["gnn"] = {"auc": w["test"]["auc"], "mae": w["test"]["mae_pos"], "rho": w["spearman"]}
D = json.dumps(d, ensure_ascii=False, separators=(",", ":"))

head = head.replace("<style>", "<style>\n/* --- MapLibre GL CSS (내장) --- */\n" + mlcss + "\n/* --- app --- */\n", 1)
html = (head
        + "/* --- MapLibre GL JS (내장) --- */\n" + mljs + "\n"
        + f"const D={D};\nconst G={G};\nconst M={M};\nconst GW={GW};\n"
        + app + "\n" + mp + "\n" + gn + "\nrender();\n"
        + "</script>\n</body>\n</html>\n")

out = os.path.join(BASE, "천안_배치시뮬레이터.html")
with open(out, "w", encoding="utf-8") as f: f.write(html)
print(f"{out}  {os.path.getsize(out)/1024/1024:.2f} MB  (variant={VARIANT})")

# 문법 검사 — node 가 있으면 <script> 본문만 떼어 확인
try:
    import subprocess, tempfile
    js = html.split("<script>")[1].split("</script>")[0]
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False, encoding="utf-8") as t:
        t.write(js); tmp = t.name
    r = subprocess.run(["node", "--check", tmp], capture_output=True, text=True)
    print("JS 문법:", "OK" if r.returncode == 0 else r.stderr[:400])
except FileNotFoundError:
    print("JS 문법: node 없음 — 건너뜀")
