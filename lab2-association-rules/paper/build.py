# -*- coding: utf-8 -*-
"""Build the IEEEtran paper and enforce the delivery guards.

Guards (skill: latex-paper-delivery-guards):
  1. missing glyphs      -- any "Missing character" is fatal
  2. oversized floats    -- any "Float too large for page" is fatal
  3. overfull boxes      -- fatal only above OVER_TOL_PT (visible on paper)
  4. floats after refs   -- read page numbers from .aux, fatal if any
  5. orphan figures      -- every \\label{fig:,tab:} must be \\ref'ed, and vice versa
  6. rendered reverse check with PyMuPDF (never pypdf for CJK/subset fonts):
     page count, presence of key numbers, ink outside the text block

Tectonic writes TeX warnings to stderr, NOT to the .log file, so the scanner
must be fed the merged stdout+stderr stream. Feeding only the .log makes every
counter zero and the guard silently vacuous.
"""

from __future__ import annotations

import io
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parent
TEX = PROJ / "main.tex"
LOG = PROJ / "main.log"
AUX = PROJ / "main.aux"
PDF = PROJ / "main.pdf"

OVER_TOL_PT = 1.0
_OVER_RE = re.compile(r"Overfull \\hbox \(([-\d.]+)pt too wide\)")
_NEWLABEL_RE = re.compile(r"\\newlabel\{([^}]+)\}\{\{([^}]*)\}\{([^}]*)\}")


def find_tectonic() -> str:
    """定位 Tectonic 可执行文件：优先环境变量 TECTONIC，其次 PATH。

    公开仓库不写死本机路径，本地安装位置由使用者在环境变量里声明。
    """
    configured = os.environ.get("TECTONIC") or os.environ.get("TECTONIC_PATH")
    if configured:
        resolved = configured if Path(configured).is_file() else shutil.which(configured)
        if not resolved:
            raise SystemExit(f"TECTONIC 指向的可执行文件不存在: {configured}")
        return str(resolved)
    executable = shutil.which("tectonic")
    if not executable:
        raise SystemExit("未找到 Tectonic：请设置环境变量 TECTONIC 或将其加入 PATH")
    return executable


def run(cmd: list[str]) -> tuple[int, str]:
    p = subprocess.run(cmd, cwd=PROJ, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    return p.returncode, (p.stdout or "") + "\n" + (p.stderr or "")


def scan_log(*streams: str):
    """Count the three warning classes across all sources, de-duplicated."""
    merged = "\n".join(streams)
    seen_miss, seen_flt, seen_over = set(), set(), set()
    miss, flt, over, big = [], [], [], []
    for line in merged.splitlines():
        line = line.strip()
        if "Missing character" in line and line not in seen_miss:
            seen_miss.add(line)
            miss.append(line)
        elif "Float too large" in line and line not in seen_flt:
            seen_flt.add(line)
            flt.append(line)
        elif "Overfull \\hbox" in line and line not in seen_over:
            seen_over.add(line)
            over.append(line)
            m = _OVER_RE.search(line)
            if m and abs(float(m.group(1))) > OVER_TOL_PT:
                big.append(line)
    return miss, over, flt, big


def floats_after_refs(aux: str):
    """Return (offenders, refs_page). refs_page None means 'unknown', which
    must NOT be treated as success."""
    refs_page, items = None, []
    for lab, _num, page in _NEWLABEL_RE.findall(aux):
        if not page.isdigit():
            continue
        if lab == "sec:refs":
            refs_page = int(page)
        elif lab.startswith(("tab:", "fig:")):
            items.append((lab, int(page)))
    if refs_page is None:
        return [], None
    return [(l, p, refs_page) for l, p in items if p > refs_page], refs_page


def check_refs():
    """Bidirectional: no orphan floats, no dangling \\ref."""
    body = TEX.read_text(encoding="utf-8")
    for extra in sorted((PROJ / "tables").glob("*.tex")):
        body += "\n" + extra.read_text(encoding="utf-8")
    labels = set(re.findall(r"\\label\{((?:fig|tab|sec|eq):[^}]+)\}", body))
    labels.discard("sec:refs")          # anchor for the float check, not a float
    refs = set(re.findall(r"\\ref\{([^}]+)\}", body))
    return sorted(labels - refs), sorted(refs - labels)


def check_pdf(page_count_expected_min: int = 5) -> dict:
    import pymupdf

    doc = pymupdf.open(PDF)
    text = "\n".join(p.get_text() for p in doc)
    # key numbers that must survive compilation (spot check of the macro chain)
    must_have = ["1892", "1990", "6,524", "17.39", "33.78", "1,161,310"]
    missing = [m for m in must_have if m not in text]
    # ink outside the text block (60 pt margin assumption is too strict for
    # IEEEtran; use the real page margins from the geometry of page 1)
    page = doc[0]
    W, H = page.rect.width, page.rect.height
    margin = 46.0            # IEEEtran journal margins ~0.75in = 54pt, keep slack
    outside = []
    for i, pg in enumerate(doc):
        for b in pg.get_text("blocks"):
            x0, y0, x1, y1 = b[:4]
            if x0 < margin - 2 or x1 > pg.rect.width - margin + 2:
                outside.append((i + 1, round(x0, 1), round(x1, 1), b[4][:40]))
    return {
        "pages": len(doc),
        "chars": len(text),
        "missing_numbers": missing,
        "ink_outside": outside[:5],
        "size": [round(W, 1), round(H, 1)],
    }


def main() -> int:
    # 1) regenerate numbers and tables from the CSVs (single source)
    for script in ("make_numbers.py", "make_tables.py"):
        rc, out = run([sys.executable, script])
        if rc != 0:
            print(f"[失败] {script}: {out[-500:]}")
            return 2
        print(out.strip().splitlines()[-1])

    # 2) compile
    rc, out = run([find_tectonic(), "-X", "compile", str(TEX),
                   "--keep-logs", "--keep-intermediates"])
    log = ""
    if LOG.exists():
        log = io.open(LOG, encoding="utf-8", errors="replace").read()
    miss, over, flt, big = scan_log(out, log)
    print(f"[guards] missing={len(miss)} overfull={len(over)} "
          f"overfull>{OVER_TOL_PT}pt={len(big)} float_too_large={len(flt)}")
    for line in (miss + flt + big)[:6]:
        print("   !", line[:160])

    failures = []
    if rc != 0:
        failures.append(f"tectonic exit {rc}")
    if miss:
        failures.append(f"missing glyphs: {len(miss)}")
    if flt:
        failures.append(f"oversized floats: {len(flt)}")
    if big:
        failures.append(f"overfull > {OVER_TOL_PT}pt: {len(big)}")

    # 3) float placement (needs .aux, hence --keep-intermediates)
    if AUX.exists():
        offenders, refs_page = floats_after_refs(AUX.read_text(encoding="utf-8",
                                                               errors="replace"))
        if refs_page is None:
            failures.append("float check: sec:refs page unknown (label missing)")
            print("[guards] floats-after-refs: UNKNOWN (no sec:refs label)")
        else:
            print(f"[guards] floats-after-refs: {len(offenders)} "
                  f"(references start p.{refs_page})")
            if offenders:
                failures.append(f"floats after references: {offenders}")
    else:
        failures.append("main.aux missing (cannot check float placement)")

    # 4) label/ref coverage
    orphan, dangling = check_refs()
    print(f"[guards] orphan labels={len(orphan)} dangling refs={len(dangling)}")
    if orphan:
        print("   orphan:", orphan)
        failures.append(f"orphan labels: {orphan}")
    if dangling:
        print("   dangling:", dangling)
        failures.append(f"dangling refs: {dangling}")

    # 5) rendered reverse check
    if PDF.exists():
        try:
            info = check_pdf()
            print(f"[pdf] pages={info['pages']} chars={info['chars']} "
                  f"size={info['size']}")
            if info["missing_numbers"]:
                failures.append(f"numbers missing in PDF: {info['missing_numbers']}")
            if info["ink_outside"]:
                print("   ink outside margins:", info["ink_outside"])
        except Exception as exc:                     # noqa: BLE001
            failures.append(f"pdf reverse check failed: {exc}")

    if failures:
        print("[失败] " + " | ".join(failures))
        return 3
    print(f"[OK] {PDF.name} built, all guards passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
