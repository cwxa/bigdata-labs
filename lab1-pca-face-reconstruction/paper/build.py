# -*- coding: utf-8 -*-
"""IEEE paper build: numbers generated from the experiment console (single source),
Tectonic compile, and the three delivery guards (missing chars / big floats / overfull),
plus float-after-references and PyMuPDF reverse checks.

The default build reuses recorded data and figures. Experimental refreshes are
opt-in through --refresh-data or --refresh-figures.
"""
import argparse
from contextlib import contextmanager
import json
import logging
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

PROJ = Path(__file__).resolve().parent
BASE = PROJ.parent
CONSOLE_CN = BASE / 'results' / 'console_output.txt'
CONSOLE_EN = BASE / 'results_en' / 'console_output.txt'
OUT_PDF = PROJ / 'main.pdf'   # Tectonic 直接编译到 paper/ 下，与实验二目录布局保持一致
OVER_TOL_PT = 1.0
LOGGER = logging.getLogger('paper_build')
_OVER_RE = re.compile(r'Overfull \\[hv]box \(([-\d.]+)pt too (?:wide|high)\)')
_NEWLABEL_RE = re.compile(r'\\newlabel\{([^}]+)\}\{\{.*?\}\{(\d+)\}')


class BuildError(RuntimeError):
    """A delivery requirement was not met."""


@contextmanager
def stage(name):
    # 统一记录阶段入口、出口及耗时；主入口集中报告失败原因。
    started = time.perf_counter()
    LOGGER.info('Starting %s', name)
    try:
        yield
    except Exception:
        LOGGER.error('Failed %s after %.2f s', name, time.perf_counter() - started)
        raise
    else:
        LOGGER.info('Completed %s in %.2f s', name, time.perf_counter() - started)


def read_text(path):
    return Path(path).read_text(encoding='utf-8', errors='replace')


def grab(text, pattern, label):
    match = re.search(pattern, text)
    if not match:
        raise BuildError('Cannot parse %s with pattern %s' % (label, pattern))
    return match.groups() if match.groups() else (match.group(0),)


def run_python(script, timeout):
    LOGGER.info('Running optional refresh script=%s timeout=%d s', script, timeout)
    result = subprocess.run(
        [sys.executable, str(PROJ / script)], cwd=PROJ,
        capture_output=True, text=True, encoding='utf-8', errors='replace',
        timeout=timeout,
    )
    if result.stderr.strip():
        LOGGER.info('Refresh diagnostics: %s', result.stderr.strip()[-2500:])
    if result.returncode:
        raise BuildError('%s failed (exit=%d): %s' %
                         (script, result.returncode, result.stdout[-1500:]))
    return result.stdout


# ---------------- 1. numbers from console (single source) ----------------
def refresh_data():
    """Recreate numeric macros and tables only when explicitly requested."""
    cn, en = read_text(CONSOLE_CN), read_text(CONSOLE_EN)
    sweep = {}
    for match in re.finditer(
            r'^\s+(\d+)\(基线\)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)%\s*$', cn, re.M):
        sweep[0] = tuple(float(match.group(i)) for i in (2, 4, 5))
    for match in re.finditer(
            r'^\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)%\s*$', cn, re.M):
        value = int(match.group(1))
        if value:
            sweep[value] = tuple(float(match.group(i)) for i in (2, 4, 5))
    if not {0, 5, 10, 20, 50, 100}.issubset(sweep):
        raise BuildError('Incomplete k-sweep table: %s' % sorted(sweep))
    ssim_line = grab(cn, r'扩展分析9[^\n]*', 'SSIM line')[0]
    ssim = {int(k): float(v) for k, v in re.findall(r'k=(\d+): ([\d.]+)', ssim_line)}
    acc_line = grab(cn, r'扩展分析10[^\n]*', '1-NN line')[0]
    acc = {int(k): float(v) for k, v in re.findall(r'k=(\d+): ([\d.]+)', acc_line)}
    pix_w, pix_b, pix_r = grab(cn, r'像素空间（4096 维）：类内 ([\d.]+) / 类间 ([\d.]+) → 比值 ([\d.]+)', 'pixel separation')
    pca_w, pca_b, pca_r = grab(cn, r'PCA 空间（20 维）：类内 ([\d.]+) / 类间 ([\d.]+) → 比值 ([\d.]+)', 'PCA separation')
    predk, obsk, reldev = grab(cn, r'理论预测 MSE\(k=20\) = ([\d.]+)｜观测 = ([\d.]+)｜相对偏差 ([\d.]+)%', 'logged theory comparison')
    eq_raw = float(grab(cn, r'最大绝对差 = ([\d.eE+-]+)', 'solver agreement')[0])
    svd_m, svd_lo, svd_hi = grab(cn, r'SVD\s+中位 ([\d.]+) s（范围 ([\d.]+)–([\d.]+) s）', 'SVD timing')
    eig_m, eig_lo, eig_hi = grab(cn, r'协方差\+eigh 中位 ([\d.]+) s（范围 ([\d.]+)–([\d.]+) s）', 'eigh timing')
    svd_m2 = grab(en, r'SVD\s+median ([\d.]+) s \(range ([\d.]+)–([\d.]+) s\)', 'second SVD timing')[0]
    eig_m2 = grab(en, r'Cov \+ eigh\s+median ([\d.]+) s \(range ([\d.]+)–([\d.]+) s\)', 'second eigh timing')[0]
    _, mse_std = grab(cn, r'k=20 逐图 MSE：均值 ([\d.]+) ± 标准差 ([\d.]+)', 'per-image MSE')

    # held-out sanity check + measured latency (single source; also feeds the k=0 SSIM cell)
    # 默认构建不进入此分支；显式刷新才重算留出指标和延迟。
    extra = json.loads(run_python('extra_exp.py', 600).strip().splitlines()[-1])
    LOGGER.info('Refreshed split=%d/%d test_MSE=%.5f accuracy=%.1f%% latency=%.2f ms/image',
                extra['n_train'], extra['n_test'], extra['test_mse'], extra['test_acc'], extra['latency_ms'])
    n, dim, k = 400, 4096, 20
    var_ks = float(grab(cn, r'前20个主成分累计方差贡献率: ([\d.]+)%', 'retained variance')[0])
    mse_base, (mse_k, psnr_k, _) = sweep[0][0], sweep[k]
    speedup = float(eig_m) / float(svd_m)
    exponent = int(math.floor(math.log10(eq_raw)))
    mantissa = eq_raw / (10 ** exponent)
    macros = {
        'nN': str(n), 'nDim': str(dim), 'nK': str(k),
        'nVarKsSum': '%.2f\\%%' % var_ks,
        'nCompRatio': '%.2f\\%%' % (k / dim * 100),
        'nMSEBase': '%.5f' % mse_base, 'nMSEK': '%.5f' % mse_k,
        'nPSNRK': '%.3f' % psnr_k,
        'nDrop': '%.1f\\%%' % ((1 - mse_k / mse_base) * 100),
        # 原日志预测未乘 (n-1)/n；保留记录，由正文解释归一化差异。
        'nPredK': '%.6f' % float(predk), 'nObsK': '%.6f' % float(obsk),
        'nRelDev': '%s\\%%' % reldev,
        'nSsimK': '%.3f' % ssim[k], 'nAccK': '%.1f\\%%' % acc[k],
        'nAccSat': '%.1f\\%%' % acc[50],
        'nSepPix': pix_r, 'nSepPCA': pca_r,
        'nWithinPix': pix_w, 'nBetweenPix': pix_b,
        'nWithinPCA': pca_w, 'nBetweenPCA': pca_b,
        'nEqDiff': '$%.2f\\times10^{%d}$' % (mantissa, exponent),
        'nSvdMed': svd_m, 'nSvdLow': svd_lo, 'nSvdHigh': svd_hi,
        'nEighMed': '%.1f' % float(eig_m), 'nEighLow': '%.1f' % float(eig_lo),
        'nEighHigh': '%.1f' % float(eig_hi),
        'nSvdMedTwo': svd_m2, 'nEighMedTwo': '%.1f' % float(eig_m2),
        'nSpeedup': str(round(speedup)), 'nMSEstd': mse_std,
        'nCovMem': '%d\\,MB' % round(dim * dim * 8 / 1e6),
        'nStorageOrig': '%.1f\\,MB' % (n * dim * 8 / 1e6),
        'nStorageKB': '%d\\,kB' % round(n * k * 8 / 1e3),
        'nCompress': str(round(dim / k)),
    }
    # held-out macros (extra was computed before the macro/table generation above)
    macros.update({
        'nTrainN': str(extra['n_train']), 'nTestN': str(extra['n_test']),
        'nTestMSE': '%.5f' % extra['test_mse'], 'nTestPSNR': '%.3f' % extra['test_psnr'],
        'nTestSSIM': '%.3f' % extra['test_ssim'], 'nTestAcc': '%.1f\\%%' % extra['test_acc'],
        'nTestAccCI': '%.1f' % extra['test_acc_ci'], 'nTestSep': '%.3f' % extra['test_sep'],
        'nSsimBase': '%.3f' % extra['ssim_base'],
        # 同一全样本协议下作差，避免混入 held-out SSIM。
        'nSsimGain': '%.3f' % (ssim[k] - extra['ssim_base']),
        'nLatency': '%.2f' % extra['latency_ms'], 'nMseRise': '%.1f\\%%' % extra['mse_rise_pct'],
    })
    lines = ['% auto-generated by build.py from recorded console data and explicit optional experiments']
    lines.extend('\\newcommand{\\%s}{%s}' % item for item in macros.items())
    (PROJ / 'numbers.tex').write_text('\n'.join(lines) + '\n', encoding='utf-8')

    # ---------------- complete table environments ----------------
    rows_k = ['0 (mean face) & %.5f & %.3f & %.3f \\\\' %
              (sweep[0][0], sweep[0][1], extra['ssim_base'])]
    for value in (5, 10, 20, 50, 100):
        cells = [str(value), '%.5f' % sweep[value][0], '%.3f' % sweep[value][1], '%.3f' % ssim[value]]
        if value == k:
            cells = ['\\textbf{%s}' % cell for cell in cells]
        rows_k.append(' & '.join(cells) + r' \\')
    table_k = '\n'.join([
        r'\begin{table}[!t]',
        r'  \caption{Full-sample reconstruction quality. PSNR and SSIM are',
        r'  averaged over images; $k=0$ uses the mean face.}',
        r'  \label{tab:k}', r'  \centering', r'  \footnotesize',
        r'  \setlength{\tabcolsep}{7pt}', r'  \begin{tabular}{@{}lrrr@{}}',
        r'    \toprule', r'    $k$ & MSE & PSNR (dB) & SSIM \\', r'    \midrule',
        *['    ' + row for row in rows_k],
        r'    \bottomrule', r'  \end{tabular}', r'\end{table}',
    ]) + '\n'
    (PROJ / 'tab_k.tex').write_text(table_k, encoding='utf-8')
    table_eff = '\n'.join([
        r'\begin{table}[!t]',
        r'  \caption{Solver times: median and range of three runs.',
        r'  Array storage estimates assume float64 and exclude workspace.}',
        r'  \label{tab:eff}', r'  \centering', r'  \footnotesize',
        r'  \setlength{\tabcolsep}{4pt}', r'  \begin{tabular}{@{}lrrr@{}}',
        r'    \toprule', r'    Solver & Median & Range & Array \\',
        r'           & (s) & (s) & (MB) \\', r'    \midrule',
        '    Covariance + eigh & %.1f & %.1f--%.1f & %d \\\\' %
        (float(eig_m), float(eig_lo), float(eig_hi), round(dim * dim * 8 / 1e6)),
        '    Economy SVD & %s & %s--%s & %.1f \\\\' %
        (svd_m, svd_lo, svd_hi, n * dim * 8 / 1e6),
        r'    \bottomrule', r'  \end{tabular}', r'  \vspace{3pt}',
        r'  \parbox{\columnwidth}{\scriptsize Arrays: covariance matrix $C$',
        r'  ($4096\times4096$) and centered data $\tilde X$ ($400\times4096$),',
        r'  respectively. These are reference array sizes, not measured peak memory.}',
        r'\end{table}',
    ]) + '\n'
    (PROJ / 'tab_eff.tex').write_text(table_eff, encoding='utf-8')
    LOGGER.info('Wrote numeric macros and two complete tables; historical timing records retained')


def read_sources():
    # 收集项目内所有 input/include，保证生成表格也进入 label/ref 校验。
    sources, pending = {}, [PROJ / 'main.tex']
    while pending:
        path = pending.pop().resolve()
        if path in sources:
            continue
        if not path.is_relative_to(PROJ):
            raise BuildError('TeX input is outside the project: %s' % path)
        text = read_text(path)
        sources[path] = text
        for name in re.findall(r'\\(?:input|include)\{([^}]+)\}', text):
            child = path.parent / name
            pending.append(child if child.suffix else child.with_suffix('.tex'))
    return '\n'.join(sources.values())


# ---------------- 2. compile ----------------
def find_tectonic():
    configured = os.environ.get('TECTONIC') or os.environ.get('TECTONIC_PATH')
    if configured:
        path = Path(configured)
        resolved = str(path) if path.is_file() else shutil.which(configured)
        if not resolved:
            raise BuildError('Configured Tectonic executable not found: %s' % configured)
        return resolved
    executable = shutil.which('tectonic')
    if not executable:
        raise BuildError('Tectonic not found; set TECTONIC or add it to PATH')
    return executable


def compile_paper():
    executable = find_tectonic()
    # 原样继承 TECTONIC_CACHE_DIR，避免把可用缓存切换成新建的空目录。
    LOGGER.info('Compiler=%s cache=%s', executable, os.environ.get('TECTONIC_CACHE_DIR', '<default>'))
    result = subprocess.run(
        [executable, '-X', 'compile', 'main.tex', '--keep-logs', '--keep-intermediates'],
        cwd=PROJ, capture_output=True, text=True, encoding='utf-8', errors='replace',
        timeout=900, env=dict(os.environ),
    )
    output = result.stdout + '\n' + result.stderr
    (PROJ / 'build_log.txt').write_text(output, encoding='utf-8')
    LOGGER.info('Compiler exit=%d; output saved to %s', result.returncode, PROJ / 'build_log.txt')
    if result.returncode:
        raise BuildError('Tectonic compilation failed:\n' + output[-3500:])
    # 只用最后一轮日志判定未解析引用，避免把首轮临时警告误认为最终错误。
    log_path = PROJ / 'main.log'
    return read_text(log_path) if log_path.exists() else output


# ---------------- 3. delivery guards ----------------
def scan_log(log):
    patterns = {
        'missing characters': r'Missing character',
        'oversized floats': r'Float too large',
        'undefined references/citations': r'(?:Reference|Citation).*?undefined|There were undefined (?:references|citations)',
        'font fallback': r'Font shape .*?undefined|Some font shapes were not available|defaults substituted|font fallback|font.*?(?:not found|could not be found)',
    }
    problems = []
    for label, pattern in patterns.items():
        matches = list(dict.fromkeys(line for line in log.splitlines() if re.search(pattern, line, re.I)))
        LOGGER.info('Log check %s=%d', label, len(matches))
        problems.extend('%s: %s' % (label, line.strip()) for line in matches)
    overfull = list(dict.fromkeys(line for line in log.splitlines() if re.search(r'Overfull \\[hv]box', line)))
    large = [line for line in overfull if (match := _OVER_RE.search(line)) and float(match.group(1)) > OVER_TOL_PT]
    unknown = [line for line in overfull if not _OVER_RE.search(line)]
    LOGGER.info('Log check overfull h/vboxes=%d above %.1f pt=%d unparsed=%d',
                len(overfull), OVER_TOL_PT, len(large), len(unknown))
    problems.extend('overfull box: ' + line for line in large + unknown)
    if problems:
        raise BuildError('Compilation log guards failed:\n' + '\n'.join(problems[:15]))


def check_references(tex):
    aux = read_text(PROJ / 'main.aux')
    pages = {label: int(page) for label, page in _NEWLABEL_RE.findall(aux)}
    refs_page = pages.get('sec:refs')
    if refs_page is None:
        raise BuildError('Missing sec:refs page anchor in main.aux')
    late = [(label, page) for label, page in pages.items()
            if label.startswith(('fig:', 'tab:')) and page > refs_page]
    LOGGER.info('Float placement check reference_page=%d late_floats=%d', refs_page, len(late))
    if late:
        raise BuildError('Floats occur after the reference section: %s' % late)
    label_list = re.findall(r'\\label\{((?:fig|tab):[^}]+)\}', tex)
    labels, refs = set(label_list), set()
    for group in re.findall(r'\\(?:ref|autoref|[cC]ref)\{([^}]+)\}', tex):
        refs.update(part.strip() for part in group.split(',') if part.strip().startswith(('fig:', 'tab:')))
    duplicates = sorted(label for label in labels if label_list.count(label) > 1)
    if duplicates or labels - refs or refs - labels:
        raise BuildError('Figure/table reference mismatch: duplicates=%s unused=%s missing=%s' %
                         (duplicates, sorted(labels - refs), sorted(refs - labels)))
    LOGGER.info('Figure/table references closed: labels=%d', len(labels))


# ---------------- 4. PyMuPDF reverse checks ----------------
def inspect_pdf():
    pdf_path = PROJ / 'main.pdf'
    try:
        import pymupdf
    except ImportError:
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise BuildError('PDF verification needs either pymupdf or pypdf') from exc
        reader = PdfReader(str(pdf_path))
        texts = [page.extract_text() or '' for page in reader.pages]
        page_count = len(reader.pages)
        LOGGER.info('PDF reader=pypdf pages=%d; geometric checks require optional pymupdf', page_count)
    else:
        with pymupdf.open(pdf_path) as doc:
            page_count = len(doc)
            texts = [page.get_text() for page in doc]
            # 4 raster figures + 2 vector (TikZ) figures in the original paper.
            n_images = sum(len(page.get_images(full=True)) for page in doc)
            LOGGER.info('PDF reader=pymupdf pages=%d image_objects=%d', page_count, n_images)
            bad_blocks = []
            for index, page in enumerate(doc):
                for block in page.get_text('blocks'):
                    x0, _, x1, _ = block[:4]
                    if x0 < 35 or x1 > page.rect.width - 35:
                        bad_blocks.append((index + 1, round(x0, 1), round(x1, 1)))
                # 检查同一页参考文献标题后的题注；跨页浮动体由 aux 检查。
                headings = page.search_for('REFERENCES')
                if headings:
                    reference_y = min(rect.y0 for rect in headings)
                    for block in page.get_text('blocks'):
                        if block[1] > reference_y and re.match(r'\s*(?:Fig\.\s*\d+|TABLE\s+[IVX]+)', block[4]):
                            raise BuildError('A figure/table caption follows references on page %d' % (index + 1))
            if bad_blocks:
                raise BuildError('PDF text extends beyond the 35 pt side guard: %s' % bad_blocks[:8])
            LOGGER.info('PDF geometric checks passed')
    if not page_count:
        raise BuildError('Compiled PDF has no pages')
    text = '\n'.join(texts)
    if '??' in text:
        raise BuildError('Compiled PDF contains unresolved reference markers')
    for token in ('Index Terms', 'REFERENCES'):
        if token.lower() not in text.lower():
            raise BuildError('Compiled PDF lacks required text: %s' % token)
    LOGGER.info('PDF text checks passed; size=%.0f kB', pdf_path.stat().st_size / 1000)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refresh-data', action='store_true',
                        help='Explicitly rerun extra experiments and regenerate macros, tables and figures')
    parser.add_argument('--refresh-figures', action='store_true',
                        help='Regenerate figures from recorded outputs and the cached dataset')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    started = time.perf_counter()
    LOGGER.info('Project=%s refresh_data=%s refresh_figures=%s', PROJ, args.refresh_data, args.refresh_figures)
    try:
        if args.refresh_data:
            with stage('explicit experimental data refresh'):
                refresh_data()
        else:
            LOGGER.info('Reusing existing numbers, tables and recorded experimental results')
        if args.refresh_data or args.refresh_figures:
            with stage('explicit figure refresh'):
                # refresh the efficiency figure from the same console numbers
                LOGGER.info('Figure refresh output: %s', run_python('make_figs.py', 300).strip()[-2000:])
        with stage('source validation'):
            tex = read_sources()
        with stage('Tectonic compilation'):
            log = compile_paper()
        with stage('delivery guards'):
            scan_log(log)
            check_references(tex)
            inspect_pdf()
        # 校验全部通过才算构建成功；PDF 由 Tectonic 直接写在 paper/ 下。
        LOGGER.info('Build finished in %.2f s; PDF=%s', time.perf_counter() - started, OUT_PDF)
    except (BuildError, OSError, ValueError, subprocess.SubprocessError) as exc:
        LOGGER.error('Build failed after %.2f s: %s', time.perf_counter() - started, exc)
        return 3
    return 0


if __name__ == '__main__':
    sys.exit(main())
