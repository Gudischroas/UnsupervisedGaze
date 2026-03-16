"""
提取 eg 视线估计实验结果
目标实验:
  - eg-sga-pair-lr-v222-d111-eve-1CBAM-default
    - eg-sga-pair-lr-v222-d111-eve-1CBAM-MSFF-default
    - eg-sga-pair-lr-v222-d111-eve-2CBAM-MSFF-default
    - eg-sga-pair-lr-v222-d111-eve-3CBAM-MSFF-default
    - eg-sga-pair-lr-v222-d111-eve-MSFF-default
  - eg-sga-pair-lr-v222-d111-eve-nocbam-default
  - eg-sga-pair-lr-v222-d111-eve-1CBAM-50epoch-default

子目录命名规则: {exp_name}-{run_id}-{fold_id}
  run_id  : 0001/0002/... 对应预训练的不同随机种子运行
  fold_id : 0-7 对应 8 折评估
"""

import os
import re
import sys
from collections import defaultdict

import numpy as np

# ──────────────────────────────────────────────
# 配置
# ──────────────────────────────────────────────
BASE_DIR = os.path.join(os.path.dirname(__file__), "outputs", "checkpoints")

EXPERIMENTS = {
    "1CBAM":   "eg-sga-pair-lr-v222-d111-eve-1CBAM-default",
    "1CBAM_MSFF": "eg-sga-pair-lr-v222-d111-eve-1CBAM-MSFF-default",
    "2CBAM_MSFF": "eg-sga-pair-lr-v222-d111-eve-2CBAM-MSFF-default",
    "3CBAM_MSFF": "eg-sga-pair-lr-v222-d111-eve-3CBAM-MSFF-default",
    "MSFF": "eg-sga-pair-lr-v222-d111-eve-MSFF-default",
    "2CBAM":   "eg-sga-pair-lr-v222-d111-eve-2CBAM-default",
    "3CBAM":   "eg-sga-pair-lr-v222-d111-eve-3CBAM-default",
    "nocbam":  "eg-sga-pair-lr-v222-d111-eve-nocbam-default",
    "1CBAM-50epoch": "eg-sga-pair-lr-v222-d111-eve-1CBAM-50epoch-default",
}

# 匹配最终 Eval 行:
#   ... Eval at Step 2001: final_test: 8.04, min_test: 8.027, min_val: 8.84
EVAL_PATTERN = re.compile(
    r"Eval at Step\s+\d+:\s+"
    r"final_test:\s*(?P<final_test>[\d.]+),\s*"
    r"min_test:\s*(?P<min_test>[\d.]+),\s*"
    r"min_val:\s*(?P<min_val>[\d.]+)"
)

# 子目录名后缀: {run_id}-{fold_id}，例如 0002-3
SUBDIR_SUFFIX_PATTERN = re.compile(r"-(\d{4})-(\d+)$")


# ──────────────────────────────────────────────
# 解析单个 messages.log，取最后一条 Eval 行
# ──────────────────────────────────────────────
def parse_log(log_path):
    """返回 (final_test, min_test, min_val) 或 None（日志不完整时）。"""
    last_match = None
    try:
        with open(log_path, "r", errors="replace") as f:
            for line in f:
                m = EVAL_PATTERN.search(line)
                if m:
                    last_match = m
    except FileNotFoundError:
        return None

    if last_match is None:
        return None

    return (
        float(last_match.group("final_test")),
        float(last_match.group("min_test")),
        float(last_match.group("min_val")),
    )


# ──────────────────────────────────────────────
# 收集并统计一个实验的所有结果
# ──────────────────────────────────────────────
def collect_experiment(exp_label, exp_dir_name):
    exp_dir = os.path.join(BASE_DIR, exp_dir_name)
    if not os.path.isdir(exp_dir):
        print(f"[警告] 目录不存在: {exp_dir}", file=sys.stderr)
        return None

    # run_id -> list of (final_test, min_test, min_val) across folds
    run_data = defaultdict(list)
    missing = []

    for entry in sorted(os.listdir(exp_dir)):
        sub_path = os.path.join(exp_dir, entry)
        if not os.path.isdir(sub_path):
            continue

        m = SUBDIR_SUFFIX_PATTERN.search(entry)
        if not m:
            continue

        run_id, fold_id = m.group(1), m.group(2)
        log_path = os.path.join(sub_path, "messages.log")
        result = parse_log(log_path)

        if result is None:
            missing.append(entry)
        else:
            run_data[run_id].append(result)

    if missing:
        print(f"[警告] {exp_label}: {len(missing)} 个子目录日志不完整或无 Eval 行:")
        for m in missing:
            print(f"         {m}")

    return run_data


# ──────────────────────────────────────────────
# 格式化输出
# ──────────────────────────────────────────────
def report(exp_label, run_data):
    if not run_data:
        print(f"\n{'='*60}")
        print(f"实验: {exp_label}  — 无有效数据")
        return

    print(f"\n{'='*60}")
    print(f"实验: {exp_label}")
    print(f"{'='*60}")

    # 每个 run，先对 folds 取均值
    run_means_ft, run_means_mt, run_means_mv = [], [], []

    print(f"  {'运行ID':<10} {'折数':>4}  {'final_test(°)':>14}  {'min_test(°)':>12}  {'min_val(°)':>11}")
    print(f"  {'-'*8:<10} {'----':>4}  {'':>14}  {'':>12}  {'':>11}")

    for run_id in sorted(run_data.keys()):
        folds = run_data[run_id]
        ft_vals = [r[0] for r in folds]
        mt_vals = [r[1] for r in folds]
        mv_vals = [r[2] for r in folds]

        ft_mean = np.mean(ft_vals)
        mt_mean = np.mean(mt_vals)
        mv_mean = np.mean(mv_vals)

        run_means_ft.append(ft_mean)
        run_means_mt.append(mt_mean)
        run_means_mv.append(mv_mean)

        print(f"  {run_id:<10} {len(folds):>4}  {ft_mean:>14.4f}  {mt_mean:>12.4f}  {mv_mean:>11.4f}")

    print()

    # 跨 run 统计
    n = len(run_means_ft)
    ft_arr = np.array(run_means_ft)
    mt_arr = np.array(run_means_mt)
    mv_arr = np.array(run_means_mv)

    print(f"  ── 跨 {n} 次运行汇总统计 ──")
    print(f"  {'指标':<20} {'均值(°)':>10}  {'标准差(°)':>10}  {'最小(°)':>8}  {'最大(°)':>8}")
    print(f"  {'-'*18:<20} {'------':>10}  {'------':>10}  {'------':>8}  {'------':>8}")

    for label, arr in [("final_test (主结果)", ft_arr),
                       ("min_val   (选择依据)", mv_arr),
                       ("min_test  (参考上限)", mt_arr)]:
        print(f"  {label:<20} {np.mean(arr):>10.4f}  {np.std(arr, ddof=1) if n > 1 else 0.0:>10.4f}"
              f"  {np.min(arr):>8.4f}  {np.max(arr):>8.4f}")


# ──────────────────────────────────────────────
# 汇总对比表
# ──────────────────────────────────────────────
def summary_table(all_stats):
    print(f"\n{'='*60}")
    print("实验对比汇总（均值 ± 标准差，单位：°）")
    print(f"{'='*60}")
    print(f"  {'实验':<12}  {'final_test':>18}  {'min_val':>16}  {'min_test':>16}")
    print(f"  {'-'*10:<12}  {'':>18}  {'':>16}  {'':>16}")
    for label, (ft_mean, ft_std, mv_mean, mv_std, mt_mean, mt_std) in all_stats.items():
        ft_str = f"{ft_mean:.4f} ± {ft_std:.4f}"
        mv_str = f"{mv_mean:.4f} ± {mv_std:.4f}"
        mt_str = f"{mt_mean:.4f} ± {mt_std:.4f}"
        print(f"  {label:<12}  {ft_str:>18}  {mv_str:>16}  {mt_str:>16}")


# ──────────────────────────────────────────────
# 主程序
# ──────────────────────────────────────────────
def main():
    all_stats = {}

    for exp_label, exp_dir_name in EXPERIMENTS.items():
        run_data = collect_experiment(exp_label, exp_dir_name)
        report(exp_label, run_data)

        if run_data:
            run_means_ft, run_means_mt, run_means_mv = [], [], []
            for run_id in sorted(run_data.keys()):
                folds = run_data[run_id]
                run_means_ft.append(np.mean([r[0] for r in folds]))
                run_means_mt.append(np.mean([r[1] for r in folds]))
                run_means_mv.append(np.mean([r[2] for r in folds]))

            n = len(run_means_ft)
            ddof = 1 if n > 1 else 0
            all_stats[exp_label] = (
                np.mean(run_means_ft), np.std(run_means_ft, ddof=ddof),
                np.mean(run_means_mv), np.std(run_means_mv, ddof=ddof),
                np.mean(run_means_mt), np.std(run_means_mt, ddof=ddof),
            )

    summary_table(all_stats)


if __name__ == "__main__":
    main()
