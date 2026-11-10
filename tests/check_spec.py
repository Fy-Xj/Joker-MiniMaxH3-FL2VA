#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Joker 队 · 嵩山杯 2026 赛题一 —— 输出规格自动核查（测试用例）

用法：
    python tests/check_spec.py <视频文件> [--expect 768x928,124,24.0,5.17]

检查项（与官方评分脚本 minimax_h3_compare.py 的 Size / Length 判定对齐）：
    1. 视频流存在，编码为 H.264
    2. 分辨率 = 768×928（官方首帧 800×965 推导值）
    3. 逐帧解码计数 = 124
    4. 帧率 = 24.0
    5. 时长 ≈ 5.17 s
    6. 音频流存在（AAC 32 kHz 立体声）
    7. 像素格式 yuv420p（与官方一致）

退出码：0 全部通过；1 有任一项失败
"""
import argparse
import os
import sys

try:
    import av
except ImportError:
    print("[FATAL] 需要 av 库（pip install av 或从厂商源安装）")
    sys.exit(2)

EXPECT_WIDTH = 768
EXPECT_HEIGHT = 928
EXPECT_FRAMES = 124
EXPECT_FPS = 24.0
EXPECT_DURATION = 5.17
DURATION_TOL = 0.15


def probe(path):
    """返回视频/音频流的元信息与逐帧计数。"""
    info = {
        "path": path,
        "size_bytes": os.path.getsize(path),
        "video": None,
        "audio": None,
        "frames": 0,
        "duration": None,
    }
    container = av.open(path)
    try:
        if container.duration:
            info["duration"] = float(container.duration) / 1e6
        for stream in container.streams:
            cc = stream.codec_context
            if stream.type == "video":
                info["video"] = {
                    "codec": cc.name,
                    "width": cc.width,
                    "height": cc.height,
                    "fps": float(stream.average_rate) if stream.average_rate else None,
                    "pix_fmt": cc.pix_fmt,
                }
            elif stream.type == "audio":
                info["audio"] = {
                    "codec": cc.name,
                    "sample_rate": cc.sample_rate,
                    "channels": cc.channels,
                }
        # 逐帧解码，验证无丢帧
        vstreams = [s for s in container.streams if s.type == "video"]
        if vstreams:
            info["frames"] = sum(1 for _ in container.decode(vstreams[0]))
    finally:
        container.close()
    return info


def check(info, verbose=True):
    """返回 (通过项数, 失败项列表)"""
    results = []

    def rec(name, ok, detail):
        results.append((name, ok, detail))

    v = info["video"]
    a = info["audio"]

    rec("视频流存在", v is not None, v is not None)
    if v:
        rec("编码 H.264", v["codec"] in ("h264", "avc1"),
            f"codec={v['codec']}")
        rec("分辨率", (v["width"], v["height"]) == (EXPECT_WIDTH, EXPECT_HEIGHT),
            f"{v['width']}x{v['height']} (期望 {EXPECT_WIDTH}x{EXPECT_HEIGHT})")
        rec("像素格式 yuv420p", v["pix_fmt"] == "yuv420p", f"pix_fmt={v['pix_fmt']}")
        fps_ok = v["fps"] is not None and abs(v["fps"] - EXPECT_FPS) < 1e-3
        rec("帧率 24 FPS", fps_ok, f"fps={v['fps']}")

    rec("帧数 124", info["frames"] == EXPECT_FRAMES,
        f"decoded={info['frames']} (期望 {EXPECT_FRAMES})")

    if info["duration"] is not None:
        dur_ok = abs(info["duration"] - EXPECT_DURATION) <= DURATION_TOL
        rec("时长 ≈5.17s", dur_ok,
            f"{info['duration']:.4f}s (期望 {EXPECT_DURATION}±{DURATION_TOL})")

    rec("音频流存在", a is not None, "官方要求: 音频存在")
    if a:
        rec("音频 AAC 32kHz 立体声",
            a["codec"] == "aac" and a["sample_rate"] == 32000 and a["channels"] == 2,
            f"codec={a['codec']} {a['sample_rate']}Hz {a['channels']}ch")

    if verbose:
        print("=" * 62)
        print(f"规格核查: {os.path.basename(info['path'])}  ({info['size_bytes']:,} bytes)")
        print("=" * 62)
        for name, ok, detail in results:
            mark = "PASS" if ok else "FAIL"
            print(f"  [{mark}] {name:24s} {detail}")
        print("=" * 62)

    failed = [(n, d) for n, ok, d in results if not ok]
    return len(results) - len(failed), failed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="待核查的 mp4 文件")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(args.path):
        print(f"[FATAL] 文件不存在: {args.path}")
        return 2

    info = probe(args.path)
    passed, failed = check(info, verbose=not args.quiet)

    if failed:
        print(f"\n结果: {passed} 项通过, {len(failed)} 项失败")
        for n, d in failed:
            print(f"  失败: {n} -> {d}")
        return 1

    print(f"\n结果: 全部 {passed} 项通过 —— 规格与官方口径一致")
    return 0


if __name__ == "__main__":
    sys.exit(main())
