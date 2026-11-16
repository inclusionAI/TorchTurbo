#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""基础镜像 CI 的全部逻辑：判断要不要建、构建、校验、发布引用。

为什么全在这里而不是写在 yaml 里：这些判断有过实测依据（见每段注释），
写在 `run: |` 里既不能本地跑、也过不了 shellcheck，而改一行 yaml 就要推一次
才能验证。搬出来之后 `python3 -m scripts.ci.tools.image tag --dry-run` 就能复现。

子命令（对应 workflow 的步骤）：

    env       ② 环境自检：docker / buildx / 架构 / 磁盘
    tag       ③ 内容 tag = `base-<年周>-c<8位哈希>`；同时回答「要不要建」
    build     ⑥ 构建并推送，产出 digest
    verify    ⑦ 回读 manifest，确认是 Docker v2s2 而不是 OCI
    publish   ⑧ 把引用写成 artifact（PR 侧靠它找镜像）
    resolve   被 PR/nightly 调用：override → artifact → 回退
    summary   ⑨ 汇总到 Job Summary
    cleanup   ⑩ 清理 builder 与悬空镜像

tag 为什么是内容派生：同一个 Dockerfile 永远算出同一个 tag，所以「建过没有」
是一个**无状态**查询（问 registry 就行），不需要变量、不需要上一次 run 成功与否。
加年周是为了让上游漂移（cann 基础镜像、pip 里的 torch）能被发现——那类变化
在同一份 Dockerfile 下是算不出新哈希的。不加 run 号，否则「跳过」永远不成立。
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

ARTIFACT = "base-image-ref"
FILENAME = "image-ref.txt"

# SWR 只收 Docker v2s2。Accept 同时给 OCI，否则 OCI 会被 404 掩盖成「不存在」。
ACCEPT = (
    "application/vnd.docker.distribution.manifest.v2+json, "
    "application/vnd.docker.distribution.manifest.list.v2+json, "
    "application/vnd.oci.image.manifest.v1+json, "
    "application/vnd.oci.image.index.v1+json"
)
DOCKER_V2 = "application/vnd.docker.distribution.manifest.v2+json"

ARCH_OF_PLATFORM = {"linux/arm64": "aarch64", "linux/amd64": "x86_64"}


# ---------------------------------------------------------------- 公共
def env(name: str, default: str = "") -> str:
    return os.environ.get(name, default) or default


def note(msg: str) -> None:
    print(f"::notice::{msg}")


def warn(msg: str) -> None:
    print(f"::warning::{msg}", file=sys.stderr)


def fail(msg: str, code: int = 1) -> int:
    print(f"::error::{msg}", file=sys.stderr)
    return code


def out(**kv: object) -> None:
    """写 $GITHUB_OUTPUT；本地跑时静默。"""
    path = env("GITHUB_OUTPUT")
    if not path:
        for k, v in kv.items():
            print(f"  [{k}] = {v}")
        return
    with open(path, "a", encoding="utf-8") as fh:
        for k, v in kv.items():
            fh.write(f"{k}={v}\n")


def summary(markdown: str) -> None:
    path = env("GITHUB_STEP_SUMMARY")
    if path:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(markdown)


def registry_parts() -> tuple[str, str, str]:
    registry = env("REGISTRY")
    repo = env("REPOSITORY")
    return registry, repo, f"{registry}/{repo}"


def _run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def _anon_token(registry: str, repo: str) -> str:
    """SWR 的匿名 pull token。拿不到就返回空串（调用方按「不知道」处理）。"""
    host = registry.split("/")[0]
    url = (
        f"https://{host}/swr/auth/v2/registry/auth/"
        f"?service=dockyard&scope=repository:{repo}:pull"
    )
    try:
        with urllib.request.urlopen(url, timeout=20) as resp:
            return json.loads(resp.read().decode()).get("token", "")
    except (urllib.error.URLError, ValueError, OSError):
        return ""


def _manifest(registry: str, repo: str, tag: str) -> dict | None:
    token = _anon_token(registry, repo)
    if not token:
        return None
    host = registry.split("/")[0]
    req = urllib.request.Request(
        f"https://{host}/v2/{repo}/manifests/{tag}",
        headers={"Authorization": f"Bearer {token}", "Accept": ACCEPT},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode() or "{}")
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            warn(f"回读 manifest 失败：HTTP {exc.code}")
        return None
    except (urllib.error.URLError, ValueError, OSError):
        return None


# ---------------------------------------------------------------- env
def cmd_env() -> int:
    """② 环境自检。失败信息几乎全在这三件事上：有没有 docker、有没有 buildx、
    架构对不对——而它们都会以「构建跑到一半才炸」的形式出现。"""
    platform = env("PLATFORM")
    dockerfile = env("DOCKERFILE")
    print(f"host={os.uname().nodename} arch={os.uname().machine}")
    print(f"trigger={env('GITHUB_EVENT_NAME', 'local')}")

    free = _run(["free", "-h"]).stdout.strip()
    if free:
        print("--- mem ---\n" + free)
    print("--- disk ---")
    print(_run(["df", "-h", "/"]).stdout.strip().splitlines()[-1:])

    if not shutil.which("docker"):
        return fail("本机没有 docker —— 这个池不能用来构建镜像")
    ver = _run(["docker", "version", "--format",
                "client={{.Client.Version}} server={{.Server.Version}}"])
    print(ver.stdout.strip() or ver.stderr.strip()[:200])

    # 没有 buildx 就只能 docker build + push，产出 OCI manifest，SWR 会拒收。
    if _run(["docker", "buildx", "version"]).returncode != 0:
        return fail("docker buildx 不可用 —— 会产出 OCI manifest 被 SWR 拒收")
    print(_run(["docker", "buildx", "version"]).stdout.strip())
    note("docker / buildx 可用")

    want = ARCH_OF_PLATFORM.get(platform)
    have = os.uname().machine
    if want and have != want:
        return fail(f"机器 {have} 与 PLATFORM {platform} 不一致 —— 需要 QEMU，本 runner 没装")

    if not Path(dockerfile).is_file():
        return fail(f"找不到 {dockerfile}")
    print("--- Dockerfile 头 ---")
    print("\n".join(Path(dockerfile).read_text(encoding="utf-8").splitlines()[:3]))
    return 0


# ---------------------------------------------------------------- tag
def _inputs_hash(root: str = "docker/npu") -> str:
    """只哈希 docker/npu/**：镜像的全部输入都在这里，仓库别处改动不该触发 5 h 重建。"""
    base = Path(root)
    h = hashlib.sha256()
    for path in sorted(p for p in base.rglob("*") if p.is_file()):
        h.update(str(path.relative_to(base)).encode())
        h.update(b"\0")
        h.update(hashlib.sha256(path.read_bytes()).digest())
    return h.hexdigest()[:8]


def cmd_tag(dry_run: bool = False) -> int:
    """③ 算内容 tag + 判断要不要建。

    注意不能写 `inputs.should_push || 'true'`：GitHub 表达式里 false 是 falsy，
    会得到字符串 'true'。所以 workflow 把原始值传进 env，这里用 `or` 归一。
    """
    force = env("FORCE", "false") == "true"
    should_push = env("SHOULD_PUSH", "true") == "true"
    registry, repo, full = registry_parts()

    # 本地没有 docker/npu 时按「未知」处理，不阻断 dry-run。
    try:
        inputs_hash = _inputs_hash()
    except OSError:
        inputs_hash = "0" * 8

    import datetime as _dt

    week = _dt.datetime.now(_dt.timezone.utc).strftime("%Yw%V")
    immutable = f"base-{week}-c{inputs_hash}"
    print(f"week={week} inputs_hash={inputs_hash} tag={immutable}")

    out(should_push=str(should_push).lower(), week=week, immutable=immutable,
        full=full, full_tag=f"{full}:{immutable}")

    if dry_run:
        return 0

    # 只验证 Dockerfile 时不看 registry：目的就是把构建跑通。
    if not should_push:
        out(build="true")
        note("should_push=false —— 无条件构建（只验证）")
        return 0
    if force:
        out(build="true")
        note("force_rebuild=true —— 忽略已存在的 tag")
        return 0

    manifest = _manifest(registry, repo, immutable)
    if manifest is None:
        # 查不到就建：宁可多烧一轮，也不要因为「读不到」而永远不更新镜像。
        out(build="true")
        warn(f"仓库匿名不可读，无法确认 {immutable} 是否存在 —— 按需要构建处理")
        return 0

    out(build="false", ref=f"{full}:{immutable}")
    summary(
        f"### 跳过构建\n\n"
        f"`{full}:{immutable}` 已存在（输入指纹 `c{inputs_hash}`，第 {week} 周）。\n\n"
        f"每周强制重建一次，用来发现上游（cann 基础镜像 / pip 包）的漂移。"
        f"需要立刻重建：dispatch 本 workflow 并勾选 **force_rebuild**。\n"
    )
    return 0


# ---------------------------------------------------------------- build
def _max_jobs(spec: str) -> int:
    """auto：按 ~3 GiB/job 估。实测 4 核 / 15 GiB 用 4 可完成（峰值 ~12 GB），
    8 会 OOM。"""
    if spec and spec != "auto":
        return int(spec)
    cores = os.cpu_count() or 2
    mem_gb = 0
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                mem_gb = int(int(line.split()[1]) / 1048576)
                break
    except OSError:
        mem_gb = 0
    return max(1, min(cores, mem_gb // 3 or 1))


def cmd_build() -> int:
    """⑥ 构建并推送。"""
    registry, repo, full = registry_parts()
    immutable = env("TAG_IMM")
    should_push = env("SHOULD_PUSH", "true") == "true"
    use_cache = env("USE_CACHE", "true") == "true"
    builder = env("BUILDER")
    max_jobs = _max_jobs(env("MAX_JOBS", "auto"))
    out(max_jobs=max_jobs)

    if should_push:
        tag_args = ["-t", f"{full}:{immutable}", "-t",
                    f"{full}:{env('MOVING_TAG', 'base-latest')}"]
        # SWR 只收 Docker v2s2：必须 oci-mediatypes=false。
        output = "type=image,push=true,oci-mediatypes=false"
    else:
        tag_args = []            # type=cacheonly 不产出镜像，带 -t 会报错
        output = "type=cacheonly"
        note("should_push=false —— 只验证构建，不产出镜像、不推送")

    # mode=min 只导出最终层（约 6 GiB），落在 GH 单仓 10 GiB 配额内。
    # mode=max 会把全部中间层导出去，超配额后缓存导出失败会连构建一起失败。
    cache_args: list[str] = []
    if builder and (use_cache or not should_push):
        scope = "torchturbo-base"
        cache_args = [f"--cache-from=type=gha,scope={scope}",
                      f"--cache-to=type=gha,mode=min,scope={scope}"]
        print(f"缓存：type=gha,mode=min（scope={scope}）")
    else:
        print("缓存：未启用")

    # digest 是构建的产物不是参数：用 --metadata-file 落盘，比事后
    # imagetools inspect 回查可靠（不依赖 registry 立即可见）。
    meta = "/tmp/build-meta.json"
    cmd = ["docker", "buildx", "build"]
    if builder:
        cmd += ["--builder", builder]
    cmd += ["--progress=plain", f"--platform={env('PLATFORM')}",
            f"--file={env('DOCKERFILE')}", f"--build-arg=MAX_JOBS={max_jobs}",
            "--provenance=false", "--sbom=false", *cache_args]
    if should_push:
        cmd += [f"--metadata-file={meta}"]
    cmd += [*tag_args, f"--output={output}", "."]
    print("--- build ---\n" + " ".join(cmd))

    import time

    start = time.time()
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT)
    log = proc.stdout or ""
    Path("/tmp/build.log").write_text(log, encoding="utf-8")
    print(log)
    elapsed = int(time.time() - start)
    out(elapsed=elapsed, rc=proc.returncode)
    note(f"构建耗时 {elapsed // 60} min（rc={proc.returncode}）")

    if proc.returncode != 0:
        print("::error::构建失败，最后 40 行见上", file=sys.stderr)
        return proc.returncode

    digest = ""
    if should_push and Path(meta).is_file():
        try:
            digest = json.loads(Path(meta).read_text()).get("containerimage.digest", "")
        except (OSError, ValueError):
            digest = ""
    if not digest and should_push:
        insp = _run(["docker", "buildx", "imagetools", "inspect",
                     f"{full}:{immutable}", "--format", "{{.Manifest.Digest}}"])
        digest = insp.stdout.strip()

    out(digest=digest, ref=f"{full}@{digest}" if digest else "")
    note(f"digest = {digest}" if digest else "未推送到 registry，无 digest")
    return 0


# ---------------------------------------------------------------- verify
def cmd_verify() -> int:
    """⑦ 回读 manifest。SWR 拒收 OCI，所以这里必须明说存的是哪一种。"""
    registry, repo, _ = registry_parts()
    immutable = env("TAG_IMM")
    out(media_type="")

    doc = _manifest(registry, repo, immutable)
    if doc is None:
        warn("拿不到匿名 token 或 manifest，跳过格式校验")
        return 0

    fmt = doc.get("mediaType", "")
    layers = sum(int(layer.get("size", 0)) for layer in doc.get("layers", []))
    out(media_type=fmt)
    print(f"mediaType = {fmt or '<空>'}")

    if layers > 0:
        note(f"镜像压缩后 {layers / 1073741824:.2f} GiB（{layers} bytes）")

    if fmt == DOCKER_V2:
        note("存的是 Docker v2s2 manifest ✓")
        return 0
    if "oci.image" in fmt:
        return fail("存的是 OCI manifest —— SWR 下游可能拒收。检查是否漏了 "
                    "oci-mediatypes=false")
    warn(f"未识别的 manifest 格式：{fmt or '<空>'}")
    return 0


# ---------------------------------------------------------------- publish / resolve
def publish(path: str, ref: str, tag: str, digest: str) -> int:
    """写一行 JSON。解析侧只读 `ref`，其余字段给人排查用。"""
    payload = {"ref": ref, "tag": tag, "digest": digest}
    Path(path).write_text(json.dumps(payload, ensure_ascii=False) + "\n",
                          encoding="utf-8")
    print(f"published: {ref}")
    return 0


def cmd_publish(out_path: str = "/tmp/image-ref/image-ref.txt") -> int:
    """⑧ 发布当前该用哪个镜像。

    走 artifact 而不是 repository variable：`gh variable set` 需要 Variables API，
    而 GITHUB_TOKEN 对它一律 403（permissions 加 write 也没用）。
    跳过构建时也要发布——那份信息同样有效（就是刚判断出来的 tag）。
    """
    registry, _, full = registry_parts()
    immutable = env("TAG_IMM")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    return publish(out_path, f"{full}:{immutable}", immutable, env("DIGEST"))


def _gh(args: list[str]) -> str:
    proc = subprocess.run(["gh", *args], capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or f"gh {' '.join(args)} 失败")
    return proc.stdout


def _from_artifact(repo: str) -> str | None:
    """取最近一次未过期 artifact 里的 ref。任何一步失败都只是「没拿到」。"""
    listing = _gh([
        "api", f"repos/{repo}/actions/artifacts?name={ARTIFACT}&per_page=20",
        "--jq", "[.artifacts[] | {url: .archive_download_url, expired: .expired, "
                "at: .created_at}] | sort_by(.at) | reverse",
    ])
    for item in json.loads(listing or "[]"):
        if item.get("expired"):
            continue
        # artifact 是 zip；为了读几百字节先落盘再 unzip 不值得。
        blob = subprocess.run(["gh", "api", item["url"]], capture_output=True)
        if blob.returncode != 0 or not blob.stdout:
            continue
        try:
            with zipfile.ZipFile(io.BytesIO(blob.stdout)) as zf:
                names = [n for n in zf.namelist() if n.endswith(FILENAME)]
                if not names:
                    continue
                data = zf.read(names[0]).decode("utf-8", "replace").strip()
        except (zipfile.BadZipFile, OSError):
            continue
        if not data:
            continue
        try:
            ref = json.loads(data).get("ref", "")
        except json.JSONDecodeError:
            ref = data.splitlines()[0]      # 兼容最早的纯文本格式
        if ref:
            print(f"resolved from artifact ({item['at']}): {ref}", file=sys.stderr)
            return ref
    return None


def cmd_resolve(override: str, fallback: str, repo: str) -> int:
    if override:
        # 人工钉死的优先：排查「到底是不是镜像的问题」时需要一个能立刻切换的地方。
        print(f"using override: {override}", file=sys.stderr)
        print(override)
        return 0

    if repo:
        try:
            ref = _from_artifact(repo)
        except Exception as exc:                     # noqa: BLE001
            warn(f"读 image-ref artifact 失败：{type(exc).__name__}: {exc}")
            ref = None
        if ref:
            print(ref)
            return 0
        # 只有「想去读但没读到」才需要警告：kubelet 的 IfNotPresent 只比 tag
        # 字符串，移动 tag 的内容变了它不会重拉。
        warn(f"没找到镜像引用 artifact，回退到 {fallback}"
             "（kubelet 不会因 tag 内容变化而重拉）")

    # 不带 --repo = 有意使用固定链接（不是查询失败），因此不告警。
    print(fallback)
    return 0


# ---------------------------------------------------------------- summary / cleanup
def cmd_summary() -> int:
    """⑨ 汇总。tag 与 digest 的区别值得每次都写一遍：tag 可以被重新指向别的镜像。"""
    registry, _, full = registry_parts()
    immutable = env("TAG_IMM")
    summary(
        "## Image CI (base)\n\n"
        "| 项 | 值 |\n|---|---|\n"
        f"| 触发 | `{env('GITHUB_EVENT_NAME', 'local')}` |\n"
        f"| runner | `{env('RUNNER_LABEL')}` |\n"
        f"| Dockerfile | `{env('DOCKERFILE')}` |\n"
        f"| 本次构建 | `{env('BUILT')}` |\n"
        f"| 使用缓存 | `{env('USE_CACHE')}` |\n"
        f"| 推送 | `{env('SHOULD_PUSH')}` |\n"
        f"| 不可变 tag | `{full}:{immutable}` |\n"
        f"| 滚动 tag | `{full}:{env('MOVING_TAG', 'base-latest')}` |\n"
        f"| digest | `{env('DIGEST', 'n/a')}` |\n"
        f"| manifest 格式 | `{env('MEDIA_TYPE', 'n/a')}` |\n"
        f"| 构建耗时 | `{env('ELAPSED', 'n/a')}` s |\n\n"
        "### 消费方式\n\n"
        "PR / nightly 不写死地址，由 `scripts/ci/tools/image.py resolve` 从最新一份\n"
        "`base-image-ref` artifact 里读；人工覆盖走仓库变量 `BASE_IMAGE_TAG`。\n\n"
        "> tag 是名字、digest 是内容。需要「永远不变」的引用就写 `repo@sha256:...`。\n"
    )
    return 0


def cmd_cleanup() -> int:
    """⑩ 清理。大镜像很占磁盘，自托管 runner 上尤其要注意。"""
    builder = env("BUILDER")
    if builder:
        _run(["docker", "buildx", "rm", builder])
        print(f"已删除 builder {builder}")
    _run(["docker", "image", "prune", "-f"])
    print(_run(["df", "-h", "/"]).stdout.strip().splitlines()[-1:])
    return 0


# ---------------------------------------------------------------- CLI
def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="基础镜像 CI")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("env")

    p_tag = sub.add_parser("tag")
    p_tag.add_argument("--dry-run", action="store_true", help="只算 tag，不问 registry")

    sub.add_parser("build")
    sub.add_parser("verify")
    sub.add_parser("summary")
    sub.add_parser("cleanup")

    p_pub = sub.add_parser("publish")
    p_pub.add_argument("--out", default="/tmp/image-ref/image-ref.txt")
    p_pub.add_argument("--ref", default="")
    p_pub.add_argument("--tag", default="")
    p_pub.add_argument("--digest", default="")

    p_res = sub.add_parser("resolve")
    p_res.add_argument("--override", default="", help="人工钉死的引用（vars.BASE_IMAGE_TAG）")
    p_res.add_argument("--fallback", default="", help="都拿不到时用什么")
    p_res.add_argument("--repo", default="", help="owner/name；空则跳过 artifact 查询")

    args = ap.parse_args(argv)

    if args.cmd == "env":
        return cmd_env()
    if args.cmd == "tag":
        return cmd_tag(args.dry_run)
    if args.cmd == "build":
        return cmd_build()
    if args.cmd == "verify":
        return cmd_verify()
    if args.cmd == "summary":
        return cmd_summary()
    if args.cmd == "cleanup":
        return cmd_cleanup()
    if args.cmd == "publish":
        if args.ref:                      # 显式给了就照给
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            return publish(args.out, args.ref, args.tag, args.digest)
        return cmd_publish(args.out)
    return cmd_resolve(args.override, args.fallback, args.repo)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
