#!/usr/bin/env bash
# 诊断公司代理后面 conda / pip / curl 的代理与 TLS 证书问题。
# 只读：不改任何配置，不关闭证书校验。代理口令、.netrc 口令会脱敏。
#
# 用法：
#   bash scripts/diagnose_conda_tls.sh            # 快速检查，1-2 分钟
#   bash scripts/diagnose_conda_tls.sh --full     # 额外跑一次 conda create --dry-run（真实下载路径）
#   bash scripts/diagnose_conda_tls.sh --conda /home/ma-user/anaconda3/bin/conda --host mirror.example.com
#
# 输出目录 ./conda_tls_diag_<时间戳>/：report.txt 是完整报告（贴回来即可），
# chain_<host>.pem 是代理实际出示的证书链，candidate_ca_<host>.pem 是链顶自签根证书（若有）。

CONDA_BIN=""; FULL=0; EXTRA_HOSTS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --conda) CONDA_BIN=$2; shift 2 ;;
    --full) FULL=1; shift ;;
    --host) EXTRA_HOSTS+=("$2"); shift 2 ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "未知参数: $1" >&2; exit 2 ;;
  esac
done

OUT=${OUT:-$PWD/conda_tls_diag_$(date +%Y%m%d_%H%M%S)}
mkdir -p "$OUT"
exec > >(tee "$OUT/report.txt") 2>&1

HINTS=()
hint() { HINTS+=("$*"); }
section() { printf '\n==== %s ====\n' "$*"; }
redact() { sed -E 's#(://)[^/@[:space:]]+@#\1***@#g'; }
have() { command -v "$1" >/dev/null 2>&1; }
TO() { local t=$1; shift; if have timeout; then timeout "$t" "$@"; else "$@"; fi; }
realp() { readlink -f "$1" 2>/dev/null || echo "$1"; }

# ---------------------------------------------------------------- 1. 工具
section "1. 工具与版本"
echo "time=$(date '+%F %T %z')  host=$(hostname)  user=$(id -un)"
[ -r /etc/os-release ] && grep -E '^PRETTY_NAME=' /etc/os-release

if [ -z "$CONDA_BIN" ]; then
  for c in "${CONDA_EXE:-}" "$(type -P conda 2>/dev/null)" /home/ma-user/anaconda3/bin/conda \
           "$HOME/anaconda3/bin/conda" "$HOME/miniconda3/bin/conda" /opt/conda/bin/conda; do
    if [ -n "$c" ] && [ -x "$c" ]; then CONDA_BIN=$c; break; fi
  done
fi
CONDA_ROOT=""; CONDA_PY=""; CERTIFI=""
if [ -n "$CONDA_BIN" ]; then
  CONDA_ROOT=$("$CONDA_BIN" info --base 2>/dev/null)
  CONDA_PY=$CONDA_ROOT/bin/python
  echo "conda: $CONDA_BIN ($("$CONDA_BIN" --version 2>&1))  base=$CONDA_ROOT"
  "$CONDA_PY" -c 'import sys, ssl; print("base python", sys.version.split()[0], "|", ssl.OPENSSL_VERSION)'
  "$CONDA_PY" -c 'import requests, urllib3; print("requests", requests.__version__, "| urllib3", urllib3.__version__)' 2>&1
  CERTIFI=$("$CONDA_PY" -c 'import certifi; print(certifi.where())' 2>/dev/null)
  echo "certifi: ${CERTIFI:-<无>}"
  "$CONDA_BIN" list -n base 2>/dev/null | grep -E '^(conda-libmamba-solver|libmamba|truststore|ca-certificates|certifi) ' | sed 's/^/  /'
else
  echo "conda: <未找到>，可用 --conda 指定"
fi
declare -A SEEN_CURL=()
CURLS=()
for c in $(type -aP curl); do
  r=$(realp "$c"); [ -n "${SEEN_CURL[$r]:-}" ] && continue
  SEEN_CURL[$r]=1; CURLS+=("$c")
  echo "curl: $c  ($("$c" -V 2>/dev/null | head -1 | cut -d' ' -f1-2))"
done
case "${CURLS[0]:-}" in
  *conda*|*forge*|"${CONDA_ROOT:-/nonexistent}"/*)
    hint "PATH 里第一个 curl 是 ${CURLS[0]}（conda 环境里的 curl），它读 conda 自带的 CA 包而不是系统证书库，和 /usr/bin/curl 的结果可能不同。" ;;
esac
echo "openssl: $(openssl version 2>/dev/null || echo '<未找到>')"

# ---------------------------------------------------------------- 2. 代理
section "2. 代理环境变量"
env | grep -iE '^(https?|all|no|ftp)_proxy=' | redact | sort
PROXY_URL=${https_proxy:-${HTTPS_PROXY:-${all_proxy:-${ALL_PROXY:-}}}}
PROXY_SCHEME=""; PROXY_HOSTPORT=""; PUSER=""; DIAG_PROXY_PASS=""
if [ -z "$PROXY_URL" ]; then
  hint "没有设置 https_proxy/HTTPS_PROXY。若代理只配在 .condarc 的 proxy_servers 里，curl/pip 不会用它。"
else
  PROXY_SCHEME=${PROXY_URL%%://*}; [ "$PROXY_SCHEME" = "$PROXY_URL" ] && PROXY_SCHEME=http
  p=${PROXY_URL#*://}
  if [[ $p == *@* ]]; then ui=${p%%@*}; PUSER=${ui%%:*}; DIAG_PROXY_PASS=${ui#*:}; p=${p##*@}; fi
  PROXY_HOSTPORT=${p%%/*}
  echo "解析：scheme=$PROXY_SCHEME  host:port=$PROXY_HOSTPORT  带账号=$([ -n "$PUSER" ] && echo 是 || echo 否)"
  if [ -n "${https_proxy:-}" ] && [ -n "${HTTPS_PROXY:-}" ] && [ "$https_proxy" != "$HTTPS_PROXY" ]; then
    hint "https_proxy 与 HTTPS_PROXY 不一致；requests/curl 优先用小写 https_proxy。"
  fi
  [ "$PROXY_SCHEME" = https ] && hint "代理地址写成了 https://。普通公司代理对客户端说明文 HTTP（CONNECT），应写 http://host:port；scheme 写错时 conda 报 ProxyError（已复现）。"
fi
export DIAG_PROXY_PASS

bypass_proxy() {  # $1=host；按 no_proxy 后缀匹配
  local e np=${no_proxy:-${NO_PROXY:-}}
  IFS=',' read -ra _np <<< "$np"
  for e in "${_np[@]}"; do
    e=${e// /}; e=${e#\*}; e=${e#.}; [ -z "$e" ] && continue
    [ "$e" = "*" ] && return 0
    [[ $1 == "$e" || $1 == *".$e" ]] && return 0
  done
  return 1
}

# ---------------------------------------------------------------- 3. CA 变量
section "3. 证书相关环境变量"
for v in REQUESTS_CA_BUNDLE CURL_CA_BUNDLE SSL_CERT_FILE SSL_CERT_DIR PIP_CERT PIP_INDEX_URL PIP_TRUSTED_HOST CONDARC; do
  [ -z "${!v+x}" ] && continue
  val=${!v}; extra=""
  if [[ $v == *CA_BUNDLE || $v == SSL_CERT_FILE || $v == PIP_CERT ]]; then
    if [ -f "$val" ]; then extra="  [存在，$(grep -c 'BEGIN CERTIFICATE' "$val") 张证书]"; else extra="  [文件不存在!]"; fi
  fi
  echo "$v=$(printf %s "$val" | redact)$extra"
done
env | grep -E '^CONDA_' | grep -viE 'token|pass|secret|key' \
    | grep -vE '^CONDA_(PREFIX(_[0-9]+)?|DEFAULT_ENV|SHLVL|PROMPT_MODIFIER|EXE|PYTHON_EXE)=' | redact
[ -n "${CONDA_SSL_VERIFY:-}" ] && hint "设置了 CONDA_SSL_VERIFY=$CONDA_SSL_VERIFY：环境变量优先于 ~/.condarc，conda config --set ssl_verify 写进文件的值不生效（已复现）。"

# ---------------------------------------------------------------- 4. conda 配置
PROBE=$OUT/probe_conda.py
cat > "$PROBE" <<'PY'
import re, sys, warnings
warnings.filterwarnings("ignore")
red = lambda x: re.sub(r"(://)[^/@\s]+@", r"\1***@", str(x))
from conda.base.context import context, reset_context
reset_context()
if sys.argv[1] == "urls":
    from conda.models.channel import Channel
    seen = []
    for c in context.channels:
        try:
            for u in Channel(c).urls(False, ("noarch",)):
                if u.startswith("https://") and u not in seen:
                    seen.append(u)
        except Exception as e:
            print("# channel %s: %s" % (red(c), e), file=sys.stderr)
    print("\n".join(u + "/repodata.json" for u in seen))
    sys.exit()
if sys.argv[1] == "ssl_verify":
    print(context.ssl_verify)
    sys.exit()
from conda.gateways.connection import session as cs
url = sys.argv[2]
sess = cs.get_session(url) if hasattr(cs, "get_session") else cs.CondaSession()
env = sess.merge_environment_settings(url, {}, None, None, None)
print("  .condarc ssl_verify :", repr(context.ssl_verify))
print("  实际生效 verify      :", repr(env["verify"]))
print("  实际使用代理        :", red(env["proxies"].get("https") or env["proxies"].get("all") or "<无>"))
if env["verify"] != sess.verify:
    if sess.verify is True:
        print("  （ssl_verify=true，requests 改用 REQUESTS_CA_BUNDLE/CURL_CA_BUNDLE 指向的文件）")
    else:
        print("  OVERRIDDEN: .condarc 的 ssl_verify 被 REQUESTS_CA_BUNDLE/CURL_CA_BUNDLE 覆盖")
def chain(e):
    out, seen = [], set()
    while e is not None and id(e) not in seen and len(out) < 6:
        seen.add(id(e))
        out.append("%s: %s" % (type(e).__name__, red(e)[:260]))
        nxt = getattr(e, "reason", None)
        if not isinstance(nxt, BaseException):
            nxt = next((a for a in getattr(e, "args", ()) if isinstance(a, BaseException)), None)
        e = nxt or e.__cause__ or e.__context__
    return out
try:
    r = sess.get(url, stream=True, timeout=(15, 30))
    r.close()
    print("  GET -> HTTP %s" % r.status_code)
    print("RESULT=ok")
except Exception as e:
    c = chain(e)
    for line in c:
        print("  GET 失败: " + line)
    names = " ".join(c)
    print("RESULT=" + ("proxy" if "ProxyError" in names else "ssl" if "SSL" in names or "certificate" in names else "other"))
PY

CONDA_URLS=()
if [ -n "$CONDA_BIN" ]; then
  section "4. conda 配置"
  "$CONDA_BIN" config --show-sources 2>&1 | redact
  for k in ssl_verify proxy_servers channels default_channels channel_alias custom_channels solver; do
    "$CONDA_BIN" config --show "$k" 2>/dev/null | redact
  done
  SSLV=$("$CONDA_PY" "$PROBE" ssl_verify 2>/dev/null)
  echo "conda 解析后的 ssl_verify: $SSLV"
  [ "$SSLV" = False ] && hint "conda 当前 ssl_verify=false。它只影响 conda 自己，不影响 curl、pip（含 environment.yml 的 pip: 段）；修好证书后要改回。"
  for v in REQUESTS_CA_BUNDLE CURL_CA_BUNDLE; do  # requests 只取第一个非空的
    [ -n "${!v:-}" ] || continue
    if [ "$SSLV" = True ]; then
      hint "conda 实际用 $v=${!v} 验证证书（不是自带 certifi）；它必须包含公司 CA。"
    elif [ "$(realp "${!v}")" != "$(realp "$SSLV")" ]; then
      hint "设置了 $v=${!v}：conda 底层 requests 会用它覆盖 .condarc 的 ssl_verify（当前 $SSLV），ssl_verify:false 和 ssl_verify:<CA 路径> 都会被静默忽略（已复现）。这个文件必须包含公司 CA，或者先 unset。"
    fi
    break
  done
  mapfile -t CONDA_URLS < <("$CONDA_PY" "$PROBE" urls 2>/dev/null | head -4)
fi

# ---------------------------------------------------------------- 5. pip / netrc / curlrc
section "5. pip 配置 / .netrc / .curlrc"
PIP_PY=${CONDA_PY:-$(type -P python3)}
PIP_INDEX=""
if [ -n "$PIP_PY" ]; then
  "$PIP_PY" -m pip --version 2>&1
  { "$PIP_PY" -m pip config debug 2>/dev/null || "$PIP_PY" -m pip config list 2>&1; } | redact
  PIP_INDEX=${PIP_INDEX_URL:-$("$PIP_PY" -m pip config list 2>/dev/null | sed -nE "s/^[a-z]+\.index-url='(.*)'$/\1/p" | head -1)}
fi
PIP_INDEX=${PIP_INDEX:-https://pypi.org/simple}
echo "pip index: $(printf %s "$PIP_INDEX" | redact)"
if [ -f "$HOME/.netrc" ]; then
  echo "~/.netrc 存在，条目（口令已隐去）："; grep -oE 'machine[[:space:]]+[^[:space:]]+|default' "$HOME/.netrc" | sed 's/^/  /'
else
  echo "~/.netrc 不存在（conda ProxyError 提示里的 .netrc 是固定文案，不代表它是原因）"
fi
if [ -f "$HOME/.curlrc" ]; then
  echo "~/.curlrc 相关行："; grep -iE 'insecure|^-k|cacert|capath|proxy' "$HOME/.curlrc" | redact | sed 's/^/  /'
fi

# ---------------------------------------------------------------- 6. 逐 host 探测
BUNDLES=(); declare -A SEEN_B=()
SSLV_PATH=""; [ -n "${SSLV:-}" ] && [ -f "$SSLV" ] && SSLV_PATH=$SSLV
for b in /etc/ssl/certs/ca-certificates.crt /etc/pki/tls/certs/ca-bundle.crt /etc/ssl/ca-bundle.pem \
         "${CONDA_ROOT:+$CONDA_ROOT/ssl/cacert.pem}" "$CERTIFI" "${REQUESTS_CA_BUNDLE:-}" \
         "${CURL_CA_BUNDLE:-}" "${SSL_CERT_FILE:-}" "${PIP_CERT:-}" "$SSLV_PATH"; do
  [ -n "$b" ] && [ -f "$b" ] || continue
  r=$(realp "$b"); [ -n "${SEEN_B[$r]:-}" ] && continue
  SEEN_B[$r]=1; BUNDLES+=("$b")
done

TARGETS=("${CONDA_URLS[@]}")
[ ${#TARGETS[@]} -eq 0 ] && TARGETS=(https://repo.anaconda.com/pkgs/main/noarch/repodata.json https://conda.anaconda.org/conda-forge/noarch/repodata.json)
TARGETS+=("${PIP_INDEX%/}/pip/")
for h in "${EXTRA_HOSTS[@]}"; do TARGETS+=("https://$h/"); done

declare -A DONE_HOST=()
probe() {
  local url=$1 host rc res c
  host=${url#https://}; host=${host#*@}; host=${host%%/*}; host=${host%%:*}
  [ -n "${DONE_HOST[$host]:-}" ] && return; DONE_HOST[$host]=1
  section "6. $host   ($(printf %s "$url" | redact))"
  local via="经代理 $PROXY_HOSTPORT"; bypass_proxy "$host" && via="直连（命中 no_proxy）"
  [ -z "$PROXY_HOSTPORT" ] && via="直连（无代理变量）"
  echo "路由: $via"

  local curl_rc="" curl_res="" sys_ok=0
  for c in "${CURLS[@]}"; do
    res=$(TO 40 "$c" -sS -o /dev/null --max-time 30 -w 'HTTP %{http_code}' "$url" 2>&1); rc=$?
    echo "curl[$c] exit=$rc: $(printf %s "$res" | head -2 | tr '\n' ' ' | redact)"
    [ -z "$curl_rc" ] && { curl_rc=$rc; curl_res=$res; }
  done
  case "$curl_rc" in
    60) hint "$host: curl exit 60 = 证书链不被 curl 的 CA 库信任。这是 curl 自己的判断，和 conda 的 ssl_verify 无关。" ;;
    35) if [[ $curl_res == *"wrong version number"* ]]; then
          hint "$host: curl exit 35 wrong version number = 对明文 HTTP 端点说了 TLS，通常是代理 URL 写成了 https://，不是证书问题。"
        else
          hint "$host: curl exit 35 = TLS 握手失败（不是证书校验失败），留意代理 scheme 或代理是否拦截了该域名。"
        fi ;;
    5|7|56|97) hint "$host: curl exit $curl_rc = 与代理本身的连接/CONNECT 失败（代理地址、端口、403 策略拒绝、407 认证），还没走到证书这一步。" ;;
  esac

  if have openssl; then
    local sargs=(-connect "$host:443" -servername "$host" -showcerts)
    if [ "$via" = "经代理 $PROXY_HOSTPORT" ]; then
      if [ "$PROXY_SCHEME" = http ]; then
        sargs+=(-proxy "$PROXY_HOSTPORT")
        [ -n "$PUSER" ] && sargs+=(-proxy_user "$PUSER" -proxy_pass env:DIAG_PROXY_PASS)
      else
        echo "openssl: 代理 scheme 为 $PROXY_SCHEME，跳过证书链抓取"; sargs=()
      fi
    fi
    if [ ${#sargs[@]} -gt 0 ]; then
      TO 30 openssl s_client "${sargs[@]}" </dev/null >"$OUT/sclient_$host.txt" 2>&1
      if grep -qE 'CONNECT failed|unknown option|connect:errno|Connection refused' "$OUT/sclient_$host.txt"; then
        grep -E 'CONNECT failed|unknown option|errno|refused' "$OUT/sclient_$host.txt" | head -3 | sed 's/^/  openssl: /'
      else
        grep -E '^ *[0-9]+ s:|^ +i:|Verify return code' "$OUT/sclient_$host.txt" | head -20 \
          | sed -e 's/^/  /' -e 's/\(Verify return code.*\)/\1  [openssl 默认信任库]/'
      fi
      awk '/-----BEGIN CERTIFICATE-----/,/-----END CERTIFICATE-----/' "$OUT/sclient_$host.txt" >"$OUT/chain_$host.pem"
      rm -f "$OUT/cert_${host}_"*.pem
      awk -v p="$OUT/cert_${host}_" '/BEGIN CERT/{n++} n{print > (p n ".pem")}' "$OUT/chain_$host.pem"
      local n; n=$(ls "$OUT/cert_${host}_"*.pem 2>/dev/null | wc -l)
      [ "$n" -eq 0 ] && rm -f "$OUT/chain_$host.pem"
      if [ "$n" -gt 0 ]; then
        local leaf=$OUT/cert_${host}_1.pem top=$OUT/cert_${host}_${n}.pem subj iss
        subj=$(openssl x509 -noout -subject -nameopt RFC2253 -in "$top" | sed 's/^subject=//')
        iss=$(openssl x509 -noout -issuer -nameopt RFC2253 -in "$top" | sed 's/^issuer=//')
        echo "链顶: subject=$subj"
        echo "      issuer =$iss"
        openssl x509 -noout -fingerprint -sha256 -in "$top" | sed 's/^/      /'
        if [ "$subj" = "$iss" ] && [ "$n" -gt 1 ]; then
          cp "$top" "$OUT/candidate_ca_$host.pem"
          echo "      链顶是自签根证书，已存为 candidate_ca_$host.pem（使用前请找 IT 核对上面的 SHA256 指纹）"
        fi
        openssl x509 -noout -ext basicConstraints,keyUsage -in "$top" 2>/dev/null | sed 's/^/      /'
        echo "哪些 CA 包能验证这条链（普通 / x509_strict，后者对应 Python 3.13+ 的默认校验）："
        local b r rs any_ok=0 strict_bad=0
        vres() {  # 输出 OK 或第一条 "error N at depth: 原因"
          local o; o=$(openssl verify ${2:-} -CAfile "$1" -untrusted "$OUT/chain_$host.pem" "$leaf" 2>&1)
          if printf '%s' "$o" | grep -q ': OK$'; then echo OK
          else printf '%s\n' "$o" | grep -m1 -oE 'error [0-9]+ at [0-9]+ depth lookup: .*' || printf '%s\n' "$o" | tail -1; fi
        }
        for b in "${BUNDLES[@]}"; do
          r=$(vres "$b"); rs=$(vres "$b" -x509_strict)
          echo "  $b: $r / strict: $rs"
          if [ "$r" = OK ]; then any_ok=1; [[ $b == /etc/* ]] && sys_ok=1; [ "$rs" != OK ] && strict_bad=1; fi
        done
        if [ $any_ok = 0 ]; then
          hint "$host: 代理出示的证书链（链顶签发者: $iss）不被本机任何 CA 包信任，即 TLS 被公司代理重签而本机缺公司根 CA。"
        fi
        [ $strict_bad = 1 ] && hint "$host: 链只在非 strict 模式下通过。base python ≥3.13 的 conda/pip 会拒绝它（如 'CA cert does not include key usage extension'），需 IT 换合规 CA 或用 ≤3.12 的 python。"
      fi
    fi
  fi

  if [ -n "$CONDA_PY" ]; then
    echo "conda 的 HTTP 会话（与 conda create 同一套 requests 配置）："
    local cres; cres=$(TO 90 "$CONDA_PY" "$PROBE" get "$url" 2>&1)
    printf '%s\n' "$cres" | grep -v '^RESULT='
    case "$(printf '%s\n' "$cres" | sed -n 's/^RESULT=//p')" in
      proxy) hint "$host: conda 会话报 ProxyError = 与代理的 CONNECT 阶段失败（403 策略拒绝 / 407 认证 / 代理地址或 scheme 错 / 代理不可达），不是证书问题。" ;;
      ssl)   hint "$host: conda 会话报 SSL 错误；看上面“实际生效 verify”是哪个值，它才是 conda 真正用的 CA 设置。"
             [ $sys_ok = 1 ] && hint "$host: 系统证书库能验证这条链，但 conda 没用它：ssl_verify 设 truststore 或系统 bundle 路径（并确认没有 REQUESTS_CA_BUNDLE/CURL_CA_BUNDLE 抢先）。" ;;
    esac
  fi
}
for u in "${TARGETS[@]}"; do probe "$u"; done

if [ -n "$PIP_PY" ]; then
  section "7. pip 实测（conda env create 的 pip: 段走这条路，不读 .condarc）"
  tmp=$(mktemp -d)
  pres=$(TO 120 "$PIP_PY" -m pip download --no-deps --no-cache-dir --retries 1 --timeout 20 -d "$tmp" six 2>&1)
  printf '%s\n' "$pres" | grep -E 'SSL|certificate|Proxy|Tunnel|ERROR|Saved|Successfully|Looking in' | head -8 | redact
  rm -rf "$tmp"
  printf '%s' "$pres" | grep -qi 'wrong version number' && hint "pip 报 wrong version number：对明文 HTTP 端点说了 TLS，通常是代理 URL 写成了 https://，不是证书问题。"
  printf '%s' "$pres" | grep -qiE 'certificate verify failed|confirming the ssl certificate' && hint "pip 有证书错误：pip 只认 pip.conf 的 cert / PIP_CERT / --cert，conda 的 ssl_verify 对它无效（已复现）。"
  printf '%s' "$pres" | grep -qE 'ProxyError|Tunnel connection failed|Cannot connect to proxy' && hint "pip 与代理的连接/CONNECT 失败（403/407/代理地址），不是证书问题。"
fi

if [ $FULL = 1 ] && [ -n "$CONDA_BIN" ]; then
  section "8. conda create --dry-run（真实路径，可能较慢）"
  TO 900 "$CONDA_BIN" create -n __conda_tls_probe__ --dry-run python 2>&1 | redact | tail -40
fi

# ---------------------------------------------------------------- 结论
section "结论提示（自动生成，按出现顺序）"
if [ ${#HINTS[@]} -eq 0 ]; then echo "未发现明显问题。"; else
  i=0; for h in "${HINTS[@]}"; do i=$((i+1)); echo "$i. $h"; done
fi
cat <<'EOF'

修复方向（按上面的结论选，不要长期保留 ssl_verify:false）：
  A. 系统证书库已信任公司 CA（上面 /etc/... 那行 verify 为 OK）：
       conda config --set ssl_verify truststore          # 需 base python ≥ 3.10；conda 报该值非法说明 conda 太旧，改用下一行
       或 conda config --set ssl_verify /etc/ssl/certs/ca-certificates.crt   # RHEL/EulerOS: /etc/pki/tls/certs/ca-bundle.crt
       pip config set global.cert <同一个系统 bundle 路径>
  B. 本机没有公司 CA：向 IT 要根证书，或用 candidate_ca_<host>.pem（先核对指纹），做“公共 CA + 公司 CA”合并包：
       mkdir -p ~/.certs && cat "$(<conda base>/bin/python -c 'import certifi;print(certifi.where())')" corp-root.pem > ~/.certs/corp-bundle.pem
       conda config --set ssl_verify ~/.certs/corp-bundle.pem
       pip config set global.cert ~/.certs/corp-bundle.pem
       export REQUESTS_CA_BUNDLE=~/.certs/corp-bundle.pem SSL_CERT_FILE=~/.certs/corp-bundle.pem CURL_CA_BUNDLE=~/.certs/corp-bundle.pem
  C. 只要设置了 REQUESTS_CA_BUNDLE / CURL_CA_BUNDLE，它就压过 conda 的 ssl_verify；三处必须指向同一个合并包，或把变量 unset。
  D. 报 ProxyError 时先修代理：地址/端口、http:// scheme、407 账号、403 目标域名被策略拒绝；这一层和证书无关。
EOF
echo; echo "报告已保存: $OUT/report.txt"
