set -euo pipefail

if ! mkdir -p "${HOME}" "${HF_HOME}" "${TORCH_HOME}" "${XDG_CACHE_HOME}"; then
  cat >&2 <<'EOF'
ERROR: PI0.5 runtime directories are not writable for the selected host UID.
Mount the writable host cache root at /cache (for example, -v HOST_CACHE:/cache).
EOF
  exit 73
fi

if ! getent passwd "$(id -u)" >/dev/null 2>&1; then
  nss_dir="/tmp/ebim-nss-$(id -u)"
  mkdir -p "${nss_dir}"
  cp /etc/passwd "${nss_dir}/passwd"
  cp /etc/group "${nss_dir}/group"
  printf 'ebim:x:%s:%s:EBiM user:%s:/bin/bash\n' \
    "$(id -u)" "$(id -g)" "${HOME}" >> "${nss_dir}/passwd"
  printf 'ebim:x:%s:\n' "$(id -g)" >> "${nss_dir}/group"
  export NSS_WRAPPER_PASSWD="${nss_dir}/passwd"
  export NSS_WRAPPER_GROUP="${nss_dir}/group"
  export LD_PRELOAD="$(find /usr/lib -name libnss_wrapper.so -print -quit)${LD_PRELOAD:+:${LD_PRELOAD}}"
  export USER=ebim
  export LOGNAME=ebim
fi

exec python -m task2_isaacsim.baselines.pi05.portable "$@"
