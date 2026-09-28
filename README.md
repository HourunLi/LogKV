curl -skL --suppress-connect-headers -D - -o /tmp/hf403.html https://hf-mirror.com/ \
  | grep -iE '^HTTP|^server|^via|^proxy|^x-|^cf-'
head -c 1500 /tmp/hf403.html; echo

# 对照：普通站点、huggingface 官方站
for u in https://pypi.org/simple/ https://huggingface.co/ ; do
  curl -sk --suppress-connect-headers -o /dev/null -w "%{http_code}  $u\n" "$u"; done
