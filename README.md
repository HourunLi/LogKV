[ma-user lihourun]$curl -skL --suppress-connect-headers -D - -o /tmp/hf403.html https://hf-mirror.com/ \
>   | grep -iE '^HTTP|^server|^via|^proxy|^x-|^cf-'
HTTP/1.1 403 Forbidden
Server: netentsec_page_push
[ma-user lihourun]$head -c 1500 /tmp/hf403.html; echofor u in https://pypi.org/simple/ https://huggingface.co/ ; do
bash: syntax error near unexpected token `do'
[ma-user lihourun]$  curl -sk --suppress-connect-headers -o /dev/null -w "%{http_code}  $u\n" "$u"; done
bash: syntax error near unexpected token `done'
[ma-user lihourun]$
[ma-user lihourun]$for u in https://pypi.org/simple/ https://huggingface.co/ ; do
>   curl -sk --suppress-connect-headers -o /dev/null -w "%{http_code}  $u\n" "$u"; done

200  https://pypi.org/simple/
200  https://huggingface.co/
