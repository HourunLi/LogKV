import os
import ssl
import urllib3

# 忽略控制台的 InsecureRequestWarning 警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 强制清空系统的 CA 证书环境变量，让请求不去做强校验
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
ssl._create_default_https_context = ssl._create_unverified_context

from modelscope.hub.snapshot_download import snapshot_download

# 开始下载
model_dir = snapshot_download('LLM-Research/Llama-3.2-1B', cache_dir='./checkpoints/meta-llama')
print(f"模型下载到了: {model_dir}")