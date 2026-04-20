import json

# 替换成你下载的 longabc jsonl 文件路径
file_path = "/home/ma-user/work/bucket-wulan-green/zhaoyusheng/longabc/LongABC_Code.jsonl" 

with open(file_path, 'r', encoding='utf-8') as f:
    for _ in range(10):
        # 只读取第一行
        first_line_str = f.readline()
        
        # 解析成字典
        data_dict = json.loads(first_line_str)
        
        # 打印所有的 key
        print("👉 这个文件包含的 Keys 有:", data_dict.keys())
        
        # 稍微打印一点内容，确认哪个是你需要的正文
        for key, value in data_dict.items():
            # 如果内容太长，只打印前 100 个字符
            preview = str(value)[:1000] + "..." if len(str(value)) > 100 else str(value)
            print(f"[{key}]: {preview}")