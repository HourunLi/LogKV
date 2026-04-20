import os
import zstandard as zstd
from pathlib import Path
import argparse

def decompress_zst_to_jsonl(directory_path):
    """
    遍历指定文件夹，将所有 .jsonl.zst 文件解压为 .jsonl 文件。
    采用流式解压（copy_stream），以最低的内存占用处理超大文件。
    """
    folder = Path(directory_path)
    
    # 检查路径是否存在且为文件夹
    if not folder.is_dir():
        print(f"错误：路径 '{directory_path}' 不是一个有效的文件夹。")
        return

    # 查找所有的 .jsonl.zst 文件
    zst_files = list(folder.rglob('*.jsonl.zst')) # 使用 rglob 支持递归查找子文件夹
    
    if not zst_files:
        print(f"在 '{directory_path}' 中没有找到 .jsonl.zst 文件。")
        return

    print(f"共找到 {len(zst_files)} 个文件，开始解压...\n" + "-"*40)

    # 创建一个 Zstd 解压器实例
    dctx = zstd.ZstdDecompressor()

    success_count = 0
    fail_count = 0

    for zst_file in zst_files:
        # 移除最后的 .zst 后缀，生成 .jsonl 文件名
        output_file = zst_file.with_suffix('')
        
        print(f"正在解压: {zst_file.name}  ->  {output_file.name}")
        
        try:
            # 以二进制读取模式打开压缩文件，以二进制写入模式打开目标文件
            with open(zst_file, 'rb') as compressed_file:
                with open(output_file, 'wb') as uncompressed_file:
                    # copy_stream 会分块读取和解压，非常适合几个GB甚至更大的文件
                    dctx.copy_stream(compressed_file, uncompressed_file)
            success_count += 1
        except Exception as e:
            print(f"解压文件 {zst_file.name} 时发生错误: {e}")
            fail_count += 1

    print("-" * 40)
    print(f"任务完成！成功: {success_count} 个, 失败: {fail_count} 个。")

if __name__ == "__main__":
    # 使用 argparse 让脚本可以通过命令行接收文件夹路径参数
    parser = argparse.ArgumentParser(description="批量解压 .jsonl.zst 文件为 .jsonl")
    parser.add_argument(
        "folder_path", 
        nargs="?", 
        default=".", 
        help="包含 .jsonl.zst 文件的文件夹路径 (默认: 当前目录)"
    )
    
    args = parser.parse_args()
    decompress_zst_to_jsonl(args.folder_path)
