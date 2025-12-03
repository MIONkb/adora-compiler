#!/usr/bin/env python3
import sys
import re

def strip_module_attrs(text: str) -> str:
    """
    删除 MLIR 中:
        module attributes { ... } 
    的内容，只保留空的 {}。
    支持跨行与嵌套括号。
    """
    TOKEN = "module attributes "

    out = []
    i = 0
    n = len(text)

    while True:
        pos = text.find(TOKEN, i)
        if pos == -1:
            out.append(text[i:])
            break

        out.append(text[i:pos])
        out.append(TOKEN)

        brace_start = pos + len(TOKEN)

        if brace_start >= n or text[brace_start] != '{':
            # 格式不符合，也跳过
            i = brace_start
            continue

        # 写入空 {}
        out.append("{}")

        # 跳过原始 {...}
        j = brace_start + 1
        brace_depth = 1
        while j < n and brace_depth > 0:
            if text[j] == '{':
                brace_depth += 1
            elif text[j] == '}':
                brace_depth -= 1
            j += 1

        i = j  # 从右括号后继续扫描

    return "".join(out)


def main():
    if len(sys.argv) != 2:
        print("用法: python clean_module_attrs.py <file.mlir>")
        sys.exit(1)

    mlir_file = sys.argv[1]

    # 读取
    with open(mlir_file, "r", encoding="utf-8") as f:
        text = f.read()

    # 清理
    cleaned = strip_module_attrs(text)

    # 覆盖写回
    with open(mlir_file, "w", encoding="utf-8") as f:
        f.write(cleaned)

    # print(f"已清除 module attributes: {mlir_file}")


if __name__ == "__main__":
    main()
