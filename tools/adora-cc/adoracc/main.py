#!/usr/bin/env python3
import sys
import re

def strip_module_attrs(text: str) -> str:
    """
    Remove the content of:
        module attributes { ... } 
    in MLIR, leaving only empty {}.
    Supports multi-line and nested braces.
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
            # Format does not match, skip
            i = brace_start
            continue

        # Write empty {}
        out.append("{}")

        # Skip the original {...}
        j = brace_start + 1
        brace_depth = 1
        while j < n and brace_depth > 0:
            if text[j] == '{':
                brace_depth += 1
            elif text[j] == '}':
                brace_depth -= 1
            j += 1

        i = j  # Continue scanning after the closing brace

    return "".join(out)


def main():
    if len(sys.argv) != 2:
        print("Usage: python clean_module_attrs.py <file.mlir>")
        sys.exit(1)

    mlir_file = sys.argv[1]

    # Read
    with open(mlir_file, "r", encoding="utf-8") as f:
        text = f.read()

    # Clean
    cleaned = strip_module_attrs(text)

    # Overwrite in place
    with open(mlir_file, "w", encoding="utf-8") as f:
        f.write(cleaned)

    # print(f"Cleared module attributes: {mlir_file}")


if __name__ == "__main__":
    main()
