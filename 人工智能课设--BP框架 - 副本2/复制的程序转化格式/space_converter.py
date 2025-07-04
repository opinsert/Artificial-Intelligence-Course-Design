import sys


def convert_whitespace(text):
    """
    将文本中的所有空白字符转换为英文格式（UTF-8）
    包括：全角空格 → 半角空格，全角制表符 → 半角制表符，全角换行符 → 半角换行符
    """
    # 替换规则
    replacements = {
        '\u3000': ' ',  # 全角空格 → 半角空格
        '\uFF09': '\t',  # 全角制表符 → 半角制表符 (U+FF09 是全角右括号，但这里假设存在全角制表符)
        '\u300A': '\n',  # 全角换行符 → 半角换行符 (U+300A 是全角左书名号，但这里假设存在全角换行符)
        '\u00A0': ' ',  # 不换行空格 → 普通空格
        '\u200B': '',  # 零宽度空格 → 删除
        '\uFEFF': ''  # 零宽度不换行空格 → 删除
    }

    # 应用替换规则
    for fullwidth, halfwidth in replacements.items():
        text = text.replace(fullwidth, halfwidth)

    return text


def main():
    # 从标准输入读取所有文本
    input_text = sys.stdin.read()

    # 转换空白字符
    converted_text = convert_whitespace(input_text)

    # 输出到标准输出
    sys.stdout.write(converted_text)


if __name__ == "__main__":
    main()
