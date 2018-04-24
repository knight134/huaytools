"""
Common tool functions
"""
import re
import doctest


def is_chinese(char):
    """
    判断是否汉字

    汉字 Unicode 编码范围 (http://www.qqxiuzi.cn/zh/hanzi-unicode-bianma.php)
        字符集	    字数	Unicode 编码
        基本汉字	    20902字	    4E00-9FA5
        基本汉字补充	38字	        9FA6-9FCB
        扩展A	    6582字	    3400-4DB5
        扩展B	    42711字	    20000-2A6D6
        扩展C	    4149字	    2A700-2B734
        扩展D	    222字	    2B740-2B81D
        康熙部首	    214字	    2F00-2FD5
        部首扩展	    115字	    2E80-2EF3
        兼容汉字	    477字	    F900-FAD9
        兼容扩展	    542字	    2F800-2FA1D
        PUA(GBK)部件	81字	        E815-E86F
        部件扩展	    452字	    E400-E5E8
        PUA增补	    207字	    E600-E6CF
        汉字笔画	    36字	        31C0-31E3
        汉字结构	    12字	        2FF0-2FFB
        汉语注音	    22字	        3105-3120
        注音扩展	    22字	        31A0-31BA
        〇	        1字	        3007

    Examples:
        >>> is_chinese("华")
        True
        >>> is_chinese("a")
        False

    Args:
        char:

    Returns:

    """
    if '\u4e00' <= char <= '\u9fa5':
        return True
    return False


def _remove_duplicate(src, dst=None, encoding="utf8"):
    """
    Examples:
        >>> _remove_duplicate("data/stopwords_zh")
    """
    with open(src, encoding=encoding) as f:
        s = set()
        l = list()
        for i in f:
            i = i.strip().lower()
            if i not in s:
                s.add(i)
                l.append(i)

    if dst is None:
        dst = src

    with open(dst, 'w', encoding=encoding) as f:
        for i in l:
            f.write(i)
            f.write('\n')


if __name__ == '__main__':
    doctest.testmod()
