"""
语料处理相关常用函数
"""
import os
import doctest
import platform
import pathlib  # py3


def remain_chinese(sequences):
    """
    保留中文字符

    """


def load_stopwords(filepath, encoding="utf8"):
    """
    Args:
        filepath(str):
        encoding:

    Examples:
        >>> stopwords_en = load_stopwords("./data/stopwords_en")
        >>> 'the' in stopwords_en
        True

    Returns:
        set
    """
    with open(filepath, encoding=encoding) as f:
        # 去空白符，转小写（不影响中文）
        stopwords = set(word.strip().lower() for word in f if not word.isspace())

    return stopwords


def remove_stopwords(tokens, stopwords, encoding='utf8'):
    """remove stopwords

    Args:
        tokens(list of str): a list of tokens/words
        stopwords(list of str or str): a list of words or a filepath
        encoding:

    Examples:
        >>> from huaytools.nlp.stopwords import stopwords_en
        >>> remove_stopwords(['huay', 'the'], stopwords_en)
        ['huay']

    """
    if isinstance(stopwords, str):
        stopwords = load_stopwords(stopwords, encoding)
    else:
        stopwords = set(stopwords)

    return [w for w in tokens if w not in stopwords]


def clear_str_en(text):
    """clear the en text

    """


if __name__ == '__main__':
    """"""
    doctest.testmod()

    # with open('stopwords_en', encoding='utf8') as f:
    #     stopwords = set(word.strip() for word in f if not word.isspace())
    #     print(stopwords)
