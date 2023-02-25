#!/usr/bin/env python3

from my_solution import solution


# 测试用例:找出不同类的词
def test_solution():
    words = "flower grass cat"
    # 正确答案
    correct_solution = 'cat'
    
    # 程序求解结果
    result = solution(G)
    assert correct_solution == result
