import pytest

# simple ---------------------------
def calc(a,b):
    return a+b

def test_calc():
    assert calc(3,5) == 8

# error ---------------------------
def raise_error():
    raise Exception("dummy error")

def test_error():
    with pytest.raises(Exception):
        raise_error()

# class ---------------------------
def foo(msg):
    return "foo " + msg

class TestFoo:
    def test_foo_tom(self):
        assert foo("tom") == "foo tom"

    def test_foo_bar(self):
        assert foo("bar") == "foo bar"

# funcarg -------------------------
def pytest_funcarg__mike(request):
    return "MIKE"

def test_hello_mike(mike):
    assert mike == "MIKE"
