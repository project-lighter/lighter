from lighter.utils.misc import ensure_list


def test_ensure_list():
    assert ensure_list(1) == [1]
    assert ensure_list([1, 2]) == [1, 2]
