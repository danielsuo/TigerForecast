"""
Run all tests for the TigerControl framework
"""

import tigerforecast

# test all tigerforecast.* methods
def test_tigerforecast_functionality(show_results=False):
    print("\nrunning all tigerforecast functionality tests...\n")
    test_help()
    test_error()
    print("\nall tigerforecast functionality tests passed\n")


# test tigerforecast.help() method
def test_help():
    tigerforecast.help()


def test_error():
    try:
        from tigerforecast.error import Error
        raise Error()
    except Error:
        pass

if __name__ == "__main__":
    test_tigerforecast_functionality(show_results=False)