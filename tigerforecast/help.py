# tigerforecast.help() method

from tigerforecast import problem_registry, method_registry


def help():
    s_prob, s_mod = "\n", "\n"
    for problem_id in problem_registry.list_ids():
        s_prob += "\t" + problem_id + "\n"
    for method_id in method_registry.list_ids():
        s_mod += "\t" + method_id + "\n"
    print(global_help_string.format(s_prob, s_mod))



global_help_string = """

Welcome to TigerForecast!

If this is your first time using TigerForecast, you might want to read more about it in 
detail at github.com/johnhallman/tigerforecast, or documentation at tigerforecast.readthedocs.io.

If you're looking for a specific Problem or Method, you can call it via the 
tigerforecast.problem and tigerforecast.method methods respectively, such as:

    problem = tigerforecast.problem("nameOfMethod")

Below is the list of all currently avaliable problems and methods:

    Problems
    ---------
    {}

    Methods
    ---------
    {}

Good luck exploring TigerForecast!

"""


