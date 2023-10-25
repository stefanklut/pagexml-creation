import os
import sys


def get_logger_name():
    frame = sys._getframe(1)
    while frame:
        code = frame.f_code
        if os.path.join("utils", "logging_utils.") not in code.co_filename:
            mod_name = frame.f_globals["__name__"]
            if mod_name == "__main__":
                mod_name = "pagexml-creation"
            else:
                mod_name = "pagexml-creation." + mod_name
            # return mod_name, (code.co_filename, frame.f_lineno, code.co_name)
            return mod_name
        frame = frame.f_back