import os


def get_ray_cache_dir():
    import socket
    import getpass

    hostname = socket.gethostname()  # check machine
    username = getpass.getuser()
    print("Running on hostname:", hostname)
    return os.path.abspath("./ray_cache")
