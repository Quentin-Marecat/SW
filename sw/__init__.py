import os.path
import logging
import subprocess
import importlib
import platform

__version__ = "1.0.0"

log = logging.getLogger(__name__)

# --- Required modules


def import_package(name, required=True):
    fmt = "  * %-10s  v%-8s  location: %s"
    try:
        package = importlib.import_module(name.lower())
        try :
            package.__version__
        except AttributeError:
            package.__version__='0.0.0'
        log.debug(fmt, name, package.__version__, os.path.dirname(package.__file__))
        return package
    except ImportError:
        if required:
            log.critical("%s not found.", name)
            raise
        log.debug("%s not found.", name)
        return None


log.debug("Required packages:")
numpy = import_package("NumPy")
import_package("SciPy")
import_package("h5py")
import_package("quantnbody")
import_package("qiskit")
import_package("qiskit_nature")
import_package("qiskit_aer")
import_package("qiskit_algorithms")

# --- Git hashes


def get_git_hash(dir):
    git_dir = os.path.join(dir, ".git")
    cmd = ["git", "--git-dir=%s" % git_dir, "rev-parse", "--short", "HEAD"]
    try:
        githash = subprocess.check_output(cmd, universal_newlines=True, stderr=subprocess.STDOUT).rstrip()
    except subprocess.CalledProcessError:
        githash = "<Not Found>"
    return githash


log.debug("Git hashes:")
vdir = os.path.dirname(os.path.dirname(__file__))
vhash = get_git_hash(vdir)

# --- System information
log.debug("System:  node= %s  processor= %s" % (platform.node(), platform.processor()))

# --- NumPy
numpy.set_printoptions(linewidth=120)