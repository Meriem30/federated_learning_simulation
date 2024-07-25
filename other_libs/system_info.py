import functools
import os
import platform
import subprocess
from enum import StrEnum, auto
from shutil import which

from .util import readlines

"""
    Provide information about the system's operating environment (OS type, processes name)
"""
class OSType(StrEnum):
    Windows = auto()
    FreeBSD = auto()
    Ubuntu = auto()
    ArchLinux = auto()
    CentOS = auto()
    Fedora = auto()
    MacOS = auto()


@functools.cache
def get_operating_system_type() -> OSType:
    sys = platform.system().lower()
    match sys:
        case "windows":
            return OSType.Windows
        case "freebsd":
            return OSType.FreeBSD
        case "darwin":
            return OSType.MacOS
        case "linux":
            pf = platform.platform().lower()
            if "ubuntu" in pf or which("apt-get") is not None:
                return OSType.Ubuntu
            if which("pacman") is not None:
                return OSType.ArchLinux
            if os.path.isfile("/etc/centos-release"):
                return OSType.CentOS
            if os.path.isfile("/etc/fedora-release"):
                return OSType.Fedora
            if which("lsb_release") is not None:
                output = (
                    subprocess.check_output("lsb_release -s -i", shell=True)
                    .decode("utf-8")
                    .strip()
                    .lower()
                )
                if "ubuntu" in output:
                    return OSType.Ubuntu
    raise RuntimeError(f"Unknown OS: {sys}")


@functools.cache
def get_operating_system() -> str:
    sys = platform.system().lower()
    if sys in ("windows", "freebsd"):
        return sys
    if sys == "linux":
        pf = platform.platform().lower()
        if "ubuntu" in pf:
            return "ubuntu"
        if which("pacman") is not None:
            return "archlinux"
        if which("apt-get") is not None:
            return "ubuntu"
        if os.path.isfile("/etc/centos-release"):
            return "centos"
        if os.path.isfile("/etc/fedora-release"):
            return "fedora"
        if which("lsb_release") is not None:
            output = (
                subprocess.check_output("lsb_release -s -i", shell=True)
                .decode("utf-8")
                .strip()
                .lower()
            )
            if "ubuntu" in output:
                return "ubuntu"
    if sys == "darwin":
        return "macos"
    raise RuntimeError(f"Unknown OS: {sys}")


@functools.cache
def get_processor_name() -> str:
    if os.path.isfile("/proc/cpuinfo"):
        return [
            line.lower()
            for line in readlines("/proc/cpuinfo")
            if "model name" in line.lower()
        ][0]
    processor_name = ""
    if which("sysctl") is not None:
        output = os.popen("sysctl hw.model").read().lower()
        if output and "intel" in output:
            processor_name = "intel"

    if not processor_name:
        processor_name = platform.processor().lower()
    assert processor_name
    return processor_name