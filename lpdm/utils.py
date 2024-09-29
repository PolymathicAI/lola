r"""Miscellaneous helpers."""

import os
import shutil


def process_cpu_count(pid: int = 0) -> int:
    r"""Returns the number of logical CPUs usable by the calling thread of the current process."""

    return len(os.sched_getaffinity(pid))


def map_to_memory(
    file: str,
    shm: str = "/dev/shm",
    exist_ok: bool = False,
) -> str:
    r"""Maps a file to memory.

    Arguments:
        file: The source file to map.
        shm: The shared memory filesystem.

    Returns:
        The file's destination.
    """

    src = os.path.abspath(os.path.expanduser(file))
    dst = os.path.join(shm, os.path.relpath(file, "/"))

    if os.path.exists(dst):
        if exist_ok:
            return dst
        else:
            raise FileExistsError(f"{dst} already exists.")
    else:
        os.makedirs(os.path.dirname(dst), exist_ok=True)

    size = os.path.getsize(src)
    free = os.statvfs(shm).f_frsize * os.statvfs(shm).f_bavail

    if size < free:
        return shutil.copy2(src, dst)
    else:
        raise MemoryError(f"not enough space on {shm} (needed: {size} B, free: {free} B).")
