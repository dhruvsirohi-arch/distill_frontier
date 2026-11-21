"""Mirror a running exam's (unlinked) output file back to its path, complete lines only."""
import os, sys, time
pid, fd, dst = sys.argv[1], sys.argv[2], sys.argv[3]
src = f"/proc/{pid}/fd/{fd}"
while os.path.exists(f"/proc/{pid}"):
    try:
        data = open(src, "rb").read()
        data = data[:data.rfind(b"\n") + 1]
        tmp = dst + ".sync"
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, dst)
    except OSError:
        pass
    time.sleep(5)
