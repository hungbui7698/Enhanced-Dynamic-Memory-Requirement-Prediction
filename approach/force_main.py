from multiprocessing import Process

from approach.main import main

if __name__ == "__main__":
    exitcode = 1
    while exitcode != 0:
        p = Process(target=main)
        p.start()
        p.join()
        exitcode = p.exitcode
