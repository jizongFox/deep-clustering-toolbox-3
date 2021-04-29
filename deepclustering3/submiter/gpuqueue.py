import argparse
import os
import re
from queue import Queue, Empty
from subprocess import run
from threading import Thread
from typing import List, Union


def get_args():
    parser = argparse.ArgumentParser(description="Dynamic gpu job submitter")
    parser.add_argument("jobs", nargs="+", type=str)
    parser.add_argument("--available_gpus", type=str, nargs="+", default=["0"], metavar="N",
                        help="Available GPUs")
    args = parser.parse_args()
    return args


class GPUQueue:

    def __init__(self, job_array: List[str], available_gpus: List[Union[str, int]] = None, verbose=False) -> None:
        super().__init__()
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        self.job_array = job_array
        self.available_gpus = available_gpus
        self.job_queue = Queue()
        self.gpu_queue = Queue()
        self.verbose = verbose
        self._put_job2queue()
        self._put_gpuindex2queue()
        self._results = {}
        self._thread_pool = []

    def _put_job2queue(self):
        for job in self.job_array:
            self.job_queue.put(job)

    def _put_gpuindex2queue(self):
        for gpu in self.available_gpus:
            self.gpu_queue.put(gpu)

    def submit(self):
        while True:
            try:
                available_gpu = self.gpu_queue.get()  # this will wait forever
                job_str = self.job_queue.get(timeout=1)  # if it is going te be empty, end the program
                thread = self.process_daemon(job_str, available_gpu)
                self._thread_pool.append(thread)
            except Empty:  # the jobs are done
                for t in self._thread_pool:
                    t.join()
                self._print(self._results)
                break

    def _process_daemon(self, job, gpu):
        new_environment = os.environ.copy()
        new_environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
        result_code = run(job, shell=True, env=new_environment)
        self._results[job] = result_code.returncode
        self.gpu_queue.put(gpu)

    def process_daemon(self, job, gpu):
        thread = Thread(target=self._process_daemon, args=(job, gpu), )
        thread.start()
        return thread

    def _print(self, result_dict):
        for k, v in result_dict.items():
            k = ' '.join(re.split(' +|\n+', k)).strip()
            print(f"Job:\n{k}")
            print("result_code", v)


def main():
    args = get_args()
    submitter = GPUQueue(args.jobs, args.available_gpus, verbose=False)
    submitter.submit()


if __name__ == "__main__":
    main()
