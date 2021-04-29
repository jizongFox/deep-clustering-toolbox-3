import os
import subprocess
from functools import partial
from typing import List, Union

from deepclustering3.submiter.utils import randomString, random_account


def _create_sbatch_prefix(*, account: str, time=1, job_name="default_jobname", nodes=1, gres="gpu:1", cpus_per_task=6,
                          mem=16, mail_user="jizong.peng.1@etsmtl.net"):
    return (
        f"#!/bin/bash \n"
        f"#SBATCH --time=0-{time}:00 \n"
        f"#SBATCH --account={account} \n"
        f"#SBATCH --cpus-per-task={cpus_per_task} \n"
        f"#SBATCH --gres={gres} \n"
        f"#SBATCH --account={account} \n"
        f"#SBATCH --job-name={job_name} \n"
        f"#SBATCH --nodes={nodes} \n"
        f"#SBATCH --mem={mem}000M \n"
        f"#SBATCH --mail-user={mail_user} \n"
        f"#SBATCH --mail-type=FAIL \n"
    )


class CCSubmitter:
    def __init__(self, job_array: List[str], work_dir=None, account_sampler=random_account()) -> None:
        self._job_array = job_array
        self._work_dir = work_dir
        self._sbatch_prefix = None
        self._acc_sampler = account_sampler
        self._env = []
        self.__sbatch_initialized__ = False
        self.__env_initialized__ = False

    @staticmethod
    def create_sbatch_prefix(**kwargs):
        return partial(_create_sbatch_prefix, **kwargs)

    def create_env(self, cmd_list: Union[str, List[str]] = None) -> List[str]:
        if isinstance(cmd_list, str):
            cmd_list = [cmd_list, ]
        self.__env_initialized__ = True
        return cmd_list or []

    def submit(self, on_local=False):
        if not self.__sbatch_initialized__:
            self.create_sbatch_prefix()
        if not self.__env_initialized__:
            self.create_env()

        env_script = "\n".join(self._env)

        for job in self._job_array:
            account = next(self._acc_sampler)
            script = "\n".join([self.create_sbatch_prefix()(account=account), env_script, job])
            self._write_and_run(script, on_local)

    def _write_and_run(self, full_script, on_local):
        random_name = randomString() + ".sh"
        workdir = os.path.dirname(__file__)
        if self._work_dir:
            workdir = os.path.abspath(self._work_dir)
        random_bash = os.path.join(workdir, random_name)

        with open(random_bash, "w") as f:
            f.write(full_script)
        try:
            if on_local:
                code = subprocess.call(f"bash {random_bash}", shell=True)
            else:
                code = subprocess.call(f"sbatch {random_bash}", shell=True)
        finally:
            os.remove(random_bash)
        return code


if __name__ == "__main__":
    pass
