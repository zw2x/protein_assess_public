import re
import logging
import dataclasses
import subprocess
import shutil
from pathlib import Path
from typing import Dict, Union, Tuple, Optional
from collections import defaultdict, OrderedDict

import numpy as np


logger = logging.getLogger(__file__)


@dataclasses.dataclass(frozen=True)
class TMalignResult:
    tm_score1: float # normalized by length of model
    tm_score2: float # normalized by length of native
    rmsd: float
    msg: str


class TMalignRunner:
    def __init__(self, usalign_binary_path: str):
        super().__init__()

        self.usalign_binary_path = usalign_binary_path

    def run_usalign(self, model: Path, native: Path):
        cmd = [
            self.usalign_binary_path,
            str(model),
            str(native),
            "-mm", "1", "-ter", "0"
        ]
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        output, stderr = process.communicate()
        output_str = output.decode()
        retcode = process.wait()
        if retcode:
            print(output_str)
            print(stderr)
            raise RuntimeError("Error during running TMalign")

        tm_score_rets = re.findall(r"TM-score= ([0-9.]+)", output_str)
        rmsd_rets = re.findall(r"RMSD=   ([0-9.]+)", output_str)
        
        result = TMalignResult(
            tm_score1=float(tm_score_rets[0]),
            tm_score2=float(tm_score_rets[1]),
            rmsd=float(rmsd_rets[0]),
            msg=output_str
        )
        return result, cmd
