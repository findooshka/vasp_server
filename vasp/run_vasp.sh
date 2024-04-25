#!/bin/bash
head /opt/intel/oneapi/setvars.sh
source /opt/intel/oneapi/setvars.sh
mpiexec -n 10 vasp > vasp_output.txt
