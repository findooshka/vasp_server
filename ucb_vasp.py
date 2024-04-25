import subprocess, os
import pandas as pd
import numpy as np
import yaml
from collections import defaultdict
import scipy.stats
import shutil
from collections import Counter
import time
from pymatgen.io.cif import CifParser
from pymatgen.io.vasp import Poscar
from pymatgen.analysis.structure_matcher import StructureMatcher
import argparse
import scipy as sp
from math import comb


def lower_bound(X, t, cutoff=0.95):
    if len(X) < 2:
        return -1000
    mu0, tau2, alpha, beta = 0, 10, 1, 5
    n = len(X)
    X_bar = np.mean(X)
    s2 = np.var(X, ddof=1)
    
    tau2_n = 1 / (1/tau2 + n/s2)
    mu_n = tau2_n * (mu0/tau2 + n*X_bar/s2)
    
    alpha_n = alpha + n / 2
    beta_n = beta + 0.5 * np.sum((X - X_bar)**2) + 0.5 * n * (1/tau2) / (1/tau2 + n/s2) * (X_bar - mu0)**2
    
    return mu_n - sp.stats.invgamma.ppf(.95, alpha_n, scale=beta_n)*(np.log(t))**.5



def calculate_redunduncy(path):
    files = os.listdir(path)
    matcher = StructureMatcher()
    uniques = 0
    structs = []
    for i, file in enumerate(files):
        structs.append(Poscar.from_file(os.path.join(path, file)).structure)
    for i, struct in enumerate(structs):
        for j in range(i):
            if matcher.fit(struct, structs[j]):
                break
        else:
            uniques += 1
    if len(files) == 0:
        return 0
    return uniques, len(files)

def K_knowing_M(K, M, N):
    return comb(M, K) * sum((-1)**j * comb(K, j) * ((K-j)**N) for j in range(K+1)) / M**N

def M_prior(M, p):
    return p*(1-p)**(M-1)
    
def K_marginal(K, N, p, max_M=30):
    marginal = 0
    for m in range(1, max_M+1):
        marginal += M_prior(m, p)*K_knowing_M(K, m, N)
    return marginal

def M_knowing_K(M, K, N, max_M=30, p=0.2):
    if N == 0:
        return 0
    return M_prior(M, p=p)*K_knowing_M(K, M, N)/K_marginal(K,N,p=p,max_M=max_M)

def skip_space_group(name, sg):
    path = os.path.join('vasp', 'CONTCARs', name, str(sg))
    if not os.path.exists(path):
        print(f"No structures in space group {sg} yet.")
        prob = 0
    else:
        K, N = calculate_redunduncy(path)
        prob = M_knowing_K(K, K, N, max_M = 100+N)
        print(f"{K} unique structs among {N} in space group {sg}, skip prob: {prob}")
    return np.random.rand() < prob

def choose_pot(element):
    if element == 'Zr':
        return 'Zr_sv'
    if element == 'Ca':
        return 'Ca_sv'
    return element

def make_potcar():
    with open('POSCAR', 'r') as f:
        lines = f.readlines()
        comp = lines[5].strip().split()
    with open('POTCAR', 'w') as f:
        for element in comp:
            with open(os.path.join('pots', choose_pot(element), 'POTCAR'), 'r') as f_in:
                f.write(f_in.read())
            
        
def setup_files():
    if os.path.exists('OSZICAR'):
        os.remove('OSZICAR')
    make_potcar()
    
def read_energy():
    try:
        with open(os.path.join('vasp', 'OSZICAR'), 'r') as f:
            last_line = f.readlines()[-1].strip()
            last_line = last_line[last_line.find('F=')+2:last_line.find('E0=')].strip()
            energy = float(last_line)
    except:
        return None
    return energy
    
def check_distances(structure, threshold=0.5):
    distance_mat = structure.lattice.get_all_distances(structure.frac_coords, structure.frac_coords)
    distance_mat += 1000*np.identity(distance_mat.shape[0], dtype=float)
    return distance_mat.min() < threshold
    
def enthalpy(energy, composition, reference_energies):
    result = energy
    total = np.sum(list(composition.as_dict().values()))
    for element in composition:
        result -= composition[element.name] / total * reference_energies[element.name]
    return result    

def vasp_energy(file_path, reference_energies):
    try:
        parser = CifParser(file_path)
        structure = parser.get_structures()[0]
    except:
        return None
    if check_distances(structure):
        return None
    Poscar(structure).write_file(os.path.join('vasp', 'POSCAR'))
    os.chdir('vasp')
    setup_files()
    out = subprocess.run([os.path.join('./run_vasp.sh')], capture_output=True)
    with open('vasp_output.txt', 'r') as f:
        success = 'Error' not in f.read()
    os.chdir('..')
    if success:
        result = read_energy()
        if result is None:
            return None
        n_atoms = np.sum(list(structure.composition.as_dict().values()))
        result /= n_atoms
        result = enthalpy(result, structure.composition, reference_energies)
        if result < 8:
            return result
    return None

def estimate_energies_vasp(task_name, path, current_energies, reference_energies):
    struct_names = os.listdir(path)
    print(path)
    print(struct_names)
    _, sg = parse_name(path.rstrip('/').split('/')[-1])
    energies = {sg: []}
    for name in struct_names:
        energy = vasp_energy(os.path.join(path, name), reference_energies)
        os.chdir('/home/arsen/Documents/git/diffusion_atoms_batch')
        print(energy)
        if energy != None:
            if not os.path.exists(os.path.join('vasp', 'CONTCARs', task_name)):
                os.mkdir(os.path.join('vasp', 'CONTCARs', task_name))
            if not os.path.exists(os.path.join('vasp', 'CONTCARs', task_name, str(sg))):
                os.mkdir(os.path.join('vasp', 'CONTCARs', task_name, str(sg)))
            shutil.copyfile(os.path.join('vasp', 'CONTCAR'), os.path.join('vasp', 'CONTCARs', task_name, str(sg), str(len(energies[sg]) + len(current_energies[sg]))))
            energies[sg].append(float(energy))
    return energies



def parse_name(name):
    print(name)
    i, c = 0, name[0]
    composition = []
    while c != '_':
        element = ''
        number = ''
        while 'a' <= c <= 'z' or 'A' <= c <= 'Z':
            element += c
            i += 1
            c = name[i]
        while '0' <= c <= '9':
            number += c
            i += 1
            c = name[i]
        number = int(number)
        composition += [element for _ in range(number)]
    sg = int(name[i+1:])
    return composition, sg

def prepare_order(path, name, space_groups, comp, count=2, T=60):
    order = {'name': name,
             'orders': []}
    for sg in space_groups:
        order['orders'].append({'space_group': [int(sg)], 'composition': comp, 'count': count, 'T': T, 'only_final': True})
    with open(os.path.join(path, name + '.yml'), 'w') as f:
        yaml.dump(order, f)

def sample(order_name):
    process = subprocess.Popen(['python3', 'main.py', '--config', 'alex.yml', '--doc', 'alex', '--sample', '--sampling_order_path', order_name+'.yml', '--ni'])
    return process

def sort_space_groups(name, energy_estimates, t, exclusion=set()):
    bound = np.empty((len(energy_estimates), 2))
    for i, sg in enumerate(energy_estimates):
        if sg in exclusion or skip_space_group(name=name, sg=sg):
            bound[i] = 1e5
        else:
            bound[i] = (sg, lower_bound(energy_estimates[sg], t))
    return bound[:,0][np.argsort(bound[:,1])].astype(int)

def write_energies(name, current_energies):
    with open(f'energies_{name}.yml', 'w') as f:
        yaml.dump(current_energies, f)

def read_energies(name):
    with open(f'energies_{name}.yml', 'r') as f:
        return yaml.safe_load(f)

def ucb_algorithm(task_name, composition, reference_energies, n_steps, sg_per_step=1, samples_per_sg=4, start_step=0):
    assert set(composition.keys()).issubset(reference_energies.keys())
    samples_path = '/home/arsen/Documents/git/diffusion_atoms_batch/exp/cif_samples/alex/'
    t = 2 + samples_per_sg * sg_per_step * start_step
    if start_step == 0:
        current_energies = {}
        for i in range(16, 231):
            current_energies[i] = []
    else:
        current_energies = read_energies(task_name)
        for sg in current_energies:
            t += len(current_energies[sg])
    chosen_space_groups = []
    for i in range(start_step, start_step+n_steps):
        write_energies(task_name, current_energies)
        sorted_space_groups = sort_space_groups(task_name, current_energies, t, exclusion=set(chosen_space_groups))
        chosen_space_groups = sorted_space_groups[:sg_per_step]
        name = f'ucb_step_{i}'
        last_step_name = f'ucb_step_{i-1}'
        prepare_order('sampling_orders/',
                      name,
                      chosen_space_groups,
                      composition,
                      count=samples_per_sg)
        process = sample(name)
        if i > 0:
            for dir in os.listdir(os.path.join(samples_path, last_step_name)):
                estimates = estimate_energies_vasp(task_name=task_name,
                                                   path=os.path.join(samples_path, last_step_name, dir),
                                                   current_energies=current_energies,
                                                   reference_energies=reference_energies)
                if estimates != None:
                    for sg in estimates:
                        current_energies[sg] += estimates[sg]
                else:
                    print(f"No good structures in {os.path.join(samples_path, last_step_name, dir)}")
        while process.poll() == None:
            time.sleep(1)
        t += samples_per_sg * sg_per_step
    return current_energies
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--start_step",
        type=int,
        default=0
    )
    parser.add_argument(
        "--n_steps",
        type=int,
        default=1000
    )
    args = parser.parse_args()
    pure_state_E = {
        'Re': -12.42453,
        'V': -8.94119,
        'Zr': -8.51973,
        'Ca': -1.757258625,
        'Fe': -7.7654815,
        'B': -6.703473333333333
    }
    energies = ucb_algorithm("CaFeB",
                             {'Ca': 1, 'Fe': 1, 'B': 1},
                             pure_state_E,  # energies per atom for of the pure structures
                             n_steps=args.n_steps,
                             sg_per_step=1,
                             samples_per_sg=4,
                             start_step=args.start_step)
