import json
import logging
import subprocess
import os
import shutil
from uuid import uuid4
from collections import Counter
from jarvis.core.atoms import Atoms
from hull_calculation import distance_to_hull
import zipfile


class DataHandler:
    def __init__(self, download_handler):
        self.download_handler = download_handler
        
    def load_data(self, config):
        self.config = config['data']
        
        logging.info('Loading data...')
        with open(self.config['data_path']) as f:
            self.data = json.load(f)
        logging.info('Finished')
        
    def parse_composition(self, line):
        result = set()
        current_word = ""
        for c in line:
            if 'A' <= c <= 'Z':
                if len(current_word) > 0:
                    result.add(current_word)
                current_word = c
            elif (c == '-' or c == ' ') and len(current_word) > 0:
                result.add(current_word)
                current_word = ''
            else:
                current_word += c
        if len(current_word) > 0:
            result.add(current_word)
        return result
    
    def get_formula(self, index):
        try:
            struct = self.data[index]
        except Exception as e:
            logging.error(e)
            return "Invalid data request"
        else:
            counts = dict(Counter(struct['A']))
            components = [f'{el}{counts[el]}' if counts[el] > 1 else f'{el}' for el in counts]
            return ''.join(components)
    
    def get_cif(self, index, custom_path=None):
        try:
            struct = self.data[index]
            atoms = Atoms(coords=struct['X'],
                          lattice_mat=struct['L'],
                          elements=struct['A'],
                          cartesian=True)
            if custom_path is None:
                file_path = self.download_handler.new_download(str(index), extention='.cif')
            else:
                file_path = custom_path
            atoms.write_cif(filename=file_path, comment='here you go ^^', with_spg_info=False)
            logging.debug(file_path)
        except Exception as e:
            logging.error(f'Error while writing a cif: {e}')
            return None
        else:
            return file_path
        
    def get_cifs(self, name, indices, path):
        zip_path = None
        paths = []
        temp_dir_path = os.path.join(path, str(uuid4()))
        try:
            os.makedirs(temp_dir_path, exist_ok=True)
            for i in indices:
                file_path = os.path.join(temp_dir_path, f'{i}_{self.get_formula(i)}.cif')
                self.get_cif(i, custom_path=file_path)
                paths.append(file_path)
            if len(paths) == 0:
                raise Exception("Could not create cif files")
            zip_path = self.download_handler.new_download(name, extention='.zip')
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file_path in paths:
                    zipf.write(file_path, arcname=os.path.basename(file_path))  
        except Exception as e:
            zip_path = None
            logging.error(f'Error writing multiple cifs: {e}')
        finally:
            try:
                if os.path.exists(temp_dir_path):
                    shutil.rmtree(temp_dir_path)
            except:
                logging.error("Could not clean up files in DataHandler.get_cifs")
        return zip_path
    
    def complete_download(self, name):
        try:
            self.download_handler.complete_download(name)
        except:
            logging.error("Failed cleanup in complete_download")
  
    def find_structures(self, composition_line, all_required=True):
        composition = self.parse_composition(composition_line)
        result = []
        result_all_required = []
        for i, entry in enumerate(self.data):
            entry_comp = set(entry['A'])
            if entry_comp.issubset(composition):
                result.append(i)
            if entry_comp == composition:
                result_all_required.append(i)
        return result, result_all_required
    
    def get_basic_info(self, indices):
        struct_info = []
        try:
            structs = [self.data[i] for i in indices]
        except Exception as e:
            logging.error(e)
            logging.error(f'Error in DataHandler.get_basic_info: {e}')
            struct_info = []
        else:
            for i, struct in enumerate(structs):
                struct_info.append({
                    'description': self.get_formula(indices[i]),
                    'formation_e': struct['E'],
                    'e_above_hull': struct['E_above_hull'],
                    'sg': struct['sg'],
                    'index': indices[i],
                })
        return struct_info
    
    def distance_to_hull(self, indices, composition, energy):
        try:
            chosen = [self.data[i] for i in indices]
            distance, simplex_indices, decompose_to = distance_to_hull(chosen, composition, energy)
        except Exception as e:
            logging.error(f'Error in DataHandler.distance_to_hull: {e}')
            return None
        return distance, [chosen[i]['index'] for i in simplex_indices], decompose_to
        