from flask import Flask, jsonify, request, render_template_string, send_file
import requests
from data import DataHandler
from downloads import DownloadHandler
from vasp_server import vasp_mod
import yaml
import logging
import os

to_send_dir = 'to_send'
app = Flask(__name__)
app.register_blueprint(vasp_mod, url_prefix='/vasp')
download_handler = DownloadHandler(to_send_dir)
data_handler = DataHandler(download_handler)


@app.route('/')
def home():
    return render_template_string(open('templates/index.html').read())

@app.route('/files/<path:filename>')
def custom_static(filename):
    return send_from_directory('static', filename)

@app.route('/submit', methods=['POST'])
def submit():
    system = request.json['data']
    
    indices, indices_all_required = data_handler.find_structures(system)
    struct_list = data_handler.get_basic_info(indices_all_required)
    composition = list(data_handler.parse_composition(system))
    data = {
        'struct_list': struct_list,
        'composition': composition,
        'struct_indices': indices,
        'struct_indices_exact_match': indices_all_required,
    }
    
    if data is None:
        return jsonify([])
        
    for entry in struct_list:
        entry['infoUrl'] = f'/data/{entry["index"]}'
        entry['downloadUrl'] = f'/download/{entry["index"]}'
        entry['fileName'] = f'{entry["index"]}_{data_handler.get_formula(entry["index"])}.cif'
    
    return jsonify(data)

@app.route('/get-hull', methods=['POST'])
def get_hull():
    composition = request.json['composition']
    indices = request.json['indices']
    energy = request.json['energy']
    
    distance_to_hull, simplex, decompose_to = data_handler.distance_to_hull(indices, composition, energy)
    simplex_info = data_handler.get_basic_info(simplex)
    simplex_data = []
    for i, struct in enumerate(simplex_info):
        if abs(decompose_to[i]) > 1e-8:
            simplex_data.append({
                'description': struct['description'],
                'e': str(struct['formation_e']),
                'decompose_proportion': decompose_to[i]
            })
    
    return jsonify({'distance': distance_to_hull, 'simplexData': simplex_data})

@app.route('/download/<item_id>', methods=['POST'])
def download(item_id):
    cif_file = data_handler.get_cif(int(item_id))
    if cif_file is not None:
        return send_file(cif_file, as_attachment=True, download_name=f'{item_id}_{data_handler.get_formula(int(item_id))}.cif')

@app.route('/download_zip', methods=['POST'])
def download_all():
    indices = request.json['data']['indices']
    name = request.json['data']['name']
    
    zip_path = data_handler.get_cifs(name, indices, 'to_send/')
    if zip_path is not None:
        return send_file(zip_path, as_attachment=True, download_name=f'{name}.zip')

@app.route('/confirm_download', methods=['POST'])
def confirm_download():
    name = request.json['name']
    data_handler.complete_download(name)
    return jsonify(['ok'])

def set_up_logger(config):
    logger = logging.getLogger()
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
    )
    handler.setFormatter(formatter)
    if config['debug']:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
        logging.getLogger("requests").setLevel(logging.WARNING)
        logging.getLogger("transitions").setLevel(logging.WARNING)
        
def initialize():
    with open('config.yml', 'r') as f:
        config = yaml.safe_load(f)
    set_up_logger(config)
    data_handler.load_data(config)
    app.config['config'] = config

if __name__ == '__main__':
    initialize()
    app.run(host="0.0.0.0",
            debug=app.config['config']['debug'],
            port=app.config['config']['server']['port'])