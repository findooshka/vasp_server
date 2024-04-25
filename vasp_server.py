from flask import Blueprint, current_app, Flask, request, jsonify, render_template_string, send_file
import uuid
import os
from queue import PriorityQueue
from threading import Thread, Lock
import requests
from requests.exceptions import HTTPError, ConnectionError, Timeout
import logging
import time
from datetime import datetime, timedelta
from transitions import Machine
import zipfile
import json
import pandas as pd
import numpy as np
from pymatgen.core import Structure
from pymatgen.io.vasp import Poscar
from werkzeug.utils import secure_filename

def zip_folders(folder_paths, output_path, replace_with):
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for i, folder_path in enumerate(folder_paths):
            if not folder_path.endswith('/'):
                folder_path += '/'
            folder_name = os.path.basename(os.path.dirname(folder_path))
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(file_path, folder_path)
                    arcname = os.path.join(replace_with[i], relative_path)
                    zipf.write(file_path, arcname)

class Worker:
    states = ["idle", "busy", "paused", "remove"]
    
    def __init__(self, worker_id, address):
        self.machine = Machine(model=self, states=Worker.states, initial='idle')
        
        self.machine.add_transition(trigger='assign', source=['idle'], dest='busy')
        self.machine.add_transition(trigger='unpause', source='paused', dest='idle')
        self.machine.add_transition(trigger='_pause', source='idle', dest='paused')
        self.machine.add_transition(trigger='_reset', source=['busy'], dest='idle')
        self.machine.add_transition(trigger='finish', source=['busy'], dest='idle')
        self.machine.add_transition(trigger='remove', source=["idle", "busy", "paused"], dest='remove')
        
        self.machine.on_enter_idle('enter_idle')
        self.machine.on_enter_remove('enter_remove')
        
        self.pausing = False
        self.address = address
        self.worker_id = worker_id
        self.assigned_task_id = None
        self.unresponsive_time = None
    
    def _clear_task(self):
        if self.assigned_task_id is not None:
            task = tasks_db[self.assigned_task_id]
            task.reset()
            tasks_queue.put((task.priority, task))
    
    def reset(self):
        # assumes acquired lock
        self._clear_task()
        self._reset()
    
    def assign_task(self, task):
        # assumes acquired lock
        self.assigned_task_id = task.task_id
        self.assign()
    
    def enter_idle(self):
        self.assigned_task_id = None
        if self.pausing:
            self._pause()
            self.pausing = False
    
    def enter_remove(self):
        self._clear_task()
    
    def silent(self):
        if self.unresponsive_time is None:
            self.unresponsive_time = datetime.now()
        else:
            if datetime.now() - self.unresponsive_time > timedelta(seconds=90):
                self.remove()
    
    def post_request(self, url, **kwargs):
        url = self.address + url
        try:
            response = requests.post(url, **kwargs)
            self.responded()
            return response
        except requests.RequestException as e:
            self.silent()
            raise
    
    def pause(self):
        if self.state == 'idle':
            self.pausing = True
            self.enter_idle()
        else:
            self.pausing = True
    
    def responded(self):
        self.unresponsive_time = None
    
    def get_state_description(self):
        result = self.state
        if self.pausing:
            result += " (pausing)"
        if self.unresponsive_time is not None:
            result += " (unresponsive)"
        return result
    
    def to_dict(self):
        paused = "running"
        if self.state == "paused":
            paused = "paused"
        elif self.pausing:
            paused = "pausing"
        return {
            "worker_id": self.worker_id,
            "address": self.address,
            "status": self.get_state_description(),
            "paused": paused
        }
         
        
class Task:
    states = ["processing", "awaiting", "finished", "error", "cancelled"]
    
    def __init__(self, task_id, file_path, incar, kpoints, reference_energies, original_name, priority):
        self.machine = Machine(model=self, states=Task.states, initial='awaiting')
        
        self.machine.add_transition(trigger='send', source='awaiting', dest='processing')
        self.machine.add_transition(trigger='complete', source='processing', dest='finished')
        self.machine.add_transition(trigger='reset', source=['processing', 'error', 'cancelled'], dest='awaiting')
        self.machine.add_transition(trigger='cancel', source='awaiting', dest='cancelled')
        self.machine.add_transition(trigger='fail', source=['awaiting', 'processing'], dest='error')
        
        self.task_id = task_id
        self.file_path = file_path
        self.incar = incar
        self.kpoints = kpoints
        self.time = datetime.now()
        self.original_name = original_name
        self.reference_energies = reference_energies
        self.priority = priority
        self.result = "None"
        
    def __lt__(self, other):
        if not isinstance(other, Task):
            return NotImplemented
        return self.time < other.time  #False
    
    def to_dict(self):
        return {
            "task_id": self.task_id,
            "time": self.time.strftime('%Y-%m-%d %H:%M:%S'),
            "status": self.state,
            "result": self.result
        }
        
        
class TaskBatch:
    def __init__(self, batch_id, task_ids, author, priority):
        self.batch_id = batch_id
        self.tasks_ids = set(task_ids)
        self.time = datetime.now()
        self.author = author
        self.priority = priority
    
    def get_description(self):
        return [tasks_db[task_id].to_dict() for task_id in self.tasks_ids]
    
    def cancel(self):
        with workers_lock:
            for task_id in self.tasks_ids:
                if task_id in tasks_db and tasks_db[task_id].state in ('awaiting'):
                    tasks_db[task_id].cancel()
    
    def create_summary_file(self, dest):
        result = []
        with workers_lock:
            for task_id in self.tasks_ids:
                if not task_id in tasks_db:
                    continue
                task = tasks_db[task_id]
                if task.result is not None and ('e' and 'e_f' in task.result):
                    result.append([task.original_name, task.result['e'], task.result['e_f']])
        if len(result) > 0:
            df = pd.DataFrame(data=result, columns=['name', 'E', 'fE'])
            df.to_csv(dest, index=False)
    
    def to_dict(self):
        finished_tasks = 0
        with workers_lock:
            for task_id in self.tasks_ids:
                if task_id in tasks_db and tasks_db[task_id].state in ('finished', 'error', 'cancelled'):
                    finished_tasks += 1
        return {
            "batch_id": self.batch_id,
            "time": self.time.strftime('%Y-%m-%d %H:%M:%S'),
            "finished": finished_tasks,
            "total": len(self.tasks_ids),
            "author": self.author,
            "priority": self.priority
        }
        

priorities = {"low": 5, "normal": 3, "high": 1}
priority_names = {}
for priority, value in priorities.items():
    priority_names[value] = priority
        
vasp_mod = Blueprint('vasp_mod', __name__)
tasks_queue = PriorityQueue()
workers = {}
UPLOAD_FOLDER = 'vasp_orders'
BATCH_FOLDER = 'batch'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
tasks_db = {}
task_batches = {}
workers_lock = Lock()

def assign_task_to_worker(task, worker_id):
    # assumes acquired workers_lock
    worker_assigned = False
    worker = workers[worker_id]
    if worker.state == 'idle':
        try:
            with open(task.file_path, 'rb') as f:
                files = {'file': f}
                data = {'incar': task.incar,
                        'kpoints': task.kpoints,
                        'task_id': task.task_id,
                        'reference_energies': task.reference_energies}
                response = worker.post_request('/task', files=files, data=data)
            if response.status_code == 200:
                worker.assign_task(task)
                logging.debug(f"Task {task.task_id} assigned to worker {worker_id} at address {worker.address}")
                worker_assigned = True
                task.send()
            else:
                logging.error(f"Error assigning task to worker: {response.status_code}, {response.json()}")
        except (ConnectionError, Timeout) as e:
            if type(e) is Timeout:
                logging.error(f"The request to a worker {worker.address} timed out.")
            elif type(e) is ConnectionError:
                logging.error(f"Failed to connect to the worker {worker.address}.")
        except Exception as e:
            logging.error(f"Error assigning task to worker {worker_id}: {e}")
    return worker_assigned
    

def check_workers():
    while True:
        to_delete = []
        with workers_lock:
            for worker_id, worker in workers.items():
                try:
                    response = worker.post_request('/check')
                    paused = response.json()['paused']
                    idle = response.json()['idle'] == 'idle'
                except Exception as e:
                    logging.error(f'Checking up on a worker failed, id: {worker_id} exception: {e}')
                else:
                    if paused == 'paused' and worker.state != 'paused':
                        worker.pause()
                    elif paused == 'running' and worker.state == 'paused':
                        worker.unpause()
                    if worker.state == "busy" and idle:
                        # the worker has restarted on its side probably
                        worker.reset()
                if worker.state == 'remove':
                    to_delete.append(worker_id)
            for worker_id in to_delete:
                del workers[worker_id]
        time.sleep(15)
            

def assign_task():
    while True:
        priority, task = tasks_queue.get()
        worker_assigned = False
        #while not worker_assigned:
        if task.state != 'awaiting':
            break
        with workers_lock:
            for worker_id in workers:
                worker_assigned = assign_task_to_worker(task, worker_id)
                if worker_assigned:
                    break
        tasks_queue.task_done()
        if not worker_assigned:
            tasks_queue.put((priority, task))
            time.sleep(5)

Thread(target=assign_task, daemon=True).start()
Thread(target=check_workers, daemon=True).start()
        
@vasp_mod.route('/send_request', methods=['POST'])
def upload_file():
    if 'fileUpload' not in request.files:
        return 'No file part', 400
    files = request.files.getlist('fileUpload')
    incar = request.form['incar']
    kpoints = request.form['kpoints']
    reference_energies = request.form['reference_energies']
    priority = request.form['priority']
    priority_n = priorities[priority]
    
    task_ids = set()
    
    for file in files:
        task_id = str(uuid.uuid4())
        
        try:
            os.makedirs(os.path.join(UPLOAD_FOLDER, task_id), exist_ok=True)
            filename = os.path.join(UPLOAD_FOLDER, task_id, secure_filename(file.filename))
            poscar_path = os.path.join(UPLOAD_FOLDER, task_id, 'POSCAR')
            file.save(filename)
            structure = Structure.from_file(filename)
            poscar = Poscar(structure)
            poscar.write_file(poscar_path)
        except Exception as e:
            #return 'Could not save the files', 500
            continue
        
        with workers_lock:
            task = Task(task_id=task_id,
                        file_path=poscar_path,
                        incar=incar,
                        kpoints=kpoints,
                        reference_energies=reference_energies,
                        original_name=file.filename,
                        priority=priority)
            tasks_queue.put((priority_n, task))
            task_ids.add(task_id)
            tasks_db[task_id] = task
    if len(task_ids) > 0:
        batch_id = str(uuid.uuid4())
        task_batches[batch_id] = TaskBatch(batch_id, task_ids,
                                           author=request.remote_addr,
                                           priority=priority)
        return jsonify({"batch_id": batch_id}), 202
    return 'File not uploaded', 400

@vasp_mod.route('/')
def request_page():
    return render_template_string(open('templates/vasp_request.html').read())

@vasp_mod.route('/batches')
def batches_page():
    return render_template_string(open('templates/batches.html').read())

@vasp_mod.route('/list_batches', methods=['GET'])
def list_batches():
    batches_list = [batch for _, batch in task_batches.items()]
    batches_list = sorted(batches_list, key=lambda x: x.time)[::-1]
    batches_list = [batch.to_dict() for batch in batches_list]
    return jsonify(batches_list), 200

@vasp_mod.route('/tasks')
def tasks_page():
    return render_template_string(open('templates/tasks.html').read())

@vasp_mod.route('/list_tasks', methods=['GET'])
def list_tasks():
    with workers_lock:
        tasks_list = [task.to_dict() for task_id, task in tasks_db.items()]
    return jsonify(tasks_list), 200

@vasp_mod.route('/get_batch/<batch_id>')
def get_batch(batch_id):
    html_content = render_template_string(open('templates/batch.html').read(), batch_id=batch_id)
    return render_template_string(html_content)

@vasp_mod.route('/cancel_batch/<batch_id>', methods=['POST'])
def cancel_batch(batch_id):
    if batch_id in task_batches:
        task_batches[batch_id].cancel()
        return "Okay :C", 200
    return "Invalid batch id", 400 

@vasp_mod.route('/list_batch/<batch_id>', methods=['GET'])
def list_batch(batch_id):
    with workers_lock:
        if batch_id in task_batches:
            tasks_list = task_batches[batch_id].get_description()
            return jsonify(tasks_list), 200
        return "Invalid batch id", 400 

@vasp_mod.route('/download_batch', methods=['POST'])
def download():
    batch_id = request.form['batch_id']
    os.makedirs(BATCH_FOLDER, exist_ok=True)
    if not batch_id in task_batches:
        return "Invalid batch id", 400
    batch = task_batches[batch_id]
    paths = [os.path.join(UPLOAD_FOLDER, task_id) for task_id in batch.tasks_ids]
    with workers_lock:
        original_names = [tasks_db[task_id].original_name for task_id in batch.tasks_ids]
    summary_dir = os.path.join(BATCH_FOLDER, f'summary_{batch_id}')
    os.makedirs(summary_dir, exist_ok=True)
    batch.create_summary_file(os.path.join(summary_dir, 'summary.csv'))
    paths.append(summary_dir)
    original_names.append('summary')
    zip_path = os.path.join(BATCH_FOLDER, batch_id+'.zip')
    zip_folders(paths, zip_path, replace_with=original_names)
    return send_file(zip_path, as_attachment=True, download_name=f'{batch_id}.zip')
    
@vasp_mod.route('/workers')
def workers_page():
    return render_template_string(open('templates/workers.html').read())

@vasp_mod.route('/list_workers', methods=['GET'])
def list_workers():
    with workers_lock:
        workers_list = [worker.to_dict() for worker in workers.values()]
    return jsonify(workers_list), 200

@vasp_mod.route('/pause_worker/<worker_id>', methods=['POST'])
def pause_worker(worker_id):
    with workers_lock:
        if worker_id in workers:
            worker = workers[worker_id]
            try:
                response = worker.post_request('/control', json={'command': 'pause'})
            except Exception as e:
                logging.error(f'Error pausing the worker: {e}, worker_id: {worker_id}')
            else:
                worker.pause()
            return "Okay :C", 200
    return "Invalid worker id", 400 

@vasp_mod.route('/check_task/<task_id>')
def check_task(task_id):
    with workers_lock:
        if task_id in tasks_db.items():
            task = tasks_db[task_id]
            if task.state == 'finished':
                return task.result
            return task.state, 200
    return "Task not found", 404

@vasp_mod.route('/report_task/<task_id>', methods=['POST'])
def report_task(task_id):
    with workers_lock:
        try:
            workers[request.form.get('worker_id')].finish()
            if task_id in tasks_db:
                task = tasks_db[task_id]
                task.result = json.loads(request.form.get('result'))
                state = request.form.get('state')
                if state == 'error':
                    task.fail()
                else:
                    task.complete()
                for file in request.files:
                    file_path = os.path.join(UPLOAD_FOLDER, task_id, file)
                    request.files[file].save(file_path)
                return jsonify({"message": "Task completion acknowledged"}), 200
            return jsonify({"message": "Task not found"}), 404
        except Exception as e:
            logging.error(f"Received invalid task report: {e}")
            if task_id in tasks_db:
                tasks_db[task_id].fail()
    return jsonify({"message": "Something went wrong"}), 500

    
@vasp_mod.route('/volunteer', methods=['POST'])
def volunteer():
    data = request.json
    address = "http://" + request.remote_addr + ":8130"
    with workers_lock:
        for worker in workers.values():
            if worker.address == address:
                worker_id = worker.worker_id
                break
        else:
            worker_id = str(uuid.uuid4())
            workers[worker_id] = Worker(worker_id=worker_id, address=address)
            logging.info(f"New worker, id: {worker_id}, address: {address}")
    return jsonify({"message": "Worker acknowledged", "assigned_id": worker_id}), 200




