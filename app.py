from flask import Flask, render_template, request, jsonify, Response, stream_with_context, send_from_directory
import moga_ml
import json
import time
import threading
from queue import Queue
import os

app = Flask(__name__, static_folder='static')

event_queue = Queue()
optimization_process = None
stop_flag = False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

@app.route('/api/run', methods=['POST'])
def run_algorithm():
    global stop_flag
    stop_flag = False
    
    data = request.json
    dataset_name = data.get('dataset', 'digits')
    ngen = int(data.get('ngen', 5))
    pop_size = int(data.get('pop_size', 20))
    
    while not event_queue.empty():
        event_queue.get()
    
    def optimization_callback(event_type, data):
        if stop_flag:
            return False
        
        event_data = {
            'type': event_type,
            'data': data,
            'timestamp': time.time()
        }
        try:
            event_queue.put(f"data: {json.dumps(event_data)}\n\n")
        except:
            pass
        return True
    
    def run_optimization_thread():
        global stop_flag
        try:
            moga_ml.run_optimization(
                dataset_name=dataset_name,
                ngen=ngen,
                pop_size=pop_size,
                callback=optimization_callback
            )
            if not stop_flag:
                event_queue.put(f"data: {json.dumps({'type': 'done', 'data': {}})}\n\n")
        except Exception as e:
            if not stop_flag:
                error_data = {'type': 'error', 'data': {'message': str(e)}}
                event_queue.put(f"data: {json.dumps(error_data)}\n\n")
    
    global optimization_process
    optimization_process = threading.Thread(target=run_optimization_thread)
    optimization_process.daemon = True
    optimization_process.start()
    
    return jsonify({'status': 'started', 'message': 'Optimisation lancée'})

@app.route('/api/stop', methods=['POST'])
def stop_algorithm():
    global stop_flag
    stop_flag = True
    event_queue.put(f"data: {json.dumps({'type': 'stop', 'data': {}})}\n\n")
    return jsonify({'status': 'stopped', 'message': 'Arrêt demandé'})

@app.route('/events')
def events():
    def generate():
        while True:
            if not event_queue.empty():
                yield event_queue.get()
            else:
                time.sleep(0.1)
    
    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
            'Connection': 'keep-alive'
        }
    )

if __name__ == '__main__':
    os.makedirs('static/css', exist_ok=True)
    os.makedirs('static/js', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    app.run(debug=True, threaded=True)