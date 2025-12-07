from flask import Flask, render_template, request, jsonify
import moga_ml

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/run', methods=['POST'])
def run_algorithm():
    data = request.json
    dataset = data.get('dataset', 'digits')
    ngen = int(data.get('ngen', 5))
    pop_size = int(data.get('pop_size', 20))
    
    try:
        pareto_results, population_results = moga_ml.run_optimization(dataset, ngen, pop_size)
        return jsonify({
            'status': 'success',
            'pareto': pareto_results,
            'population': population_results
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
