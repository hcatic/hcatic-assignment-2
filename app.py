import numpy as np
from flask import Flask, request, jsonify, render_template
from KMeans import KMeans

app = Flask(__name__)

def generate_random_data(n_points=100):
    X = np.random.uniform(low=-3, high=3, size=(n_points, 2))
    return X.tolist()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/step_kmeans', methods=['POST'])
def step_kmeans():
    data = np.array(request.json['data'])
    k = int(request.json['k'])
    method = request.json['method']
    manual_centroids = request.json.get('centroids', None)

    kmeans = KMeans(data, k)

    if method == 'manual' and manual_centroids:
        current_centroids = np.array(manual_centroids)
    else:
        current_centroids = request.json.get('centroids', None)

    new_centroids, assignment = kmeans.lloyds_step(current_centroids)

    return jsonify(centroids=new_centroids.tolist(), assignment=assignment)

@app.route('/api/run_to_convergence', methods=['POST'])
def run_to_convergence():
    data = np.array(request.json['data'])
    k = int(request.json['k'])
    method = request.json['method'] 
    manual_centroids = request.json.get('centroids', None)

    kmeans = KMeans(data, k)

    if method == 'manual' and manual_centroids:
        current_centroids = np.array(manual_centroids)
    else:
        current_centroids = request.json.get('centroids', None)

    final_centroids, assignment = kmeans.lloyds_converge(current_centroids)

    return jsonify(centroids=final_centroids.tolist(), assignment=assignment)

@app.route('/api/generate_data', methods=['GET'])
def generate_data():
    data = generate_random_data()
    return jsonify(data=data)

if __name__ == '__main__':
    app.run(debug=True)