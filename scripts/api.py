import flask
from flask import jsonify, make_response
import pandas as pd
import random
import numpy as np
from datetime import datetime

# Set the seed
ts = datetime.now().timestamp()
ts = int(ts)
print(ts)
np.random.seed(ts)

# Generate random weights
raw_weights = np.random.randint(low=1, high=100, size=4)
weights = (raw_weights/np.sum(raw_weights))*100
weights = weights.astype(int)
weights = weights.tolist()

# Generate data
df_data = {'Instrument': ['APPL', 'F', 'CVX', 'TSLA'], 'Weight': weights}
df_data = pd.DataFrame(df_data)
results = df_data.to_json(orient='records')
print(results)

"""
def hello_world(request):
    request_json = request.get_json(silent=True)
    request_args = request.args
    response = make_response(
        results,
        #jsonify(results),
        200
    )
    response.headers["Content-Type"] = "application/json"
    return response
"""