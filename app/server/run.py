from flask import Flask
import os


# Current environment
env = os.environ.get("CURRENT_ENV", "prod")

# Flask app
app = Flask(__name__)

# Use config file defined in Dockerfile
app.config.from_envvar("APP_CONFIG_FILE")

import sys
sys.path.append(r'/root/miniconda/lib/python3.7/site-packages')

# Routes and controllers
from controllers import base
app.register_blueprint(base.blueprint)



# Run
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80, debug=True)
