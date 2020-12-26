import os
from ml_server import app

if __name__ == '__main__':
    s = app.run(host='0.0.0.0', port=5000, debug=True)
    folder = os.listdir("./")
    for item in folder:
        if item.endswith(".pkl"):
            os.remove(os.path.join("./", item))