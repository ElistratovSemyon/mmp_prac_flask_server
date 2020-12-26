from ml_server import app

if __name__ == '__main__':
    s = app.run(host='0.0.0.0', port=5000, debug=True)
    print(s)
