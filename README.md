## Hello! This is flask server which allow you fit and use regression models based on decision trees.

To run docker container you can use follow commands:
```
docker build -t user_name/repo_name:tagname .
docker run -p 5000:5000 -v "$PWD/server/data:/root/server/data" rm -i user_name/repo_name:tagname
```
or pull from dockerhub (better way):
```
docker pull semyonelistratov/mmp_prac_flask_server:tagname
docker run -p 5000:5000 -i semyonelistratov/mmp_prac_flask_server:tagname
```

To get right ip-address maybe need to use `docker-machine ip` with `:5000` as a port num (e.g. for macOs)

To fit model you should specify arguments (or use default), name your model and upload file contains 'target' column. Then you can get predict for test data (file shouldn't contain 'target') or validation score. Also you can see info about your model and download your model. Server can retain only one model.
