Hello! This is flask server allow you fit and use regression models based on decisions trees.

To run docker container you can use follow commands:

docker build -t semyonelistratov/mmp_prac_flask_server:tagname .
docker run -p 5000:5000 -v "$PWD/server/data:/root/server/data" rm -i semyonelistratov/mmp_prac_flask_server:tagname

or pull from dockerhub:

docker pull semyonelistratov/mmp_prac_flask_server:tagname
docker run -p 5000:5000 -i semyonelistratov/mmp_prac_flask_server:tagname

To fit model you should specify arguments (or use default), name your model and upload file contains 'target' coulumn. Then you can get predict for file (file shouldn't contain 'target') or validation score. Also you can see info about your model and download your model. Server can retain only one model.