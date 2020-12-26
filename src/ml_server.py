import os
import pickle
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from ensembles import RandomForestMSE, GradientBoostingMSE

from sklearn.metrics import mean_squared_error as mse

from collections import namedtuple
from flask_wtf import FlaskForm
from flask_bootstrap import Bootstrap
from flask import Flask, request, url_for, send_file
from flask import render_template, redirect, flash
from werkzeug.utils import secure_filename

from wtforms.validators import DataRequired, InputRequired, NumberRange, ValidationError
from wtforms import StringField, SubmitField, SelectField, TextAreaField, IntegerField, FileField, HiddenField


add_args = "Model parameters. Input format: arg1=val1, arg2=val2 ..."
file_fit = "CSV file for fitting. Else get fitted on our dataset model"
app = Flask(__name__, template_folder='html')
app.config['BOOTSTRAP_SERVE_LOCAL'] = True
app.config['SECRET_KEY'] = 'hello'
app.config['UPLOAD_FOLDER'] = '../../'
data_path = './../data'
Bootstrap(app)
messages = []

args_types = {"n_estimators":int, "learning_rate":float, "feature_sample_size":float, "max_depth":int, "criterion":str, 
              "splitter":str, "min_samples_split":int, "min_samples_leaf":int, "min_weight_fraction_leaf":float, "random_state":int, 
              "max_leaf_nodes":int, "min_impurity_decrease":float, "min_impurity_split":float, "ccp_alpha":float}

class Message:
    header = ''
    text = ''

class MainPageForm(FlaskForm):
    submit = SubmitField("Create model", validators=None)
    github_ref = SubmitField("Code", validators=None)

class OutputForm(FlaskForm):
    val_score = StringField("Validation RMSE")

class NewModelForm(FlaskForm):
    model = SelectField('Model', choices=[(1,"RandomForest"),(2,"GradientBoosting")], validators=[InputRequired()])
    name = StringField('Please name your model', validators=[DataRequired()])
    kwargs = TextAreaField(add_args, validators=None)
    file = FileField("Upload train file")
    submit = SubmitField("Fit", validators=None)
    info = SubmitField("Default arguments", validators=None)
    
class BackForm(FlaskForm):
    submit = SubmitField("Back to create model", validators=None)

class InfoForm(FlaskForm):
    #info = TextAreaField("Info about model")
    submit = SubmitField("Back", validators=None)
    load = SubmitField("Load verbose", validators=None)

class PredictModelForm(FlaskForm):
    file = FileField("File for predict.", validators=None)
    submit = SubmitField("Predict", validators=None)
    val = SubmitField("Validation", validators=None)
    info = SubmitField("Info about model", validators=None)
    new = SubmitField("Create new model", validators=None)
    load = SubmitField("Load your model", validators=None)




def kwargs_handler(kwargs):
    kwargs = kwargs.replace(" ", "")
    kwargs_list = kwargs.split(",")
    kwargs_dict = {}
    for x in kwargs_list:
        tmp = x.split("=")
        if args_types[tmp[0]] != str:
            tmp[1] = float(tmp[1])
            if int(tmp[1]) == float(tmp[1]):
                kwargs_dict[tmp[0]] = int(tmp[1])
            else:
                kwargs_dict[tmp[0]] = int(tmp[1])
        else:    
            kwargs_dict[tmp[0]] = tmp[1]
    return kwargs_dict

def fit(filename, name, model_type, kwargs, default_file=True):
    if len(kwargs)!= 0:
        args = kwargs_handler(kwargs)
    else:
        args = {}
    
    data = pd.read_csv(filename)
    os.remove(filename)
    target = data.pop("target")
    data = data[data.columns[data.dtypes != object]]
    X = data.to_numpy()
    y = target.to_numpy().ravel()
    if len(kwargs)!= 0:
        args = kwargs_handler(kwargs)
    else:
        args = {}
    if model_type == "1":
        model = RandomForestMSE(**args)
        model.fit(X, y)
    else:
        model = GradientBoostingMSE(**args)
        val = model.fit(X, y, X_val=X, y_val=y)
    model_type = "RandomForest" if model_type == "1" else "GradientBoosting"

    info = "Model info:\nmodel: '%s', name: '%s', parameters: %s \n" % (model_type, name, str(model.get_params()))
    info += "Dataset info:\n name: '%s', shape: (%d, %d); mean target: %.4f \n" % (filename, *data.shape, y.mean())
    if model_type == "GradientBoosting":
        fig = plt.figure(figsize=(10, 6))
        plt.plot(np.arange(10), val, lw=2, c="green")
        plt.title("Dependence of the loss function on iteration.")
        plt.xlabel("Iteration")
        plt.ylabel("RMSE")
        plt.grid()
        plt.savefig("./graph.png")
    info = info.replace('\n', '<br>')

    model_file_name = "./models/" + name + ".pkl"
    pickle.dump(model, open(model_file_name, "wb"))

    return model_file_name, info, model_type

def predict(file, model):
    file.save(file.filename)
    data = pd.read_csv(file.filename)
    if "target" in data.columns:
        data.pop("target")
    data = data[data.columns[data.dtypes != object]]
    data = data.to_numpy()
    os.remove(file.filename)
    rg = pickle.load(open(model, "rb"))
    res = rg.predict(data)
    res = pd.DataFrame(res, columns=["target"])
    res.to_csv("result.csv")
    return send_file("result.csv", as_attachment=True)

def validation(file, model, val_field):
    file.save(file.filename)
    data = pd.read_csv(file.filename)
    y = data.pop("target")
    data = data[data.columns[data.dtypes != object]]
    data = data.to_numpy()
    os.remove(file.filename)
    rg = pickle.load(open(model, "rb"))
    res = rg.predict(data)
    score = np.sqrt(mse(y, res))
    
    return score

def save_fit_file(file):
    file.save(file.filename)
    return file.filename

@app.route('/', methods=['GET', 'POST'])
def main_page():
    form = MainPageForm()
    if form.validate_on_submit():
        if form.github_ref.data:
            return redirect('https://github.com/ElistratovSemyon/ModelSQL')
        return redirect(url_for('new_model'))
    return render_template('main_page.html', form=form)

@app.route('/new_model', methods=['GET', 'POST'])
def new_model():
    ml_form = NewModelForm()
    ml_form.name.data = "NewModel"
    if ml_form.validate_on_submit():
        if ml_form.info.data:
            return redirect(url_for('default_args'))
        if ml_form.submit.data:
            #filename = secure_filename(ml_form.fileName.File.filename)
            #file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            #ml_form.fileName.File.save(file_path)
            if ml_form.file.data is None:
                model, info = fit(ml_form.name.data, ml_form.model.data, ml_form.kwargs.data)
            else:
                filename = save_fit_file(request.files[ml_form.file.name])
                model, info, model_type = fit(filename, ml_form.name.data, ml_form.model.data, ml_form.kwargs.data, default_file=False)
            return redirect(url_for('get_predict', model_name=model, model_info=info, model_type=model_type))
    return render_template('index.html', form=ml_form)

@app.route('/predict', methods=['GET', 'POST'])
def get_predict():
    ml_form = PredictModelForm()
    out = OutputForm()
    model_info = request.args.get("model_info")
    model_name = request.args.get("model_name")
    model_type = request.args.get("model_type")
    if ml_form.validate_on_submit():
        if ml_form.info.data:
            return redirect(url_for('model_info', info=model_info, model=model_name, model_type=model_type))
        if ml_form.submit.data:
            return predict(request.files[ml_form.file.name], model_name)
        if ml_form.new.data:
            return redirect(url_for('new_model'))
        if ml_form.load.data:
            return send_file(model_name, as_attachment=True)
        if ml_form.val.data:
            score = validation(request.files[ml_form.file.name], model_name, out.val_score) 
            flash("RMSE: %.4f" % score)
            return redirect(url_for('get_predict', model_name=model_name, model_info=model_info, model_type=model_type))
    return render_template('from_form.html', form=ml_form)

@app.route('/default_args', methods=['GET', 'POST'])
def default_args():
    button = BackForm()
    if button.validate_on_submit():
        return redirect(url_for('new_model'))
    return render_template('default_args.html', form=button)

@app.route('/model_info', methods=['GET', 'POST'])
def model_info():
    form = InfoForm()
    info = request.args.get("info")
    model_type = request.args.get("model_type")
    if form.validate_on_submit():
        return redirect(url_for('get_predict', model_info=info))
    if model_type == "1":
        return render_template('model_info.html', text=info, form=form)
    else:
        return render_template('model_info_with_verbose.html', text=info, form=form)

@app.route('/val_score', methods=['GET', 'POST'])
def val_page():
    form = BackForm()
    
    