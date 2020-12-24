import os
import pickle
import pandas as pd
import numpy as np
import re
from ensembles import RandomForestMSE, GradientBoostingMSE

from collections import namedtuple
from flask_wtf import FlaskForm
from flask_bootstrap import Bootstrap
from flask import Flask, request, url_for, send_file
from flask import render_template, redirect
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


class Message:
    header = ''
    text = ''

class MainPageForm(FlaskForm):
    submit = SubmitField("Create model", validators=None)
    github_ref = SubmitField("Code", validators=None)

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

class PredictModelForm(FlaskForm):
    file = FileField("File for predict.", validators=None)
    submit = SubmitField("Predict", validators=None)
    info = SubmitField("Info about model", validators=None)
    new = SubmitField("Create new model", validators=None)
    load = SubmitField("Load your model", validators=None)


def kwargs_handler(kwargs):
    kwargs = kwargs.replace(" ", "")
    kwargs_list = kwargs.split(",")
    kwargs_dict = dict([(x.split("=")[0], int(x.split("=")[1])) for x in kwargs_list])
    return kwargs_dict

def fit(name, model_type, kwargs, default_file=True):
    if len(kwargs)!= 0:
        args = kwargs_handler(kwargs)
    else:
        args = {}
    #if default_file:
    #    s = "default_fit.csv"
    #    data = pd.read_csv(s)
    #else:
    #    s = "fit_file.csv"
    #    data = pd.read_csv(s)
    #    os.remove(s)
    if 1:
        s = "fit_file.csv"
        data = pd.read_csv(s)
    #    os.remove(s)
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
    else:
        model = GradientBoostingMSE(**args)
    model.fit(X, y)
    info = "Model info:\nmodel: %s, name: %s, parameters: %s \n" % (model_type, name, str(model.get_params()))
    info += "Dataset info:\name: %s, shape: (%d, %d); mean target: %.4f" % ("data", *data.shape, y.mean())
    model_file_name = "./models/" + name + ".pkl"
    pickle.dump(model, open(model_file_name, "wb"))
    return model_file_name, info

def predict(file, model):
    s = "predict_file.csv"
    file.save(s)
    data = pd.read_csv(s)
    data.pop("target")
    data.pop("date")
    data = data.to_numpy()
    os.remove(s)
    rg = pickle.load(open(model, "rb"))
    res = rg.predict(data)
    res = pd.DataFrame(res, columns=["target"])
    res.to_csv("result.csv")
    return send_file("result.csv", as_attachment=True)

def save_fit_file(file):
    s = "fit_file.csv"
    file.save(s)

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
                print("\n\n\n\n_________________\n\n\n\n")
                save_fit_file(request.files[ml_form.file.name])
                model, info = fit(ml_form.name.data, ml_form.model.data, ml_form.kwargs.data, default_file=False)
            return redirect(url_for('get_predict', model=model, model_info=info))
    return render_template('index.html', form=ml_form)

@app.route('/predict', methods=['GET', 'POST'])
def get_predict():
    ml_form = PredictModelForm()
    model = request.args.get("model")
    model_info = request.args.get("model_info")
    model_name = request.args.get("model_name")
    if ml_form.validate_on_submit():
        if ml_form.info.data:
            return redirect(url_for('model_info', info=model_info))
        if ml_form.submit.data:
            return predict(request.files[ml_form.file.name], model)
        if ml_form.new.data:
            return redirect(url_for('new_model'))
        if ml_form.load.data:
            return send_file("./models/model.pkl", as_attachment=True)
        return redirect(url_for('get_predict'))
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
    text = request.args.get("info")
    if form.validate_on_submit():
        return redirect(url_for('get_predict'))    
    return render_template('model_info.html', text=text, form=form)