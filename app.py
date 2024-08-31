from flask import Flask, render_template, redirect, url_for, flash, request, Markup
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
import secrets
from flask_login import (LoginManager, login_user, login_required,
                             logout_user, current_user, UserMixin)
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import (DataRequired, Length, EqualTo, Email,
                                ValidationError)
import logging
from werkzeug.security import generate_password_hash, check_password_hash
import requests
import pandas as pd
from keras.models import load_model
import numpy as np
import joblib
import warnings
import datetime
import plotly.express as px
import folium
from folium.plugins import HeatMap

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

app = Flask(__name__)
app.config['SECRET_KEY'] = secrets.token_hex(16)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///weather.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
migrate = Migrate(app, db)
login_manager = LoginManager(app)
login_manager.login_view = 'admin_login'  # Separate view for admin login
api_key = '6476247d6af7e1d43d41e7b1dd4779d6'

# Load your pre-trained models and scalers
model_48h = load_model('hourly_cnn_lstm_weather_model.h5')
model_7d = load_model('daily_cnn_lstm_weather_model_best.h5')
scaler_48h = joblib.load('hourly_scaler.pkl')
scaler_7d = joblib.load('daily_scaler.pkl')

API_KEYS = [
    "G82543YHG2RCV4T2NKB7DADZF",
    "D8ATGRGWAUQT3RF8HQ45QXU56",
    "R3TKYVPDGHETCD6YPRMCE5CYU",
    "5ZH42W7ER6FFD2FZ9UQLXV3DY",
    "TQQLX778KF3JTAXDDXXF7SKJC"
]

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))

    def __init__(self, username, email, password):
        self.username = username
        self.email = email
        self.set_password(password)  # Set password when creating a new user

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class AdminRegistrationForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired(), Length(min=2, max=20)])
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired()])
    confirm_password = PasswordField('Confirm Password', validators=[DataRequired(), EqualTo('password')])
    submit = SubmitField('Register')

    def validate_username(self, username):
        user = User.query.filter_by(username=username.data).first()
        if user:
            raise ValidationError('That username is taken. Please choose a different one.')

    def validate_email(self, email):
        user = User.query.filter_by(email=email.data).first()
        if user:
            raise ValidationError('That email is already in use. Please choose a different one.')


class AdminLoginForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])
    submit = SubmitField('Login')


class EditProfileForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired(), Length(min=2, max=20)])
    email = StringField('Email', validators=[DataRequired(), Email()])
    old_password = PasswordField('Current Password', validators=[DataRequired()])
    new_password = PasswordField('New Password', validators=[DataRequired()])
    submit = SubmitField('Update')



@app.template_filter('format_number')
def format_number(value):
    return f"{value:.2f}"



def download_weather_data(city_name, api_key):
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city_name}&units=metric&appid={api_key}"
    response = requests.get(url)
    if response.status_code != 200:
        logging.warning(f"Failed to fetch weather data. Status code: {response.status_code}")
        return None
    weather_data = response.json()
    logging.info(f"Weather data: {weather_data}")
    return weather_data if 'main' in weather_data else None



def create_geo_heatmap(predicted_48h, hourly_df, selected_hour):
    melaka_coordinates = [2.1896, 102.2501]
    m = folium.Map(location=melaka_coordinates, zoom_start=12)

    precip_value = float(predicted_48h[selected_hour, 2])
    color = get_precipitation_color(precip_value)

    folium.Circle(
        location=melaka_coordinates,
        radius=500,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.7,
        tooltip=f'Precip: {precip_value} mm<br>{interpret_precipitation(precip_value)}'
    ).add_to(m)

    # Add Legend
    legend_html = '''
    <div style="position: fixed; 
            bottom: 5px; left: 5px; width: 200px; height: 150px; 
            background-color: white; z-index:9999; font-size:14px;
            border:2px solid grey; border-radius:5px; padding: 10px;">
            <b>Legend</b><br>
            <span style="background: gray; width: 20px; height: 20px; display: inline-block; margin-right: 5px;"></span> No Rain<br>
            <span style="background: lightblue; width: 20px; height: 20px; display: inline-block; margin-right: 5px;"></span> Light Rain<br>
            <span style="background: blue; width: 20px; height: 20px; display: inline-block; margin-right: 5px;"></span> Moderate Rain<br>
            <span style="background: darkblue; width: 20px; height: 20px; display: inline-block; margin-right: 5px;"></span> Heavy Rain<br>
            </div>
            '''

    m.get_root().html.add_child(folium.Element(legend_html))

    return m._repr_html_()



def interpret_precipitation(value):
    if value < 0.05:
        return "No Rain"
    elif 0.05 < value < 0.3:
        return "Light Rain"
    elif 0.3 <= value < 4.0:
        return "Moderate Rain"
    else:
        return "Heavy Rain"



def get_precipitation_color(value):
    if value < 0.05:
        return 'gray'
    elif 0.05 < value < 0.3:
        return 'lightblue'
    elif 0.3 <= value < 4.0:
        return 'blue'
    else:
        return 'darkblue'



def get_time_of_day(hour):
    if hour == 0 or hour == 24:  # This is 12 AM, midnight
        return 'night'
    elif hour == 12:  # This is 12 PM, noon
        return 'afternoon'
    elif 5 <= hour < 12:  # From 1 AM to 11 AM
        return 'morning'
    elif 13 <= hour < 18:  # From 1 PM to 5 PM
        return 'afternoon'
    elif 18 <= hour < 22:  # From 6 PM to 11 PM
        return 'evening'
    else:
        return 'night'

    


def fetch_weather_data(api_keys, period='48h'):
    today = datetime.date.today()
    day_before = today - datetime.timedelta(days=1)
    today_str = today.strftime('%Y-%m-%d')
    day_before_str = day_before.strftime('%Y-%m-%d')

    BASE_URL = (f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/melaka/{day_before_str}/{today_str}"
                if period == '48h' else
                "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/melaka/last7days")

    PARAMS = {
        'unitGroup': 'metric',
        'key': api_keys,
        'include': 'hours' if period == '48h' else 'days'
    }

    response = requests.get(BASE_URL, params=PARAMS)
    if response.status_code != 200:
        return None
    return response.json()



def preprocess_and_predict_48h(weather_data):
    df = pd.DataFrame(weather_data['days'])
    hourly_data = pd.DataFrame(df['hours'].tolist()).stack().reset_index(level=1, drop=True)
    hourly_df = pd.DataFrame(hourly_data.tolist())
    hourly_df.index = pd.date_range(start=df.index[0], periods=len(hourly_df), freq='H')
    hourly_df.rename(columns={'temp': 'T2M', 'windspeed': 'WS10M', 'precip': 'PRECTOTCORR', 'humidity': 'RH2M'}, inplace=True)
    hourly_df['hour_sin'] = np.sin(2 * np.pi * hourly_df.index.hour / 24)
    hourly_df['hour_cos'] = np.cos(2 * np.pi * hourly_df.index.hour / 24)
    features = ['T2M', 'WS10M', 'PRECTOTCORR', 'RH2M', 'hour_sin', 'hour_cos']

    if len(hourly_df) > 48:
        hourly_df = hourly_df.iloc[-48:]

    scaled_input = scaler_48h.transform(hourly_df[features])
    scaled_input = scaled_input.reshape((1, 48, len(features)))
    predicted = []

    for i in range(48):
        pred = model_48h.predict(scaled_input)
        predicted.append(pred[0])
        pred_with_time = np.concatenate([pred, np.array([[np.sin(2 * np.pi * (48+i) / 24), np.cos(2 * np.pi * (48+i) / 24)]])], axis=1)
        pred_with_time = scaler_48h.transform(pred_with_time)
        scaled_input = np.append(scaled_input[:, 1:, :], np.expand_dims(pred_with_time, axis=0), axis=1)

    return np.array(predicted), hourly_df



def preprocess_and_predict_7d(weather_data):
    df = pd.DataFrame(weather_data['days'])
    df['date'] = pd.to_datetime(df['datetime'])
    df.set_index('date', inplace=True)
    df['YEAR'] = df.index.year
    df['MO'] = df.index.month
    df['DY'] = df.index.day
    df['sin_day_year'] = np.sin(2 * np.pi * df.index.dayofyear / 365)
    df['cos_day_year'] = np.cos(2 * np.pi * df.index.dayofyear / 365)
    df.rename(columns={'tempmax': 'T2M_MAX', 'tempmin': 'T2M_MIN', 'temp': 'T2M', 'windspeed': 'WS10M_MAX', 'winddir': 'WD10M', 'precip': 'PRECTOTCORR', 'humidity': 'RH2M'}, inplace=True)
    features = ['YEAR', 'MO', 'DY', 'sin_day_year', 'cos_day_year', 'T2M_MAX', 'T2M_MIN', 'T2M', 'WS10M_MAX', 'WD10M', 'PRECTOTCORR', 'RH2M']

    if len(df) > 7:
        df = df.iloc[-7:]

    scaled_input = scaler_7d.transform(df[features])
    scaled_input = scaled_input.reshape((1, 7, len(features)))
    predicted = []

    for i in range(7):
        pred = model_7d.predict(scaled_input)
        predicted.append(pred[0])
        pred_with_time = np.concatenate([pred, np.array([[np.sin(2 * np.pi * (7+i) / 24), np.cos(2 * np.pi * (7+i) / 24)]])], axis=1)
        current_date = df.index[-1] + pd.DateOffset(days=i+1)
        pred_with_time = np.concatenate([[[current_date.year, current_date.month, current_date.day]], pred_with_time], axis=1)
        pred_with_time = scaler_7d.transform(pred_with_time)
        scaled_input = np.append(scaled_input[:, 1:, :], np.expand_dims(pred_with_time, axis=0), axis=1)

    return np.array(predicted), df



@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))



@app.route('/admin_login', methods=['GET', 'POST'])
def admin_login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))  # Redirect if already logged in
    form = AdminLoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user and user.check_password(form.password.data):
            login_user(user)
            flash('Login successful!', 'success')
            return redirect(url_for('admin_dashboard'))  # Or your admin dashboard
        else:
            flash('Login unsuccessful. Please check username and password.', 'danger')
    return render_template('admin_login.html', form=form)



@app.route('/admin_register', methods=['GET', 'POST'])
def admin_register():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))  # Redirect if already logged in
    form = AdminRegistrationForm()
    if form.validate_on_submit():
        user = User(username=form.username.data, email=form.email.data, password=form.password.data)
        db.session.add(user)
        db.session.commit()
        flash('Admin account created! You can now log in.', 'success')
        return redirect(url_for('admin_login'))
    return render_template('admin_register.html', form=form)



@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('dashboard'))



@app.route('/edit_profile', methods=['GET', 'POST'])
@login_required
def edit_profile():
    form = EditProfileForm()
    if form.validate_on_submit():
        if current_user.check_password(form.old_password.data):
            current_user.username = form.username.data
            current_user.email = form.email.data
            current_user.set_password(form.new_password.data)  # Update password
            db.session.commit()
            flash('Your profile has been updated!', 'success')
            return redirect(url_for('admin_dashboard'))
        else:
            flash('Incorrect current password. Please try again.', 'danger')
    elif request.method == 'GET':
        form.username.data = current_user.username
        form.email.data = current_user.email
    return render_template('edit_profile.html', form=form)



@app.route('/')
@app.route('/dashboard')
def dashboard():
    current_weather_data = download_weather_data('Melaka', api_key)
    weather = None
    if current_weather_data:
        weather = {
            'temperature': current_weather_data['main']['temp'],
            'humidity': current_weather_data['main']['humidity'],
            'feels_like': current_weather_data['main']['feels_like'],
            'description': current_weather_data['weather'][0]['description']
        }
    return render_template('dashboard.html', weather=weather)



@app.route('/realtime_weather')
def realtime_weather():
    logging.basicConfig(level=logging.INFO)
    from datetime import datetime, timedelta

    current_time = datetime.now().strftime('%I:%M %p')
    current_hour = datetime.now().hour
    now = datetime.now()

    api_keys = API_KEYS[3]  # Use the first API key or rotate if needed
    weather_data_48h = fetch_weather_data(api_keys, period='48h')
    weather_data_7d = fetch_weather_data(api_keys, period='7d')
    city_name = "Melaka"
    
    if weather_data_48h:
        predicted_48h, hourly_df = preprocess_and_predict_48h(weather_data_48h)
        predicted_48h = predicted_48h.tolist()
    else:
        predicted_48h = []
        hourly_df = pd.DataFrame()

    if weather_data_7d:
        predicted_7d, daily_df = preprocess_and_predict_7d(weather_data_7d)
        predicted_7d = predicted_7d.tolist()
    else:
        predicted_7d = []
        daily_df = pd.DataFrame()

    # For 48-hour predictions
    start_index = current_hour % 24 
    for i, prediction in enumerate(predicted_48h[start_index:], start=start_index):
        hour_of_day = i % 24
        hour = hour_of_day % 12 or 12
        am_pm = 'AM' if hour_of_day < 12 else 'PM'
        prediction.insert(0, f'{hour}:00 {am_pm}')
        prediction.insert(1, (now + timedelta(hours=i - start_index)).strftime('%Y-%m-%d'))

    # For 7-day predictions
    for i, prediction in enumerate(predicted_7d):
        prediction.insert(0, (now + timedelta(days=i+1)).strftime('%Y-%m-%d'))

    current_weather_data = download_weather_data('Melaka', api_key)
    weather = None
    if current_weather_data:
        weather = {
            'city': city_name,
            'temperature': current_weather_data['main']['temp'],
            'humidity': current_weather_data['main']['humidity'],
            'feels_like': current_weather_data['main']['feels_like'],
            'description': current_weather_data['weather'][0]['description'],
            'wind_speed': current_weather_data['wind']['speed'],
            'precipitation': current_weather_data.get('rain', {}).get('1h', 0),
            'current_time': datetime.now().strftime('%Y-%m-%d %H:%M'),
            'sunrise': datetime.fromtimestamp(current_weather_data['sys']['sunrise']).strftime('%H:%M'),
            'sunset': datetime.fromtimestamp(current_weather_data['sys']['sunset']).strftime('%H:%M')
        }
    else:
        logging.warning("Failed to retrieve current weather data")

    return render_template('realtime_weather.html', 
                           weather=weather, 
                           predicted_48h=predicted_48h, 
                           hourly_df=hourly_df, predicted_7d=predicted_7d, 
                           daily_df=daily_df, current_time=current_time, 
                           current_hour=current_hour,
                           now=now,
                           timedelta=timedelta,
                           get_time_of_day=get_time_of_day)



@app.route('/weather_map', methods=['GET', 'POST'])
def weather_map():
    api_keys = API_KEYS[3]
    weather_data_48h = fetch_weather_data(api_keys, period='48h')
    selected_hour = int(request.form.get('hour', 0))
    if weather_data_48h is not None:
        predicted_48h, hourly_df = preprocess_and_predict_48h(weather_data_48h)
        geo_heatmap_html = create_geo_heatmap(predicted_48h, hourly_df, selected_hour)
    else:
        geo_heatmap_html = None
        flash('Failed to fetch weather data.', 'danger')

    return render_template('weather_map.html', geo_heatmap_html=geo_heatmap_html, num_hours=48, selected_hour=selected_hour)



@app.route('/past-weather')
def past_weather():
    api_keys = API_KEYS[3]  # Use the first API key or rotate if needed
    weather_data_7d = fetch_weather_data(api_keys, period='7d')

    past_weather_data = []
    if weather_data_7d:
        for day in weather_data_7d['days']:
            past_weather_data.append({
                'date': day['datetime'],
                'temp_max': day.get('tempmax', 'N/A'),
                'temp_min': day.get('tempmin', 'N/A'),
                'temp_avg': day.get('temp', 'N/A'),
                'humidity': day.get('humidity', 'N/A'),
                'precip': day.get('precip', 'N/A'),
                'windspeed': day.get('windspeed', 'N/A')
            })

    return render_template('past_weather.html', past_weather_data=past_weather_data)



@app.route('/admin_dashboard')
@login_required
def admin_dashboard():
    from datetime import datetime, timedelta

    api_keys = API_KEYS[3]
    current_weather_data = download_weather_data('Melaka', api_key) 
    weather = None
    if current_weather_data:
        weather = {
            'temperature': current_weather_data['main']['temp'],
            'humidity': current_weather_data['main']['humidity'],
            'feels_like': current_weather_data['main']['feels_like'],
            'description': current_weather_data['weather'][0]['description']
        }

    # Reading hourly data
    hourly_data = pd.read_csv('melaka.csv')
    hourly_data['datetime'] = pd.to_datetime(hourly_data[['YEAR', 'MO', 'DY', 'HR']].astype(str).agg('-'.join, axis=1), format='%Y-%m-%d-%H')
    
    # Reading daily data
    daily_data = pd.read_csv('melaka_daily.csv')
    daily_data['datetime'] = pd.to_datetime(daily_data[['YEAR', 'MO', 'DY']].astype(str).agg('-'.join, axis=1), format='%Y-%m-%d')

    # Fetch and predict 48-hour weather data
    weather_data_48h = fetch_weather_data(api_keys, period='48h')
    predicted_48h, hourly_df_48h = preprocess_and_predict_48h(weather_data_48h)

    # Fetch and predict 7-day weather data
    weather_data_7d = fetch_weather_data(api_keys, period='7d')
    predicted_7d, daily_df_7d = preprocess_and_predict_7d(weather_data_7d)
    
    # Get tomorrow's date
    tomorrow = datetime.now() + timedelta(days=1)

    # Get today's date
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    
    # Format data for Plotly
    hourly_labels = (pd.date_range(start=today, periods=len(hourly_df_48h), freq='H')).strftime('%Y-%m-%d %H:%M').tolist()
    daily_labels = (pd.date_range(start=tomorrow, periods=len(daily_df_7d), freq='D')).strftime('%Y-%m-%d').tolist()

    hourly_t2m = predicted_48h[:, 0].tolist()

    daily_t2m_min = predicted_7d[:, 1].tolist()
    daily_t2m_max = predicted_7d[:, 0].tolist()

    # Add data for Bar Chart (Hourly Wind Speed)
    hourly_wind_speed_data = predicted_48h[:, 1].tolist()
    
    # Add data for Wind Rose Diagram
    wind_rose_data = [
    {'WS10M_MAX': value, 'WD10M': direction} 
    for value, direction in zip(predicted_7d[:, 3].tolist(), predicted_7d[:, 4].tolist())
    ]

    # Add data for Column Chart (Daily Precipitation)
    daily_precipitation_data = predicted_7d[:, 5].tolist()
    
    # Create Heatmap data for monthly precipitation intensity
    monthly_precipitation_data = daily_data.groupby(daily_data['datetime'].dt.month)['PRECTOTCORR'].mean().tolist()

    # Add data for Column Chart (Daily Precipitation)
    hourly_humidity_data = predicted_48h[:, 3].tolist()
    
    # Create Heatmap data for monthly precipitation intensity
    monthly_humidity_data = daily_data.groupby(daily_data['datetime'].dt.month)['RH2M'].mean().tolist()
    
    return render_template('admin_dashboard.html', username=current_user.username,
                       hourly_labels=hourly_labels,
                       hourly_t2m=hourly_t2m,
                       daily_labels=daily_labels,
                       daily_t2m_min=daily_t2m_min,
                       daily_t2m_max=daily_t2m_max,
                       hourly_wind_speed_data=hourly_wind_speed_data,
                       daily_precipitation_data=daily_precipitation_data,
                       monthly_precipitation_data=monthly_precipitation_data,
                       wind_rose_data=wind_rose_data,
                       hourly_humidity_data=hourly_humidity_data,
                       monthly_humidity_data=monthly_humidity_data,
                       weather=weather)



if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
