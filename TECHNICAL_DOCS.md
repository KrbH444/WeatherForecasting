# 🔧 Technical Documentation - Weather Forecasting System

## 📋 Overview

This document provides detailed technical information about the Weather Forecasting System architecture, codebase, and development guidelines.

## 🏗️ System Architecture

### High-Level Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Client    │    │   Flask Server  │    │   ML Models     │
│                 │◄──►│                 │◄──►│                 │
│ - Dashboard     │    │ - API Endpoints │    │ - CNN-LSTM      │
│ - Real-time     │    │ - Auth System   │    │ - Scaler Models │
│ - Maps          │    │ - Database      │    │ - Predictions   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌─────────────────┐
                       │ External APIs   │
                       │                 │
                       │ - OpenWeather   │
                       │ - Multiple Keys │
                       └─────────────────┘
```

### Technology Stack Details

#### Backend Framework
- **Flask 2.3.3**: Lightweight web framework
- **SQLAlchemy 3.0.5**: Database ORM
- **Flask-Login 0.6.3**: User authentication
- **Flask-WTF 1.1.1**: Form handling and CSRF protection
- **Flask-Migrate 4.0.5**: Database migrations

#### Machine Learning
- **TensorFlow 2.13.0**: Deep learning framework
- **Keras 2.13.1**: High-level neural network API
- **Scikit-learn 1.3.0**: Machine learning utilities
- **Pandas 2.0.3**: Data manipulation
- **NumPy 1.24.3**: Numerical computing
- **Joblib 1.3.2**: Model persistence

#### Data Visualization
- **Plotly 5.16.1**: Interactive charts
- **Folium 0.14.0**: Geographic mapping
- **Matplotlib 3.7.2**: Static plotting

## 📁 Code Structure

### Core Application (`app.py`)

#### Main Components

```python
# Application Configuration
app = Flask(__name__)
app.config['SECRET_KEY'] = secrets.token_hex(16)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///weather.db'

# Database and Authentication
db = SQLAlchemy(app)
migrate = Migrate(app, db)
login_manager = LoginManager(app)

# Model Loading
model_48h = load_model('hourly_cnn_lstm_weather_model.h5')
model_7d = load_model('daily_cnn_lstm_weather_model_best.h5')
scaler_48h = joblib.load('hourly_scaler.pkl')
scaler_7d = joblib.load('daily_scaler.pkl')
```

#### Key Functions

1. **Data Fetching Functions**
   ```python
   def fetch_weather_data(api_keys, period='48h')
   def download_weather_data(city_name, api_key)
   ```

2. **Prediction Functions**
   ```python
   def preprocess_and_predict_48h(weather_data)
   def preprocess_and_predict_7d(weather_data)
   ```

3. **Visualization Functions**
   ```python
   def create_geo_heatmap(predicted_48h, hourly_df, selected_hour)
   def get_precipitation_color(value)
   def interpret_precipitation(value)
   ```

### Database Models

#### User Model
```python
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))
```

### Form Classes

#### Authentication Forms
```python
class AdminRegistrationForm(FlaskForm)
class AdminLoginForm(FlaskForm)
class EditProfileForm(FlaskForm)
```

## 🔌 API Endpoints

### Public Endpoints

#### Dashboard
- **Route**: `/` or `/dashboard`
- **Method**: GET
- **Description**: Main application dashboard
- **Response**: HTML template with current weather overview

#### Real-time Weather
- **Route**: `/realtime_weather`
- **Method**: GET
- **Description**: Detailed real-time weather data and 48-hour forecasts
- **Features**:
  - Current weather conditions
  - 48-hour hourly predictions
  - Interactive charts
  - Weather parameter trends

#### Weather Map
- **Route**: `/weather_map`
- **Method**: GET, POST
- **Description**: Interactive geographic weather visualization
- **Features**:
  - Folium-based interactive map
  - Precipitation heatmap
  - Time-based weather selection
  - Geographic weather patterns

#### Historical Weather
- **Route**: `/past-weather`
- **Method**: GET
- **Description**: Historical weather data access
- **Features**:
  - Past weather records
  - Data analysis tools
  - Export capabilities

### Protected Endpoints

#### Admin Authentication
- **Route**: `/admin_login`
- **Method**: GET, POST
- **Authentication**: Required for POST
- **Description**: Administrator login interface

#### Admin Dashboard
- **Route**: `/admin_dashboard`
- **Method**: GET
- **Authentication**: Required
- **Description**: Administrative control panel

#### Profile Management
- **Route**: `/edit_profile`
- **Method**: GET, POST
- **Authentication**: Required
- **Description**: User profile editing

#### Logout
- **Route**: `/logout`
- **Method**: GET
- **Authentication**: Required
- **Description**: User logout functionality

## 🤖 Machine Learning Models

### Model Architecture

#### Hourly Model (48-hour forecast)
```python
# Model Structure
Sequential([
    Conv1D(filters=64, kernel_size=3, activation='relu'),
    LSTM(units=50, return_sequences=True),
    LSTM(units=50),
    Dense(units=25),
    Dense(units=4)  # Temperature, Wind Speed, Precipitation, Humidity
])
```

#### Daily Model (7-day forecast)
```python
# Similar architecture with daily aggregated data
Sequential([
    Conv1D(filters=64, kernel_size=3, activation='relu'),
    LSTM(units=50, return_sequences=True),
    LSTM(units=50),
    Dense(units=25),
    Dense(units=4)
])
```

### Data Preprocessing

#### Feature Engineering
```python
# Cyclical time encoding
data['hour_sin'] = np.sin(2 * np.pi * data.index.hour / 24)
data['hour_cos'] = np.cos(2 * np.pi * data.index.hour / 24)

# Data normalization
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)
```

#### Input Features
- **T2M**: Temperature at 2 meters (°C)
- **WS10M**: Wind speed at 10 meters (m/s)
- **PRECTOTCORR**: Total precipitation (mm)
- **RH2M**: Relative humidity at 2 meters (%)
- **hour_sin**: Cyclical hour encoding (sine)
- **hour_cos**: Cyclical hour encoding (cosine)

### Model Training

#### Training Configuration
```python
# Model compilation
model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

# Training with early stopping
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

# Model training
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping]
)
```

## 📊 Data Flow

### Real-time Data Processing

1. **API Data Fetching**
   ```python
   def fetch_weather_data(api_keys, period='48h'):
       # Multiple API keys for redundancy
       # Error handling and fallback mechanisms
       # Data validation and cleaning
   ```

2. **Data Preprocessing**
   ```python
   def preprocess_and_predict_48h(weather_data):
       # Feature engineering
       # Data scaling
       # Model input preparation
   ```

3. **Model Prediction**
   ```python
   # Load pre-trained models
   predictions = model.predict(processed_data)
   ```

4. **Result Processing**
   ```python
   # Inverse scaling
   # Format results
   # Generate visualizations
   ```

### Historical Data Management

#### Data Sources
- **melaka.csv**: Hourly historical data (2.9MB, ~14,386 records)
- **melaka_daily.csv**: Daily aggregated data (780KB, ~14,386 records)

#### Data Schema
```python
# CSV Structure
columns = ['YEAR', 'MO', 'DY', 'HR', 'T2M', 'WS10M', 'PRECTOTCORR', 'RH2M']
```

## 🔐 Security Implementation

### Authentication System
```python
# Password hashing
def set_password(self, password):
    self.password_hash = generate_password_hash(password)

def check_password(self, password):
    return check_password_hash(self.password_hash, password)
```

### Form Validation
```python
# CSRF protection
app.config['WTF_CSRF_ENABLED'] = True

# Form validation
class AdminRegistrationForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired(), Length(min=2, max=20)])
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired()])
    confirm_password = PasswordField('Confirm Password', validators=[DataRequired(), EqualTo('password')])
```

### Session Management
```python
# Flask-Login configuration
login_manager = LoginManager(app)
login_manager.login_view = 'admin_login'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))
```

## 🎨 Frontend Implementation

### Template Structure
```
templates/
├── dashboard.html          # Main dashboard
├── realtime_weather.html   # Real-time weather page
├── weather_map.html        # Interactive map
├── past_weather.html       # Historical data
├── admin_dashboard.html    # Admin panel
├── admin_login.html        # Admin login
├── admin_register.html     # Admin registration
└── edit_profile.html       # Profile editing
```

### Static Assets
```
static/
└── css/
    └── styles.css          # Custom stylesheets
```

### JavaScript Integration
- **Plotly.js**: Interactive charts and graphs
- **Folium**: Geographic mapping
- **Bootstrap**: Responsive design framework

## 🧪 Testing Strategy

### Unit Testing
```python
# Example test structure
def test_weather_data_fetching():
    # Test API data fetching
    pass

def test_model_prediction():
    # Test ML model predictions
    pass

def test_user_authentication():
    # Test user login/logout
    pass
```

### Integration Testing
- API endpoint testing
- Database integration testing
- Model prediction testing
- User workflow testing

### Performance Testing
- Load testing for concurrent users
- API response time testing
- Model prediction speed testing

## 🔧 Configuration Management

### Environment Variables
```python
# Configuration structure
app.config['SECRET_KEY'] = secrets.token_hex(16)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///weather.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
```

### API Configuration
```python
# Multiple API keys for redundancy
API_KEYS = [
    "G82543YHG2RCV4T2NKB7DADZF",
    "D8ATGRGWAUQT3RF8HQ45QXU56",
    "R3TKYVPDGHETCD6YPRMCE5CYU",
    "5ZH42W7ER6FFD2FZ9UQLXV3DY",
    "TQQLX778KF3JTAXDDXXF7SKJC"
]
```

## 📈 Performance Optimization

### Model Optimization
- **Model Caching**: Models loaded once at startup
- **Prediction Caching**: Cache predictions for repeated requests
- **Batch Processing**: Process multiple predictions efficiently

### Database Optimization
- **Indexing**: Proper database indexing for queries
- **Connection Pooling**: Efficient database connections
- **Query Optimization**: Optimized database queries

### API Optimization
- **Request Caching**: Cache API responses
- **Rate Limiting**: Respect API rate limits
- **Error Handling**: Graceful error handling and fallbacks

## 🚀 Deployment

### Development Environment
```bash
# Local development setup
python -m venv tf2env
source tf2env/bin/activate  # or tf2env\Scripts\activate on Windows
pip install -r requirements.txt
python app.py
```

### Production Deployment
```bash
# Production considerations
- Use production WSGI server (Gunicorn)
- Set up reverse proxy (Nginx)
- Configure SSL certificates
- Set up monitoring and logging
- Database backup strategies
```

### Docker Deployment
```dockerfile
# Example Dockerfile
FROM python:3.8-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["python", "app.py"]
```

## 🔍 Monitoring and Logging

### Application Logging
```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

### Performance Monitoring
- **Response Time**: Monitor API response times
- **Error Rates**: Track application errors
- **Resource Usage**: Monitor CPU and memory usage
- **Model Performance**: Track prediction accuracy

## 🤝 Contributing Guidelines

### Code Standards
- **PEP 8**: Follow Python style guidelines
- **Docstrings**: Comprehensive function documentation
- **Type Hints**: Use type hints for better code clarity
- **Error Handling**: Proper exception handling

### Git Workflow
1. Fork the repository
2. Create feature branch
3. Make changes with proper commits
4. Write tests for new features
5. Submit pull request

### Testing Requirements
- Unit tests for new functions
- Integration tests for new endpoints
- Performance tests for critical paths
- Documentation updates

## 📚 Additional Resources

### Documentation
- [Flask Documentation](https://flask.palletsprojects.com/)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [SQLAlchemy Documentation](https://docs.sqlalchemy.org/)

### Best Practices
- [Python Best Practices](https://docs.python-guide.org/)
- [Flask Best Practices](https://flask.palletsprojects.com/en/2.3.x/patterns/)
- [Machine Learning Best Practices](https://developers.google.com/machine-learning/guides/rules-of-ml)

---

**Note**: This technical documentation is intended for developers and contributors. For user documentation, see the main README.md and QUICK_START.md files.

**Last Updated**: December 2024 