# 🌤️ Weather Forecasting System - Melaka, Malaysia

A comprehensive weather forecasting web application that provides real-time weather data, AI-powered predictions, and interactive visualizations for Melaka, Malaysia. Built with Flask, TensorFlow/Keras, and modern web technologies.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Technology Stack](#technology-stack)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [API Documentation](#api-documentation)
- [Machine Learning Models](#machine-learning-models)
- [Contributing](#contributing)
- [License](#license)

## 🌟 Overview

This weather forecasting system combines real-time weather data from OpenWeatherMap API with advanced machine learning models to provide accurate weather predictions for Melaka, Malaysia. The application features both hourly (48-hour) and daily (7-day) forecasting capabilities using CNN-LSTM hybrid models.

### Key Highlights

- **Real-time Weather Data**: Live weather information from multiple API sources
- **AI-Powered Predictions**: CNN-LSTM models for accurate weather forecasting
- **Interactive Visualizations**: Dynamic charts, maps, and heatmaps
- **User Management**: Secure admin authentication system
- **Responsive Design**: Modern, mobile-friendly interface

## ✨ Features

### 🌡️ Weather Forecasting
- **48-Hour Hourly Forecast**: Detailed hourly predictions for the next 48 hours
- **7-Day Daily Forecast**: Extended daily weather predictions
- **Real-time Weather**: Current weather conditions with live updates
- **Historical Data**: Access to past weather records

### 🗺️ Interactive Visualizations
- **Geographic Heatmaps**: Visual precipitation intensity mapping
- **Time-series Charts**: Temperature, humidity, and precipitation trends
- **Weather Maps**: Interactive map-based weather display
- **Dynamic Graphs**: Real-time updating weather charts

### 🔐 User Management
- **Admin Authentication**: Secure login system for administrators
- **Profile Management**: User profile editing capabilities
- **Session Management**: Secure user sessions

### 📊 Data Analysis
- **Multiple API Integration**: Redundant API keys for reliability
- **Data Preprocessing**: Advanced feature engineering
- **Model Performance**: Real-time prediction accuracy metrics

## 🛠️ Technology Stack

### Backend
- **Python 3.8+**: Core programming language
- **Flask**: Web framework for the application
- **SQLAlchemy**: Database ORM
- **Flask-Login**: User authentication
- **Flask-WTF**: Form handling and validation

### Machine Learning
- **TensorFlow/Keras**: Deep learning framework
- **Scikit-learn**: Machine learning utilities
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing

### Data Visualization
- **Plotly**: Interactive charts and graphs
- **Folium**: Geographic mapping and heatmaps
- **Matplotlib**: Static plotting (for model development)

### Frontend
- **Tailwind CSS**: Utility-first CSS framework
- **Font Awesome**: Icon library
- **Inter Font**: Modern typography

### External APIs
- **OpenWeatherMap API**: Real-time weather data
- **Multiple API Keys**: Redundant data sources for reliability

### Database
- **SQLite**: Lightweight database for user management

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)
- Git (for cloning the repository)
- Node.js 16+ (for Tailwind CSS)
- npm (comes with Node.js)

### Step-by-Step Setup

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd weather-forecasting-system
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv tf2env
   ```

3. **Activate Virtual Environment**
   ```bash
   # Windows
   tf2env\Scripts\activate
   
   # macOS/Linux
   source tf2env/bin/activate
   ```

4. **Install Python Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Install Frontend Dependencies**
   ```bash
   npm install
   npm run build:css:prod
   ```

6. **Set Up Environment Variables**
   Create a `.env` file in the root directory:
   ```env
   FLASK_APP=app.py
   FLASK_ENV=development
   SECRET_KEY=your-secret-key-here
   ```

7. **Initialize Database**
   ```bash
   flask db init
   flask db migrate
   flask db upgrade
   ```

8. **Run the Application**
   ```bash
   python app.py
   ```

The application will be available at `http://localhost:5000`

## 📖 Usage

### For End Users

1. **Access the Dashboard**
   - Navigate to the main dashboard to view current weather conditions
   - Explore real-time weather data and basic forecasts

2. **View Real-time Weather**
   - Check current temperature, humidity, wind speed, and precipitation
   - View hourly weather trends and patterns

3. **Interactive Weather Map**
   - Explore geographic weather visualization
   - Select different time periods for weather analysis

4. **Historical Weather Data**
   - Access past weather records
   - Analyze weather patterns over time

### For Administrators

1. **Admin Login**
   - Access the admin panel at `/admin_login`
   - Register new admin accounts if needed

2. **Admin Dashboard**
   - Monitor system performance
   - View detailed weather analytics
   - Manage user accounts

3. **Profile Management**
   - Update admin credentials
   - Modify user information

## 📁 Project Structure

```
weather-forecasting-system/
├── app.py                          # Main Flask application
├── templates/                      # HTML templates
│   ├── dashboard.html             # Main dashboard
│   ├── realtime_weather.html      # Real-time weather page
│   ├── weather_map.html           # Interactive map
│   ├── past_weather.html          # Historical data
│   ├── admin_dashboard.html       # Admin panel
│   ├── admin_login.html           # Admin login
│   ├── admin_register.html        # Admin registration
│   └── edit_profile.html          # Profile editing
├── static/                        # Static assets
│   └── css/                       # Stylesheets
├── models/                        # Machine learning models
│   ├── hourly_cnn_lstm_weather_model.h5
│   └── daily_cnn_lstm_weather_model_best.h5
├── data/                          # Data files
│   ├── melaka.csv                 # Historical weather data
│   ├── melaka_daily.csv           # Daily aggregated data
│   ├── hourly_scaler.pkl          # Hourly data scaler
│   └── daily_scaler.pkl           # Daily data scaler
├── notebooks/                     # Jupyter notebooks
│   └── Hourly and Daily LSTM MODEL.ipynb
├── instance/                      # Database files
├── tf2env/                        # Virtual environment
└── README.md                      # Project documentation
```

## 🔌 API Documentation

### Weather Data Endpoints

#### Real-time Weather
- **URL**: `/realtime_weather`
- **Method**: GET
- **Description**: Displays current weather conditions and 48-hour forecasts

#### Weather Map
- **URL**: `/weather_map`
- **Method**: GET, POST
- **Description**: Interactive geographic weather visualization

#### Historical Weather
- **URL**: `/past-weather`
- **Method**: GET
- **Description**: Access to historical weather records

### Admin Endpoints

#### Admin Login
- **URL**: `/admin_login`
- **Method**: GET, POST
- **Description**: Administrator authentication

#### Admin Dashboard
- **URL**: `/admin_dashboard`
- **Method**: GET
- **Description**: Administrative control panel

#### Profile Management
- **URL**: `/edit_profile`
- **Method**: GET, POST
- **Description**: User profile editing

## 🤖 Machine Learning Models

### Model Architecture

The system uses hybrid CNN-LSTM models for weather prediction:

#### Hourly Model (48-hour forecast)
- **Input**: Historical weather data with cyclical time features
- **Architecture**: Conv1D + LSTM + Dense layers
- **Output**: Temperature, wind speed, precipitation, humidity
- **Features**: 
  - Temperature (T2M)
  - Wind Speed (WS10M)
  - Precipitation (PRECTOTCORR)
  - Relative Humidity (RH2M)
  - Cyclical time encoding (hour_sin, hour_cos)

#### Daily Model (7-day forecast)
- **Input**: Daily aggregated weather data
- **Architecture**: Similar CNN-LSTM hybrid
- **Output**: Daily weather predictions
- **Features**: Daily averages of weather parameters

### Data Preprocessing

1. **Feature Engineering**
   - Cyclical encoding for time features
   - Data normalization using MinMaxScaler
   - Missing value handling

2. **Model Training**
   - Train-test split for validation
   - Early stopping to prevent overfitting
   - Hyperparameter optimization

3. **Model Performance**
   - Mean Squared Error (MSE) evaluation
   - Mean Absolute Error (MAE) metrics
   - Real-time prediction accuracy monitoring

## 🔧 Configuration

### API Keys
The application uses multiple OpenWeatherMap API keys for redundancy:
```python
API_KEYS = [
    "G82543YHG2RCV4T2NKB7DADZF",
    "D8ATGRGWAUQT3RF8HQ45QXU56",
    "R3TKYVPDGHETCD6YPRMCE5CYU",
    "5ZH42W7ER6FFD2FZ9UQLXV3DY",
    "TQQLX778KF3JTAXDDXXF7SKJC"
]
```

### Database Configuration
- **Database**: SQLite
- **Location**: `instance/weather.db`
- **Tables**: Users (for admin management)

## 🚨 Troubleshooting

### Common Issues

1. **Model Loading Errors**
   - Ensure all model files are in the correct directory
   - Check TensorFlow version compatibility

2. **API Connection Issues**
   - Verify API keys are valid
   - Check internet connectivity
   - Monitor API rate limits

3. **Database Errors**
   - Run database migrations: `flask db upgrade`
   - Check file permissions for the instance directory

4. **Import Errors**
   - Activate virtual environment
   - Install missing dependencies: `pip install -r requirements.txt`

### Performance Optimization

1. **Model Caching**
   - Models are loaded once at startup
   - Predictions are cached for better performance

2. **API Optimization**
   - Multiple API keys for load balancing
   - Request caching to reduce API calls

## 🤝 Contributing

We welcome contributions to improve the weather forecasting system!

### How to Contribute

1. **Fork the Repository**
2. **Create a Feature Branch**: `git checkout -b feature/amazing-feature`
3. **Commit Changes**: `git commit -m 'Add amazing feature'`
4. **Push to Branch**: `git push origin feature/amazing-feature`
5. **Open a Pull Request**

### Development Guidelines

- Follow PEP 8 style guidelines
- Add comprehensive docstrings
- Include unit tests for new features
- Update documentation for API changes

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **OpenWeatherMap**: For providing weather data APIs
- **TensorFlow/Keras**: For deep learning capabilities
- **Flask Community**: For the excellent web framework
- **Melaka Weather Data**: Historical weather records for model training

## 📞 Support

For support and questions:
- Create an issue in the GitHub repository
- Contact the development team
- Check the troubleshooting section above

---

**Note**: This weather forecasting system is specifically designed for Melaka, Malaysia. For use in other locations, the models would need to be retrained with local weather data.

**Last Updated**: December 2024 