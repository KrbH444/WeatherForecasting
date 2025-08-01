# 🚀 Deployment Guide - Weather Forecasting System

This guide covers different deployment options for the Weather Forecasting System.

## 📋 Table of Contents

- [Local Development](#local-development)
- [Docker Deployment](#docker-deployment)
- [Cloud Deployment](#cloud-deployment)
- [Production Considerations](#production-considerations)
- [Monitoring and Logging](#monitoring-and-logging)
- [Troubleshooting](#troubleshooting)

## 💻 Local Development

### Prerequisites
- Python 3.8+
- pip
- Git

### Quick Start
```bash
# Clone repository
git clone <repository-url>
cd weather-forecasting-system

# Run setup script
python setup.py

# Activate virtual environment
tf2env\Scripts\activate  # Windows
source tf2env/bin/activate  # macOS/Linux

# Run application
python app.py
```

### Manual Setup
```bash
# Create virtual environment
python -m venv tf2env
tf2env\Scripts\activate  # Windows
source tf2env/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export FLASK_APP=app.py
export FLASK_ENV=development
export SECRET_KEY=your-secret-key

# Initialize database
flask db init
flask db migrate
flask db upgrade

# Run application
python app.py
```

## 🐳 Docker Deployment

### Prerequisites
- Docker
- Docker Compose

### Quick Start with Docker
```bash
# Build and run with Docker Compose
docker-compose up --build

# Run in background
docker-compose up -d

# View logs
docker-compose logs -f weather-app

# Stop services
docker-compose down
```

### Manual Docker Build
```bash
# Build image
docker build -t weather-forecast .

# Run container
docker run -p 5000:5000 \
  -e SECRET_KEY=your-secret-key \
  -v $(pwd)/instance:/app/instance \
  weather-forecast
```

### Docker Environment Variables
```bash
# Create .env file
SECRET_KEY=your-secret-key-here
FLASK_ENV=production
DATABASE_URL=sqlite:///weather.db
```

## ☁️ Cloud Deployment

### Heroku Deployment

#### Prerequisites
- Heroku CLI
- Git repository

#### Deployment Steps
```bash
# Login to Heroku
heroku login

# Create Heroku app
heroku create your-weather-app

# Set environment variables
heroku config:set SECRET_KEY=your-secret-key
heroku config:set FLASK_ENV=production

# Deploy
git push heroku main

# Open app
heroku open
```

#### Heroku Files Required
```python
# Procfile
web: gunicorn app:app

# runtime.txt
python-3.8.10
```

### AWS Deployment

#### EC2 Instance
```bash
# Connect to EC2 instance
ssh -i your-key.pem ubuntu@your-instance-ip

# Install dependencies
sudo apt update
sudo apt install python3-pip python3-venv nginx

# Clone repository
git clone <repository-url>
cd weather-forecasting-system

# Setup application
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Setup Gunicorn
pip install gunicorn

# Create systemd service
sudo nano /etc/systemd/system/weather-app.service
```

#### Systemd Service File
```ini
[Unit]
Description=Weather Forecasting System
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/weather-forecasting-system
Environment="PATH=/home/ubuntu/weather-forecasting-system/venv/bin"
ExecStart=/home/ubuntu/weather-forecasting-system/venv/bin/gunicorn --workers 3 --bind unix:weather-app.sock -m 007 app:app

[Install]
WantedBy=multi-user.target
```

#### Nginx Configuration
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        include proxy_params;
        proxy_pass http://unix:/home/ubuntu/weather-forecasting-system/weather-app.sock;
    }
}
```

### Google Cloud Platform

#### App Engine
```yaml
# app.yaml
runtime: python38
entrypoint: gunicorn -b :$PORT app:app

env_variables:
  SECRET_KEY: "your-secret-key"
  FLASK_ENV: "production"

handlers:
- url: /static
  static_dir: static
- url: /.*
  script: auto
```

#### Deployment Command
```bash
gcloud app deploy
```

### Azure Deployment

#### Azure App Service
```bash
# Install Azure CLI
az login

# Create resource group
az group create --name weather-app-rg --location eastus

# Create app service plan
az appservice plan create --name weather-app-plan --resource-group weather-app-rg --sku B1

# Create web app
az webapp create --name your-weather-app --resource-group weather-app-rg --plan weather-app-plan --runtime "PYTHON|3.8"

# Deploy
az webapp deployment source config-local-git --name your-weather-app --resource-group weather-app-rg
```

## 🔧 Production Considerations

### Security
```python
# Production settings
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY')
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# HTTPS only
if not app.debug:
    app.config['SESSION_COOKIE_SECURE'] = True
    app.config['SESSION_COOKIE_HTTPONLY'] = True
```

### Performance Optimization
```python
# Use production WSGI server
# requirements.txt
gunicorn==20.1.0

# Run with Gunicorn
gunicorn --workers 4 --bind 0.0.0.0:5000 app:app
```

### Database Configuration
```python
# PostgreSQL (recommended for production)
DATABASE_URL=postgresql://username:password@localhost/weather_db

# MySQL
DATABASE_URL=mysql://username:password@localhost/weather_db
```

### Environment Variables
```bash
# Production environment variables
export SECRET_KEY=your-very-secure-secret-key
export FLASK_ENV=production
export DATABASE_URL=postgresql://user:pass@localhost/weather_db
export OPENWEATHER_API_KEY=your-api-key
```

## 📊 Monitoring and Logging

### Application Logging
```python
import logging
from logging.handlers import RotatingFileHandler

if not app.debug:
    file_handler = RotatingFileHandler('logs/weather-app.log', maxBytes=10240, backupCount=10)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
    ))
    file_handler.setLevel(logging.INFO)
    app.logger.addHandler(file_handler)
    app.logger.setLevel(logging.INFO)
    app.logger.info('Weather Forecasting System startup')
```

### Health Checks
```python
@app.route('/health')
def health_check():
    return {'status': 'healthy', 'timestamp': datetime.now().isoformat()}
```

### Performance Monitoring
```python
# Add monitoring middleware
from flask import request, g
import time

@app.before_request
def before_request():
    g.start = time.time()

@app.after_request
def after_request(response):
    diff = time.time() - g.start
    app.logger.info(f'Request to {request.endpoint} took {diff:.2f}s')
    return response
```

## 🔍 Troubleshooting

### Common Issues

#### Port Already in Use
```bash
# Find process using port 5000
lsof -i :5000
# or
netstat -tulpn | grep :5000

# Kill process
kill -9 <PID>
```

#### Database Issues
```bash
# Reset database
rm instance/weather.db
flask db upgrade

# Check database connection
flask shell
>>> from app import db
>>> db.engine.execute('SELECT 1')
```

#### Model Loading Errors
```bash
# Check model files exist
ls -la *.h5 *.pkl

# Verify TensorFlow version
python -c "import tensorflow as tf; print(tf.__version__)"
```

#### API Key Issues
```bash
# Test API connection
curl "https://api.openweathermap.org/data/2.5/weather?q=Melaka&appid=YOUR_API_KEY"
```

### Log Analysis
```bash
# View application logs
tail -f logs/weather-app.log

# Search for errors
grep ERROR logs/weather-app.log

# Monitor system resources
htop
df -h
free -h
```

### Performance Issues
```bash
# Profile application
python -m cProfile -o profile.stats app.py

# Analyze profile
python -c "import pstats; p = pstats.Stats('profile.stats'); p.sort_stats('cumulative').print_stats(10)"
```

## 📈 Scaling Considerations

### Horizontal Scaling
- Use load balancer (Nginx, HAProxy)
- Multiple application instances
- Shared database (PostgreSQL, MySQL)
- Redis for session storage

### Vertical Scaling
- Increase server resources
- Optimize database queries
- Use connection pooling
- Implement caching

### Microservices Architecture
- Separate weather API service
- Dedicated ML prediction service
- Independent user management service
- Message queue for async tasks

## 🔄 CI/CD Pipeline

### GitHub Actions
```yaml
# .github/workflows/deploy.yml
name: Deploy to Production

on:
  push:
    branches: [ main ]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    
    - name: Deploy to Heroku
      uses: akhileshns/heroku-deploy@v3.12.12
      with:
        heroku_api_key: ${{ secrets.HEROKU_API_KEY }}
        heroku_app_name: ${{ secrets.HEROKU_APP_NAME }}
        heroku_email: ${{ secrets.HEROKU_EMAIL }}
```

### Docker Hub
```bash
# Build and push to Docker Hub
docker build -t yourusername/weather-forecast .
docker push yourusername/weather-forecast
```

---

**Note**: Always test deployment in a staging environment before deploying to production. Keep backups of your database and configuration files. 