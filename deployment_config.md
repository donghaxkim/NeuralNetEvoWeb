# NeuralEVO Deployment Configuration

## Environment Variables for Render.com

### Required Variables:
- `PORT`: Automatically set by Render (8000 for Flask, 8501 for Streamlit)
- `PYTHONPATH`: Set to `/opt/render/project/src` for proper module imports

### Optional Variables:
- `FLASK_ENV`: Set to `production` for production deployment
- `STREAMLIT_SERVER_PORT`: Set to `8501` for Streamlit deployment

## Deployment Options

### Option 1: Flask Deployment (Default)
- Uses `app.py` as the main application
- Configured in `Procfile` with gunicorn
- Accessible at the root URL

### Option 2: Streamlit Deployment
- Uses `streamlit_app.py` as the main application
- Uncomment the streamlit line in `Procfile`
- Comment out the gunicorn line in `Procfile`
- Provides interactive web interface

## Render.com Configuration

The `render.yaml` file contains two service configurations:
1. `neural-evo-flask`: Flask-based deployment
2. `neural-evo-streamlit`: Streamlit-based deployment

Choose one service to deploy based on your preference.
