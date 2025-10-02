# NeuralEVO Deployment Guide for Render.com

This guide will help you deploy your Neural Network Evolution simulation to Render.com.

## Prerequisites

1. A GitHub account
2. A Render.com account (free tier available)
3. Your NeuralEVO project pushed to a GitHub repository

## Deployment Steps

### Step 1: Prepare Your Repository

1. Ensure all files are committed and pushed to your GitHub repository:
   ```bash
   git add .
   git commit -m "Prepare for Render deployment"
   git push origin main
   ```

### Step 2: Deploy to Render.com

#### Option A: Using render.yaml (Recommended)

1. Go to [Render.com](https://render.com) and sign in
2. Click "New +" → "Blueprint"
3. Connect your GitHub repository
4. Render will automatically detect the `render.yaml` file
5. Choose either:
   - **Flask Service** (`neural-evo-flask`): Traditional web app
   - **Streamlit Service** (`neural-evo-streamlit`): Interactive dashboard

#### Option B: Manual Web Service

1. Go to [Render.com](https://render.com) and sign in
2. Click "New +" → "Web Service"
3. Connect your GitHub repository
4. Configure the service:
   - **Name**: `neural-evo`
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: 
     - For Flask: `gunicorn app:app --bind 0.0.0.0:$PORT`
     - For Streamlit: `streamlit run streamlit_app.py --server.port=$PORT --server.address=0.0.0.0`
   - **Plan**: Free

### Step 3: Environment Variables

Render will automatically set:
- `PORT`: The port your app should listen on
- `PYTHONPATH`: Set to `/opt/render/project/src`

### Step 4: Deploy

1. Click "Create Web Service"
2. Render will build and deploy your application
3. Wait for the deployment to complete (usually 2-5 minutes)
4. Your app will be available at the provided URL

## Deployment Options

### Flask Deployment (Default)
- **File**: `app.py`
- **Features**: REST API endpoints, real-time simulation
- **Best for**: API-based applications, mobile apps
- **URL Structure**: 
  - `/` - Main page
  - `/api/status` - Simulation status
  - `/api/frame` - Current frame
  - `/api/control` - Control simulation

### Streamlit Deployment
- **File**: `streamlit_app.py`
- **Features**: Interactive dashboard, real-time controls
- **Best for**: Data science demos, interactive presentations
- **Features**:
  - Interactive controls
  - Real-time statistics
  - Neural network visualization
  - Auto-play functionality

## Switching Between Deployments

To switch from Flask to Streamlit (or vice versa):

1. Edit the `Procfile`:
   ```bash
   # For Flask (default)
   web: gunicorn app:app --bind 0.0.0.0:$PORT
   
   # For Streamlit (uncomment this line and comment the above)
   # web: streamlit run streamlit_app.py --server.port=$PORT --server.address=0.0.0.0
   ```

2. Commit and push changes:
   ```bash
   git add Procfile
   git commit -m "Switch to Streamlit deployment"
   git push origin main
   ```

3. Render will automatically redeploy with the new configuration

## Troubleshooting

### Common Issues

1. **Build Failures**:
   - Check that all dependencies are in `requirements.txt`
   - Ensure Python version compatibility (3.11.0 specified in `runtime.txt`)

2. **Import Errors**:
   - Verify all Python files are in the repository
   - Check that `PYTHONPATH` is set correctly

3. **Pygame Issues**:
   - Pygame should work on Render's free tier
   - If issues persist, consider using a different graphics library

4. **Memory Issues**:
   - The free tier has limited memory
   - Consider reducing population size or simulation complexity

### Performance Optimization

1. **Reduce Resource Usage**:
   - Lower population size in the simulation
   - Reduce frame rate or update frequency
   - Optimize neural network size

2. **Caching**:
   - Implement caching for rendered frames
   - Use session storage for simulation state

## Monitoring

- Check Render dashboard for deployment status
- Monitor logs for any errors
- Use Render's built-in metrics to track performance

## Cost Considerations

- **Free Tier**: 750 hours/month, sleeps after 15 minutes of inactivity
- **Paid Plans**: Start at $7/month for always-on service
- **Auto-sleep**: Free tier apps sleep when not in use (wake up takes ~30 seconds)

## Support

- Render Documentation: https://render.com/docs
- Render Community: https://community.render.com
- Project Issues: Create an issue in your GitHub repository

## Next Steps

After successful deployment:

1. Test all functionality
2. Share the URL with others
3. Monitor performance and usage
4. Consider upgrading to a paid plan for production use
5. Set up custom domain (paid plans only)
