# NYC Taxi Trip Duration Prediction - MLOps Platform

A complete MLOps platform for predicting NYC taxi trip durations with automated data augmentation, model training, experiment tracking, and a user-friendly web interface.

## Features

### 1. ✅ Automated Data Augmentation Scheduler
- **Airflow DAG** running daily at 2 AM UTC
- Generates augmented data with 30% augmentation
- Memory-efficient streaming processing
- Coordinate and duration noise injection

### 2. ✅ Model Training with Augmented Data
- RandomForest regressor with optimized hyperparameters
- Feature engineering (hour, day_of_week, distance)
- Automatic train/validation split
- Data cleaning and outlier removal

### 3. ✅ MLflow Experiment Tracking
- Full experiment tracking with metrics (RMSE, MAE, R²)
- Model registry with versioning
- PostgreSQL backend for metadata
- MinIO S3-compatible storage for artifacts
- Automatic champion model selection

### 4. ✅ Model API Exposure
Two separate APIs for different use cases:
- **Management API** (port 8000): Full MLOps workflow control
- **Prediction API** (port 8888): Production predictions

### 5. ✅ User-Friendly Web UI
- **Streamlit-based interface** (port 8501)
- UX-friendly prediction options
- Real-time predictions
- Test sample browser
- Model dashboard

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Streamlit UI  │────▶│   FastAPI APIs   │────▶│     MLflow      │
│   (port 8501)   │     │ (8000 & 8888)    │     │   (port 6200)   │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                               │                          │
                               │                          │
                               ▼                          ▼
                        ┌─────────────┐          ┌────────────────┐
                        │   Airflow   │          │   PostgreSQL   │
                        │ (port 8080) │          │   + MinIO      │
                        └─────────────┘          └────────────────┘
```

## Quick Start

### 1. Start the Platform

```bash
cd airflow
docker-compose up -d
```

### 2. Access Services

| Service | URL | Description |
|---------|-----|-------------|
| **Web UI** | http://localhost:8501 | User-friendly prediction interface |
| **MLflow** | http://localhost:6200 | Experiment tracking dashboard |
| **Airflow** | http://localhost:8080 | Pipeline orchestration (user: airflow, pass: airflow) |
| **Management API** | http://localhost:8000/docs | FastAPI Swagger UI |
| **Prediction API** | http://localhost:8888/docs | Prediction API Swagger UI |
| **MinIO Console** | http://localhost:9101 | S3 storage console (user: minioadmin, pass: minioadmin123) |

## Using the Web UI

### Page 1: Quick Predict 🎯

Make instant predictions with custom trip parameters:

1. **Pickup Location**: Adjust longitude/latitude sliders
   - Longitude range: -74.05 (west) to -73.75 (east)
   - Latitude range: 40.60 (south) to 40.90 (north)

2. **Pickup Time**: Select date and time
   - Date range: Jan 1, 2016 - Jun 30, 2016

3. **Dropoff Location**: Set destination coordinates

4. **Trip Details**:
   - Number of passengers (1-6)
   - Vendor ID (1 or 2)

5. Click **"Predict Trip Duration"**

**Output:**
- Predicted duration in seconds and minutes
- Trip summary with all input parameters

### Page 2: Test Sample Prediction 🔍

Test the model against real data:

1. **Browse test samples**: View up to 100 real test cases
2. **Filter samples**:
   - By passenger count
   - By vendor ID
3. **Select a sample**: Enter sample ID from the table
4. Click **"Predict Selected Sample"**

**Output:**
- Predicted duration
- Actual duration
- Error percentage
- Complete sample details

### Page 3: Model Dashboard 📈

Monitor your MLOps system:

1. **MLflow Experiments**: View all tracked experiments
2. **Champion Model Info**:
   - Current model version
   - Performance metrics (RMSE, R²)
3. **System Health**:
   - MLflow connection status
   - Airflow connection status
   - Data availability

## API Usage

### Management API (port 8000)

#### Trigger Training
```bash
curl -X POST "http://localhost:8000/train" \
  -H "Content-Type: application/json" \
  -d '{"experiment_name": "my-experiment", "use_augmented": true}'
```

#### Get Experiments
```bash
curl "http://localhost:8000/experiments"
```

#### Predict with Test Sample
```bash
curl -X POST "http://localhost:8000/predict/test" \
  -H "Content-Type: application/json" \
  -d '{"sample_id": 42}'
```

### Prediction API (port 8888)

#### Make Prediction
```bash
curl -X POST "http://localhost:8888/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "pickup_datetime": "2016-03-15 14:30:00",
    "pickup_longitude": -73.99,
    "pickup_latitude": 40.75,
    "dropoff_longitude": -73.98,
    "dropoff_latitude": 40.76,
    "passenger_count": 1,
    "vendor_id": 1
  }'
```

#### Health Check
```bash
curl "http://localhost:8888/health"
```

## Scheduled Pipeline

The Airflow DAG (`nyc_taxi_pipeline_v2`) runs daily at 2 AM UTC:

1. **Split Data**: Divide raw data into train/test sets
2. **Baseline Training**: Train model on original data
3. **Augmentation**: Generate augmented dataset (30% increase)
4. **Augmented Training**: Train model on augmented data
5. **Evaluation**: Compare models and select champion

### Trigger Manual Run

Via UI:
1. Go to http://localhost:8080
2. Login (airflow/airflow)
3. Find `nyc_taxi_pipeline_v2`
4. Click the play button

Via API:
```bash
curl -X POST "http://localhost:8000/trigger-dag" \
  -H "Content-Type: application/json" \
  -d '{"dag_id": "nyc_taxi_pipeline_v2"}'
```

## MLflow Tracking

### View Experiments

1. Open http://localhost:6200
2. Navigate to "Experiments" tab
3. Compare runs:
   - `nyc-taxi-baseline-v2`: Models trained on original data
   - `nyc-taxi-augmented-v2`: Models trained on augmented data

### Key Metrics Tracked

- **RMSE**: Root Mean Squared Error
- **MAE**: Mean Absolute Error
- **R²**: R-squared score
- **Training Time**: Duration of training

### Model Registry

- Models are automatically registered
- Champion model is tagged as `champion`
- Version history maintained

## Configuration

### Environment Variables

Key configuration in `docker-compose.yaml`:

```yaml
MLFLOW_TRACKING_URI: "http://mlflow:5000"
MODEL_NAME: nyc-taxi-regressor
REGISTRY_ALIAS: champion
```

### Data Augmentation Parameters

Edit `mlops/config.py`:

```python
AUGMENT_FRAC = 0.3          # 30% augmentation
COORD_NOISE = 0.0008        # GPS jitter
TARGET_NOISE = 0.10         # ±10% duration variance
```

## Troubleshooting

### UI Not Loading

1. Check if all services are running:
   ```bash
   docker-compose ps
   ```

2. Check UI logs:
   ```bash
   docker-compose logs ui
   ```

3. Restart UI service:
   ```bash
   docker-compose restart ui
   ```

### API Connection Issues

Verify API health:
```bash
curl http://localhost:8000/health
curl http://localhost:8888/health
```

### Model Not Found

Ensure champion model exists:
1. Check MLflow UI: http://localhost:6200
2. Run the pipeline to train a model
3. Verify `champion.json` exists in project root

### Container Issues

Reset everything:
```bash
docker-compose down -v
docker-compose up -d
```

## Development

### Project Structure

```
mlops-flow-1/
├── airflow/                    # Orchestration
│   ├── dags/                   # Airflow DAGs
│   ├── docker-compose.yaml     # Full stack deployment
│   ├── Dockerfile              # Airflow image
│   ├── Dockerfile.mlflow       # MLflow server image
│   └── Dockerfile.fastapi      # FastAPI services image
├── mlops/                      # Core ML modules
│   ├── augment.py             # Data augmentation
│   ├── train.py               # Model training
│   └── config.py              # Configuration
├── app/                        # API services
│   ├── main.py                # Management API
│   └── service/app.py         # Prediction API
├── ui/                         # Web interface
│   ├── app.py                 # Streamlit UI
│   ├── Dockerfile             # UI image
│   └── requirements.txt       # UI dependencies
└── data/                       # Data directory
```

### Adding Custom Features to UI

Edit `ui/app.py`:

1. **Add new pages**: Modify the `page` radio button options
2. **Add visualizations**: Use `st.pyplot()` or `st.plotly_chart()`
3. **Add metrics**: Use `st.metric()` for KPIs
4. **Add forms**: Use `st.form()` for complex inputs

### Extending the Pipeline

Edit `airflow/dags/nyc_taxi_pipeline_dag.py`:

1. Add new tasks to the DAG
2. Define task dependencies
3. Use `@task` decorator for Python tasks
4. Trigger manually or adjust schedule

## Performance Tuning

### Model Training

Adjust hyperparameters in `mlops/train.py`:

```python
n_estimators = 80      # Number of trees
max_depth = 12         # Tree depth
random_state = 42      # Reproducibility
```

### Data Augmentation

Tune augmentation in `mlops/augment.py`:

```python
chunk_size = 200_000   # Memory vs speed tradeoff
```

### API Performance

For production:
1. Increase workers: `--workers 4` in docker-compose command
2. Use Gunicorn instead of Uvicorn
3. Enable caching for predictions

## Security Notes

**Development Setup - Not Production Ready!**

- Default credentials are used (change in production)
- No authentication on APIs
- No HTTPS/TLS
- No rate limiting
- No input validation hardening

For production deployment:
1. Add authentication (OAuth2, JWT)
2. Enable HTTPS with proper certificates
3. Implement rate limiting
4. Add comprehensive input validation
5. Use secrets management (not environment variables)
6. Enable audit logging

## Next Steps

1. **Data Quality Monitoring**: Add data drift detection
2. **Model Monitoring**: Track prediction quality over time
3. **A/B Testing**: Compare multiple models in production
4. **Real-time Predictions**: WebSocket support for live updates
5. **Advanced Visualizations**: Add charts for feature importance, error analysis
6. **User Authentication**: Add login system
7. **Prediction History**: Store and analyze past predictions
8. **Custom Models**: Allow uploading custom trained models

## Support

For issues or questions:
1. Check logs: `docker-compose logs [service-name]`
2. Review MLflow experiments for training issues
3. Check Airflow DAG logs for pipeline failures
4. Verify data files exist in `data/` directory

## License

This project is for educational and development purposes.
