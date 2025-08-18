#!/usr/bin/env python3
# test_api_v2.py - Test script for updated MLOps FastAPI endpoints

import requests
import json
import time
from datetime import datetime
import statistics

BASE_URL = "http://localhost:8000"


def test_health():
    """Test health endpoint"""
    print("Testing health endpoint...")
    response = requests.get(f"{BASE_URL}/health")
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Health check passed: {data['status']}")
        for component, status in data['components'].items():
            status_emoji = "✅" if status else "❌"
            print(f"  {status_emoji} {component}: {status}")
    else:
        print(f"❌ Health check failed: {response.status_code}")
    return response.status_code == 200


def test_data_split():
    """Test data splitting"""
    print("\nTesting data split endpoint...")
    payload = {
        "test_size": 0.001,  # 0.1% for test
        "seed": 42
    }

    response = requests.post(f"{BASE_URL}/data/split", json=payload)
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Data split successful!")
        print(f"  Train samples: {data['train_samples']:,}")
        print(f"  Test samples: {data['test_samples']:,}")
        print(f"  Test percentage: {data['test_percentage']}")
        return True
    else:
        print(f"❌ Data split failed: {response.status_code}")
        print(f"  Response: {response.text}")
        return False


def test_get_test_info():
    """Test getting test dataset info"""
    print("\nTesting test info endpoint...")
    response = requests.get(f"{BASE_URL}/data/test-info")
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Test info retrieved successfully!")
        print(f"  Total samples: {data['total_samples']}")
        print(f"  Test ID range: {data['test_id_range']}")
        print(f"  Average trip duration: {data['stats']['avg_trip_duration']:.2f} seconds")
        print(f"  Min duration: {data['stats']['min_trip_duration']:.2f} seconds")
        print(f"  Max duration: {data['stats']['max_trip_duration']:.2f} seconds")
        return True
    else:
        print(f"❌ Failed to get test info: {response.status_code}")
        return False


def test_get_test_samples():
    """Test getting test samples"""
    print("\nTesting test samples endpoint...")
    response = requests.get(f"{BASE_URL}/data/test-samples?start_id=0&limit=5")
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Retrieved {data['count']} test samples")
        print(f"  Total test samples available: {data['total_test_samples']}")

        # Display first sample
        if data['samples']:
            sample = data['samples'][0]
            print(f"\n  Sample (test_id={sample['test_id']}):")
            print(f"    Passengers: {sample['passenger_count']}")
            print(
                f"    Pickup: ({sample['pickup_location']['latitude']:.4f}, {sample['pickup_location']['longitude']:.4f})")
            print(
                f"    Dropoff: ({sample['dropoff_location']['latitude']:.4f}, {sample['dropoff_location']['longitude']:.4f})")
            print(
                f"    Actual duration: {sample['actual_duration']:.2f} seconds ({sample['actual_duration_minutes']:.2f} minutes)")
        return True
    else:
        print(f"❌ Failed to get test samples: {response.status_code}")
        return False


def test_training():
    """Test training endpoint"""
    print("\nTesting training endpoint...")
    payload = {
        "experiment_name": "test-experiment-v2",
        "use_augmentation": False,  # Faster training for testing
        "n_estimators": 30,  # Smaller model for faster training
        "max_depth": 8
    }

    response = requests.post(f"{BASE_URL}/train", json=payload)
    if response.status_code == 200:
        data = response.json()
        job_id = data['job_id']
        print(f"✅ Training job submitted: {job_id}")

        # Poll for job completion
        print("  Waiting for job to complete...")
        for i in range(60):  # Wait up to 60 seconds
            time.sleep(1)
            job_response = requests.get(f"{BASE_URL}/jobs/{job_id}")
            if job_response.status_code == 200:
                job_data = job_response.json()
                if job_data['status'] in ['success', 'failed']:
                    if job_data['status'] == 'success':
                        print(f"  ✅ Job completed successfully!")
                        if 'metrics' in job_data:
                            metrics = job_data['metrics']
                            print(f"    RMSE: {metrics['rmse']:.2f}")
                            print(f"    MAE: {metrics['mae']:.2f}")
                            print(f"    R²: {metrics['r2']:.3f}")
                    else:
                        print(f"  ❌ Job failed: {job_data.get('error', 'Unknown error')}")
                    break
            if i % 5 == 0:
                print(f"    Still running... ({i}s)")
        else:
            print("  ⚠️  Job did not complete within timeout")
        return True
    else:
        print(f"❌ Failed to submit training job: {response.status_code}")
        print(f"  Response: {response.text}")
        return False


def test_prediction_on_test_samples():
    """Test prediction on test samples with actual comparison"""
    print("\nTesting prediction on test samples...")

    # Get test info first
    info_response = requests.get(f"{BASE_URL}/data/test-info")
    if info_response.status_code != 200:
        print("❌ Cannot get test info")
        return False

    test_info = info_response.json()
    total_samples = min(5, test_info['total_samples'])  # Test on first 5 samples

    errors = []
    error_percentages = []

    print(f"  Testing predictions on {total_samples} samples...")
    for test_id in range(total_samples):
        payload = {
            "test_id": test_id,
            "experiment_name": "test-experiment-v2"
        }

        response = requests.post(f"{BASE_URL}/predict/test", json=payload)
        if response.status_code == 200:
            data = response.json()
            error = data['error']
            error_pct = data['error_percentage']
            errors.append(abs(error))
            error_percentages.append(abs(error_pct))

            print(f"\n  Test ID {test_id}:")
            print(f"    Predicted: {data['predicted_duration']:.2f} seconds")
            print(f"    Actual: {data['actual_duration']:.2f} seconds")
            print(f"    Error: {error:.2f} seconds ({error_pct:.1f}%)")

            # Show some features
            features = data['input_features']
            print(f"    Distance: {features['distance']:.4f}")
            print(f"    Hour: {features['hour']}, Day: {features['day_of_week']}")
        elif response.status_code == 404:
            print(f"  ⚠️  Test ID {test_id}: Model not found (train a model first)")
        else:
            print(f"  ❌ Test ID {test_id}: Prediction failed ({response.status_code})")

    if errors:
        print(f"\n  📊 Prediction Statistics:")
        print(f"    Average absolute error: {statistics.mean(errors):.2f} seconds")
        print(f"    Median absolute error: {statistics.median(errors):.2f} seconds")
        print(f"    Average error percentage: {statistics.mean(error_percentages):.1f}%")
        print(f"    Max error: {max(errors):.2f} seconds")
        print(f"    Min error: {min(errors):.2f} seconds")
        return True

    return False


def test_manual_prediction():
    """Test manual prediction endpoint"""
    print("\nTesting manual prediction endpoint...")
    payload = {
        "passenger_count": 2,
        "pickup_datetime": datetime.now().isoformat(),
        "pickup_longitude": -73.98,
        "pickup_latitude": 40.75,
        "dropoff_longitude": -73.97,
        "dropoff_latitude": 40.76
    }

    response = requests.post(
        f"{BASE_URL}/predict/manual?experiment_name=test-experiment-v2",
        json=payload
    )
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Manual prediction successful!")
        print(f"  Predicted duration: {data['predicted_duration']:.2f} seconds")
        print(f"  ({data['predicted_duration_minutes']:.2f} minutes)")
    elif response.status_code == 404:
        print("⚠️  No model found. Please train a model first.")
    else:
        print(f"❌ Manual prediction failed: {response.status_code}")
        print(f"  Response: {response.text}")
    return response.status_code in [200, 404]


def main():
    print("========================================")
    print("NYC Taxi MLOps API Test Suite v2.0")
    print("========================================")

    # Wait for services to be ready
    print("\nWaiting for services to be ready...")
    for i in range(30):
        try:
            response = requests.get(f"{BASE_URL}/")
            if response.status_code == 200:
                print("✅ API is ready!")
                break
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(1)
        if i % 5 == 0:
            print(f"  Waiting... ({i}s)")
    else:
        print("❌ API is not responding. Please check if services are running.")
        return

    # Run tests in sequence
    tests = [
        ("Health Check", test_health),
        ("Data Split", test_data_split),
        ("Get Test Info", test_get_test_info),
        ("Get Test Samples", test_get_test_samples),
        ("Model Training", test_training),
        ("Predictions on Test Samples", test_prediction_on_test_samples),
        ("Manual Prediction", test_manual_prediction),
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\n{'=' * 50}")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} failed with error: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 50)
    print("Test Summary")
    print("=" * 50)
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name}: {status}")

    passed = sum(1 for _, success in results if success)
    total = len(results)
    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed!")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed.")


if __name__ == "__main__":
    main()