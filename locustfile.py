from locust import HttpUser, task, between
import random

class NurturePredictionUser(HttpUser):
    # Wait time between task executions (this simulates the thinking time of the user)
    wait_time = between(1, 3)  

    # task for making the prediction request
    @task
    def predict(self):
        # Prepare the input data
        data = {
            "Age": random.randint(12, 70), 
            "SystolicBP": random.randint(60, 200),  
            "DiastolicBP": random.randint(40, 140),  
            "BS": round(random.uniform(2.0, 15.0), 2),  
            "BodyTemp": round(random.uniform(90.0, 110.0), 2),  
            "HeartRate": random.randint(50, 120)  
        }

        # Send POST request to the /predict endpoint with the above data
        response = self.client.post("/predict", json=data)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            print(f"Prediction success: {response.json()}")
        else:
            print(f"Prediction failed with status code {response.status_code}: {response.text}")

        # log the response time
        print(f"Response time: {response.elapsed.total_seconds()} seconds")