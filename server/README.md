### Airline Classification Prediction

To create a Python virtual environment, you can use the following commands:

1. Navigate to your project directory:
```sh
    cd /Users/andressalguero/Documents/lambton/2203_Advanced_Python_AI_and_ML/airline_reviews/server
```

2. Create a virtual environment:
```sh
    python3 -m venv venv
```

3. Activate the virtual environment:
- On macOS and Linux:
```sh
    source venv/bin/activate
```
- On Windows:
```sh
    .\venv\Scripts\activate
```

4. To deactivate the virtual environment, simply run:
```sh
    deactivate
```

5. Install the required packages:
```sh
    pip install -r requirements.txt
```

6. Run the Flask server with Gunicorn:
```sh
    gunicorn -w 4 app:app
```
7. Enable Docker Buildx to create a multi-architecture image:

```bash
    docker buildx create --use
    docker buildx inspect --bootstrap
```

8. LTo build for a specific architecture (amd64) and push the image to Docker Hub, use the following command:
```bash
docker buildx build --platform linux/amd64 -t afscomercial/airline_reviews:latest --push .
```
