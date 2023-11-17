docker build -t model_train . --no-cache

docker run --mount source=saved_model,destination=/app/saved_model model_train