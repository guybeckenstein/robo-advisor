FROM registry.access.redhat.com/ubi8/python-39

# Install application into container image
WORKDIR /app
COPY . .

# Expose port used by our app server
EXPOSE 8000

# Install dependencies into container image
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Run the application
ENTRYPOINT [ "/app/entrypoint.sh" ]