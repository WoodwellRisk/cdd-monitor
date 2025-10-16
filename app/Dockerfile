# 1. Containerize app to create Dockerfile (done)
# 2. Build Docker image: docker build -t shiny-app:latest --platform linux/amd64 .
# 3. docker run --platform linux/amd64 -p 8080:8080 shiny-app:latest

# https://hub.docker.com/_/python/
FROM python:3.12

# set environment variables
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_ROOT_USER_ACTION=ignore

# make the app available by listening on general host and exposing the right port
ENV HOST=0.0.0.0
EXPOSE 8080

# install curl so that we can install google cloud sdk
RUN apt-get -y update; apt-get -y install curl

# downloads the gCloud package
# https://stackoverflow.com/questions/70902923/unable-to-start-a-cloud-run-container-on-m1-macbook/71072262#71072262
RUN curl https://dl.google.com/dl/cloudsdk/release/google-cloud-sdk.tar.gz > /tmp/google-cloud-sdk.tar.gz

# install google cloud sdk
# https://stackoverflow.com/questions/28372328/how-to-install-the-google-cloud-sdk-in-a-docker-image
RUN curl -sSL https://sdk.cloud.google.com | bash
ENV PATH=$PATH:/root/google-cloud-sdk/bin

# RUN conda install -y \
RUN pip install --no-cache-dir \
    anywidget \
    fiona  \
    folium \
    geopandas \
    h5netcdf \
    matplotlib \
    netcdf4 \
    numpy \
    pandas \
    plotly \
    proj \
    pyarrow \
    pyproj \
    rasterio \
    rioxarray \
    shapely \
    shiny \
    shinywidgets \
    xarray \
    zarr

# copy application files
COPY . /app
WORKDIR /app

# run the application
# https://docs.docker.com/reference/dockerfile/#understand-how-cmd-and-entrypoint-interact
CMD ["shiny", "run", "app.py", "--host", "0.0.0.0", "--port", "8080"]