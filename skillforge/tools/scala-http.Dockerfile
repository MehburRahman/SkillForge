ARG JRE_IMAGE=eclipse-temurin:21.0.13_9-jre-jammy@sha256:dff6eb46527bd293a3c06bbbe9e64e8c9f91f01905b5abbdad31761c03059d2c
FROM ${JRE_IMAGE}
RUN apt-get update && apt-get install -y --no-install-recommends python3 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY src /app/src
ENV PYTHONDONTWRITEBYTECODE=1
USER 65534:65534
