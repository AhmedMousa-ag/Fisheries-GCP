# Fishiries-GCP

Preforming a full data science pipeline from ETL to ML model deployment on Google Cloud Platform on “fisheries Alaska data” using BigQuery to retrieve and initial statistics to data, gaining some insight using seaborn, connect BigQuery to PySpark using Data-Proc to transform data and training random forest PySpark model, and deploying the trained pipeline by making a custom model registry in Vertex Ai using Fastapi, and deployment in Vertex Ai Endpoint.

Used Pyspark Random Forest model to predict the lat400sqkm of fishes as I found that there's a correlation between the place of the fish and the year, "Could mean that global warming has effects on where fishes are or maybe something else".

![image](https://user-images.githubusercontent.com/59775002/201627757-511a9615-64b8-4cd0-a1fa-ee4d3b775b16.png)

## Data Source

https://www.fisheries.noaa.gov/about/alaska-fisheries-science-center

## Navigate Code

**Fishiries.ipynb:** Where you can visualize and see all the implemented codes on GCP which has the most important code cells to excute.

**config.py:** Used for local development.

**Utils:** Used for local development.
