# TD-Join: Leveraging Temporal Patterns in Subsequence Joins

## Installation

### Clone the repository

```git clone https://github.com/gianlucarossi15/TD-Join.git```

### Switch to main branch

```git checkout main```

### Install influxDB
Follow the instructions at [https://docs.influxdata.com/influxdb/v2/install/](https://docs.influxdata.com/influxdb/v2/install/) to install InfluxDB. 

### Create a env file for storing influxDB credentials
Create a file named .env in the root directory of the project and add the following lines:
```bucket = ""
org = "your-organization"
token = "your-token"
url = "http://localhost:8086"
```
where you must create an organization ( the "database") in influxDB and the bucket (the "table") where the data will be stored.
Then a token must be created with read and write permissions for the bucket.

### Install the requirements

```pip3 install -r requirements.txt```

### Run the application

```./script.sh ```