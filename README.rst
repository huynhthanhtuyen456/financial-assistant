===================================
FinSC - Financial Assistant AI API
===================================

**FinSC** (Financial Assistant AI API Trading) is a comprehensive backend system designed to facilitate financial data analysis and algorithmic trading. This project leverages a robust stack including TimescaleDB, PostgreSQL, and MongoDB to manage financial datasets and serve API requests.

.. contents:: Table of Contents
   :depth: 2

Description
===========

FinSC serves as the core engine for the "Financial Assistant AI API Trading" platform. It is built to handle the ingestion, storage, and processing of stock market data, specifically targeting the Vietnam exchange stock market.

* **Version:** 0.1
* **Stack Name:** financial-assistant-api

Prerequisites
=============

Before you begin, ensure you have the following installed on your machine:

* `Docker <https://docs.docker.com/get-docker/>`_
* `Docker Compose <https://docs.docker.com/compose/install/>`_

Installation
============

Follow these steps to set up the development environment and get the application running.

1. Prepare Environment Variables
--------------------------------

Create a ``.env`` file in the root directory of your project. This file will store sensitive configuration details and connection strings for your database services.

Copy and paste the following configuration into your ``.env`` file:

.. code-block:: bash

    # TimescaleDB Configuration
    DB_HOST=timescale
    DB_HOSTNAME=timescale
    DB_PORT=5432
    DB_USER=fai
    DB_NAME=fai
    DB_PASSWORD=changeme

    # PostgreSQL Configuration
    POSTGRES_HOST=db
    POSTGRES_HOSTNAME=db
    POSTGRES_PORT=5432
    POSTGRES_USER=fai
    POSTGRES_NAME=fai
    POSTGRES_PASSWORD=changeme

    # MongoDB Configuration
    MONGO_DB_HOST=mongo
    MONGO_DB_HOSTNAME=mongo
    MONGO_DB_PORT=27017
    MONGO_DB_USER=mongo
    MONGO_DB_NAME=admin
    MONGO_DB_PASSWORD=mongo

    # Application Settings
    PROJECT_NAME=FinSC
    DEBUG_LOGS=true
    ECHO_SQL=true
    VERSION=0.1
    DESCRIPTION='Financial Assistant AI API Trading'
    STACK_NAME=financial-assistant-api
    DATAFEED_BUCKET_NAME=finsc-datafeed

2. Setup and Run Docker Containers
----------------------------------

Build and start the application containers in detached mode using Docker Compose. This will initialize the backend service along with the required databases (TimescaleDB, PostgreSQL, and MongoDB).

.. code-block:: bash

    docker compose up -d --build

3. Database Migration
---------------------

Once the containers are up and running, apply the database schema changes using Alembic. This ensures your TimescaleDB and PostgreSQL instances have the correct table structures.

.. code-block:: bash

    docker compose run -it --rm backend alembic upgrade head

Data Ingestion
==============

After setting up the infrastructure, you need to populate the database with financial data.

1. Download Symbol List
-----------------------

Fetch the list of available stock symbols on the Vietnam exchange stock market from DNSE.

.. code-block:: bash

    docker compose run -it --rm backend python scripts/download_symbol_list.py

2. Download Financial Data
--------------------------

Download historical and current financial data using the TCBS API. This script relies on the symbol list downloaded in the previous step.

.. code-block:: bash

    docker compose run -it --rm backend python scripts/download_scfa_data.py

Troubleshooting
===============

If you encounter issues connecting to the databases, ensure that:

* Ports ``5432`` (Postgres/Timescale) and ``27017`` (Mongo) are not being used by other services on your host machine.
* The ``.env`` file is correctly placed in the root directory.