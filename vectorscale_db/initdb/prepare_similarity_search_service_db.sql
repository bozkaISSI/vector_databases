-- 1. Create the database
CREATE DATABASE similarity_search_service_db;

-- 2. Connect to the new database
\c similarity_search_service_db

-- 3. Enable vectorscale extension
CREATE EXTENSION IF NOT EXISTS vectorscale CASCADE;
