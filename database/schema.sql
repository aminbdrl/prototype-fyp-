CREATE DATABASE kelantan_sentiment_system;

USE kelantan_sentiment_system;

CREATE TABLE sentiment_data (
    id INT AUTO_INCREMENT PRIMARY KEY,
    post_keyword VARCHAR(255),
    comment_text TEXT,
    username VARCHAR(255),
    like_count INT,
    reply_count INT,
    time_created VARCHAR(100),
    sentiment_label VARCHAR(50),
    sarcasm_label VARCHAR(50),
    language_id VARCHAR(50)
);

CREATE TABLE prediction_logs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    input_text TEXT,
    prediction VARCHAR(50),
    confidence INT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE admin_users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(100),
    password VARCHAR(255)
);