# Prediction of Drug Effects on Planarian Regeneration

This project explores the impact of various pharmacological compounds on the regenerative capabilities of planarians, leveraging machine learning techniques to predict outcomes based on experimental data.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Feature Overview](#feature-overview)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Planarians are renowned for their remarkable regenerative abilities, making them ideal subjects for studying tissue regeneration and the effects of various substances on this process.

Understanding how different drugs influence regeneration can provide insights into cellular pathways and potential therapeutic applications.

This project aims to predict the effects of specific drugs on planarian regeneration using machine learning models trained on experimental data.

## Features

- Data preprocessing and cleaning
- Model training and evaluation
- Flask-based web interface for predictions
- Visualization of prediction results

## Technologies Used

- Python
- Flask
- HTML/CSS
- scikit-learn
- Pandas, NumPy, Matplotlib

## Project Structure

├── front.py # Flask application script 
├── index (2).html # Main HTML page 
├── prediction_of_drug_effects (2).py# Script for data processing and prediction
├── results.html # HTML page to display prediction results
├── styles.css # CSS file for styling the web pages 
├── training_script.py # Script to train machine learning models


## Feature Overview

The model uses seven key experimental features known to affect planarian regeneration:

- **Drug Concentration**: Amount of the compound administered.
- **Exposure Duration**: Time the organism is exposed to the drug.
- **Temperature**: Environmental temperature during the experiment.
- **pH Level**: Acidity or alkalinity of the surrounding medium.
- **Light Conditions**: Lighting conditions that may influence behavior.
- **Oxygen Levels**: Availability of oxygen in the water.
- **Nutrient Availability**: Presence of essential nutrients supporting regeneration.

These features were selected based on biological relevance and experimental significance.

## Installation

1. Clone the Repository:

```bash
git clone https://github.com/kaushikayj/Prediction-of-drug-effects-on-planarian-regeneration.git
cd Prediction-of-drug-effects-on-planarian-regeneration
