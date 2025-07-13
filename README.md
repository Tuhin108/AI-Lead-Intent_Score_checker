# 🎯 AI Lead Scoring Dashboard

A comprehensive machine learning-powered lead scoring system built with Streamlit, featuring real-time predictions, analytics, and a modern responsive UI.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Model Details](#model-details)
- [File Structure](#file-structure)
- [Configuration](#configuration)
- [Deployment](#deployment)
- [Contributing](#contributing)

## 🎯 Overview

The AI Lead Scoring Dashboard is an intelligent sales automation tool that predicts lead conversion probability using machine learning. It analyzes multiple data points including financial profile, demographics, professional background, and digital engagement to provide actionable insights for sales teams.

### Key Benefits
- **Automated Lead Scoring**: ML-powered predictions with 0-100% conversion probability
- **Real-time Analytics**: Interactive dashboards with historical trends
- **Priority Classification**: Automatic lead prioritization (High/Medium/Low)
- **Modern UI**: Clean, responsive interface with dark mode support
- **Data Privacy**: GDPR-compliant with consent management

## ✨ Features

### Core Functionality
- **Machine Learning Model**: Gradient Boosting Classifier for accurate predictions
- **Multi-factor Analysis**: Credit score, income, demographics, engagement metrics
- **Real-time Scoring**: Instant lead evaluation with detailed breakdowns
- **Historical Tracking**: Complete lead history with analytics
- **Priority Recommendations**: Automated follow-up suggestions

### User Interface
- **Responsive Design**: Mobile-friendly with modern CSS styling
- **Interactive Charts**: Plotly-powered visualizations
- **Progress Indicators**: Visual score representations
- **Export Capabilities**: Data download and reporting features
- **Form Validation**: Input validation with helpful error messages

### Analytics & Reporting
- **Score Distribution**: Histogram analysis of lead scores
- **Priority Breakdown**: Pie chart of priority levels
- **Performance Metrics**: Average scores, conversion rates
- **Trend Analysis**: Time-series lead scoring patterns

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Frontend (Streamlit)                     │
├─────────────────────────────────────────────────────────────┤
│  • Form Input Processing    • Data Visualization           │
│  • Real-time Updates        • User Authentication          │
│  • Responsive UI            • Export Functions             │
└─────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────┐
│                  Business Logic Layer                       │
├─────────────────────────────────────────────────────────────┤
│  • Input Validation         • Score Calculation            │
│  • Data Preprocessing       • Priority Assignment          │
│  • Model Integration        • History Management           │
└─────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────┐
│                   ML Model Layer                           │
├─────────────────────────────────────────────────────────────┤
│  • GradientBoostingClassifier  • Feature Engineering       │
│  • Preprocessing Pipeline      • Model Validation          │
│  • Real-time Prediction        • Performance Monitoring    │
└─────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────┐
│                    Data Layer                              │
├─────────────────────────────────────────────────────────────┤
│  • Lead Database (CSV)      • Model Persistence (PKL)     │
│  • Session State Storage    • Configuration Files         │
│  • Backup & Recovery        • Data Validation             │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- 4GB+ RAM recommended

### Quick Setup

1. **Clone the repository**:
```bash
git clone <repository-url>
cd ai-lead-scoring-dashboard
```

2. **Run the automated setup**:
```bash
python setup.py
```

The setup script will:
- Install all required dependencies
- Generate sample training data
- Train the ML model
- Verify installation

### Manual Installation

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Generate sample data** (optional):
```bash
python generate_sample_data.py
```

3. **Train the model**:
```bash
python train_model.py
```

4. **Start the application**:
```bash
streamlit run app.py
```

## 📖 Usage

### Starting the Application

```bash
streamlit run app.py
```

The dashboard will be available at `http://localhost:8501`

### Using the Dashboard

1. **Lead Information Input**:
   - Fill out the sidebar form with prospect details
   - Include financial profile (credit score, income)
   - Add demographics and professional information
   - Set website engagement score

2. **Score Generation**:
   - Click "Calculate Lead Score" to get predictions
   - View conversion probability and priority level
   - Review automated recommendations

3. **Analytics Review**:
   - Check historical lead performance
   - Analyze score distributions and trends
   - Export data for further analysis

### Input Fields

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| Credit Score | Slider | 300-850 | FICO credit score |
| Annual Income | Number | $0-$1M | Yearly income in USD |
| Age Group | Select | 18-65+ | Age demographic |
| Family Background | Select | Various | Marital status |
| Education Level | Select | High School-PhD | Educational attainment |
| Employment Type | Select | Various | Employment status |
| Website Engagement | Slider | 0-100 | Digital engagement score |

## 🤖 Model Details

### Algorithm
- **Primary Model**: Gradient Boosting Classifier
- **Framework**: scikit-learn
- **Features**: 7 engineered features
- **Target**: Binary classification (convert/no-convert)

### Feature Engineering
- **Categorical Encoding**: Label encoding for categorical variables
- **Numerical Scaling**: StandardScaler for continuous features
- **Missing Value Handling**: Median/mode imputation

### Model Performance
- **Training Method**: 80/20 train-test split
- **Validation**: Stratified sampling
- **Metrics**: ROC-AUC, precision, recall, F1-score
- **Cross-validation**: 5-fold CV for robust evaluation

### Preprocessing Pipeline
```python
# Feature encoding
encoders = {
    'age_group': LabelEncoder(),
    'family_background': LabelEncoder(),
    'education_level': LabelEncoder(),
    'employment_type': LabelEncoder(),
    'scaler': StandardScaler()
}

# Feature scaling
scaled_features = scaler.fit_transform(numerical_features)
```

## 📁 File Structure

```
ai-lead-scoring-dashboard/
├── app.py                      # Main Streamlit application
├── train_model.py              # ML model training script
├── generate_sample_data.py     # Sample data generation
├── setup.py                    # Automated setup script
├── requirements.txt            # Python dependencies
├── model.pkl                   # Trained model (generated)
├── leads.csv                   # Sample dataset (generated)
└── README.md                   # Readme file
```

### Key Files

- **`app.py`**: Main application with Streamlit UI and business logic
- **`train_model.py`**: Model training pipeline with evaluation
- **`setup.py`**: Automated installation and configuration
- **`requirements.txt`**: All Python package dependencies
- **`leads.csv`**: 10,000 Records of leads

## ⚙️ Configuration

### Environment Variables
```bash
# Optional configurations
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=localhost
MODEL_PATH=model.pkl
DATA_PATH=leads.csv
```

### Model Parameters
```python
# Gradient Boosting configuration
GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)
```

### Model Performance
```
Model Accuracy: 0.9126
Precision: 0.9277
Recall: 0.9555
F1 Score: 0.9414

Classification Report:
              precision    recall  f1-score   support

           0       0.87      0.79      0.83      2652
           1       0.93      0.96      0.94      7348

    accuracy                           0.91     10000
   macro avg       0.90      0.87      0.88     10000
weighted avg       0.91      0.91      0.91     10000


Confusion Matrix:
[[2105  547]
 [ 327 7021]]
```

### UI Customization
- **Theme**: Modify CSS in `app.py` for custom styling
- **Colors**: Update color scheme in the style section
- **Layout**: Adjust column layouts and spacing
- **Fonts**: Change font family imports

## 🚀 Deployment

### Local Development
```bash
streamlit run app.py
```

### Production Deployment

#### Streamlit Cloud
1. Push code to GitHub repository
2. Connect to Streamlit Cloud
3. Deploy with automatic CI/CD


## 🛠️ Development

### Adding New Features
1. **New Input Fields**: Add to sidebar form in `app.py`
2. **Model Updates**: Modify `train_model.py` for new features
3. **UI Enhancements**: Update CSS and layout components
4. **Analytics**: Add new charts and metrics

### Testing
```bash
# Model testing
python -m pytest tests/test_model.py

# UI testing
streamlit run app.py --server.headless=true
```

### Code Quality
- **Linting**: Use flake8 for code style
- **Formatting**: Use black for consistent formatting
- **Type Hints**: Add type annotations for better maintainability

## 📊 Performance Monitoring

### Key Metrics
- **Model Accuracy**: Track prediction performance
- **Response Time**: Monitor API response times
- **User Engagement**: Analyze dashboard usage
- **Data Quality**: Monitor input data quality

### Logging
```python
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
```

## 🔒 Security & Privacy

### Data Protection
- **PII Handling**: Minimal personal data collection
- **Consent Management**: User consent for data processing
- **Data Retention**: Configurable data retention policies
- **Encryption**: Secure data transmission

### Compliance
- **GDPR Ready**: Privacy-by-design implementation
- **Data Minimization**: Collect only necessary data
- **Right to Deletion**: Clear data deletion capabilities

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guide
- Add tests for new features
- Update documentation
- Ensure backward compatibility


### Roadmap
- [ ] Advanced ML models (XGBoost, Neural Networks)
- [ ] Real-time data integration
- [ ] Advanced analytics and reporting
- [ ] Mobile app development
- [ ] API endpoints for external integration

---

**Built with ❤️ using Streamlit, scikit-learn, and modern web technologies**
