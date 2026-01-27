# Interactive ML Playground

An interactive web platform to learn Machine Learning through storytelling and hands-on experimentation.

## Features

- **Explainer Mode**: Step-by-step storytelling that builds intuition about ML algorithms
- **Try It With Data**: Upload your own CSV data, see real Python code, and run models
- **No accounts required**: Everything runs in-memory, refresh resets all
- **Beginner friendly**: Clean UI with technically correct explanations

## Tech Stack

- **Frontend**: Next.js 14 + React + TypeScript + Tailwind CSS
- **Backend**: FastAPI + Python
- **ML**: scikit-learn
- **Charts**: Recharts

## Project Structure

```
interactive-ml-playground/
├── frontend/                 # Next.js application
│   ├── src/
│   │   ├── app/             # Next.js app router pages
│   │   │   ├── page.tsx     # Homepage with model cards
│   │   │   └── models/
│   │   │       └── linear-regression/
│   │   │           └── page.tsx
│   │   ├── components/      # React components
│   │   │   ├── ModelCard.tsx
│   │   │   ├── ExplainerTab.tsx
│   │   │   ├── TryItTab.tsx
│   │   │   ├── CodeBlock.tsx
│   │   │   ├── ScatterPlot.tsx
│   │   │   └── MetricsDisplay.tsx
│   │   └── lib/
│   │       └── api.ts       # API client
│   └── package.json
│
├── backend/                  # FastAPI application
│   ├── app/
│   │   ├── main.py          # FastAPI entry point
│   │   ├── models/
│   │   │   └── linear_regression.py
│   │   ├── schemas/
│   │   │   └── schemas.py
│   │   └── routers/
│   │       └── ml.py        # API endpoints
│   ├── sample_data/
│   │   └── housing_sample.csv
│   └── requirements.txt
│
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.10+
- Node.js 18+
- npm or yarn

### Backend Setup

1. Navigate to the backend directory:
   ```bash
   cd interactive-ml-playground/backend
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Start the FastAPI server:
   ```bash
   uvicorn app.main:app --reload --port 8000
   ```

   The API will be available at `http://localhost:8000`

   API docs at `http://localhost:8000/docs`

### Frontend Setup

1. Open a new terminal and navigate to the frontend directory:
   ```bash
   cd interactive-ml-playground/frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm run dev
   ```

   The app will be available at `http://localhost:3000`

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/ml/models` | GET | List available ML models |
| `/api/ml/upload-csv` | POST | Upload and parse CSV file |
| `/api/ml/parse-csv-text` | POST | Parse CSV from text input |
| `/api/ml/train/linear-regression` | POST | Train a linear regression model |
| `/api/ml/predict/linear-regression` | POST | Make predictions |

## Sample Dataset

A sample housing dataset is included at `backend/sample_data/housing_sample.csv`:

| Column | Description |
|--------|-------------|
| size_sqft | House size in square feet |
| bedrooms | Number of bedrooms |
| age_years | Age of the house in years |
| distance_to_center | Distance to city center in miles |
| price | House price (target variable) |

## Usage

1. **Home Page**: Browse available ML models and click to explore
2. **Explainer Tab**: Read through the interactive story explaining Linear Regression
3. **Try It With Data Tab**:
   - Upload a CSV file or paste CSV data
   - Select feature columns (inputs) and target column (what to predict)
   - Click "Train Model" to see results
   - View metrics, coefficients, predictions chart, and generated Python code

## Development

### Running Tests

Backend:
```bash
cd backend
pytest
```

Frontend:
```bash
cd frontend
npm run lint
```

### Building for Production

Frontend:
```bash
cd frontend
npm run build
npm start
```

Backend:
```bash
cd backend
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Adding New Models

To add a new ML model:

1. Create model trainer in `backend/app/models/`
2. Add schemas in `backend/app/schemas/`
3. Add API routes in `backend/app/routers/ml.py`
4. Update model list in the `/api/ml/models` endpoint
5. Create frontend page at `frontend/src/app/models/{model-name}/`
6. Add Explainer content and TryIt configuration

## Design Principles

- **No user accounts**: Privacy-first, no data stored
- **In-memory only**: Refresh resets everything
- **Beginner friendly**: Clear explanations with visual aids
- **Technically correct**: Real sklearn code, actual metrics
- **Clean UI**: Minimal, focused on learning

## License

MIT
