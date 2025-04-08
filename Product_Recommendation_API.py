from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random


# Initialize FastAPI application
app = FastAPI(
    title="Wholesale Product Recommendation API",
    description="API for generating personalized product recommendations for wholesale customers"
)

# Configuration constants
DATASET_PATHS = {
    'companies': 'wholesale_companies.csv',
    'customers': 'wholesale_customers.csv',
    'transactions': 'wholesale_transactions.csv'
}

PRODUCT_CATALOG = [
    "Organic Apples", "Whole Grain Bread", "Almond Milk",
    "Free-Range Eggs", "Grass-Fed Beef", "Quinoa",
    "Fresh Spinach", "Avocado Oil", "Dark Chocolate",
    "Greek Yogurt", "Oatmeal", "Raw Honey",
    "Fresh Salmon", "Coconut Water", "Kale Chips",
    "Protein Powder", "Mixed Nuts", "Basmati Rice",
    "Spaghetti Pasta", "Extra Virgin Olive Oil"
]


def initialize_data():
    """Load and prepare the datasets."""
    companies = pd.read_csv(DATASET_PATHS['companies'])
    customers = pd.read_csv(DATASET_PATHS['customers'])
    transactions = pd.read_csv(DATASET_PATHS['transactions'])

    # Assign random products to transactions
    transactions['item_name'] = [random.choice(PRODUCT_CATALOG)
                                 for _ in range(len(transactions))]

    # Merge datasets with clear column naming
    merged = (
        transactions.merge(companies, on='company_id')
        .merge(customers, on='customer_id')
        .rename(columns={
            'company_id_x': 'company_id',
            'name_x': 'company_name',
            'location_x': 'company_location',
            'name_y': 'customer_name',
            'location_y': 'customer_location'
        })
    )

    return merged


def build_recommendation_engine(data):
    """Construct the recommendation components from the data."""
    # Create customer purchase profiles
    profiles = data[[
        'customer_id', 'item_name', 'quantity',
        'total_amount', 'inventory_value',
        'business_type', 'credit_limit',
        'customer_location'
    ]].copy()

    # Generate item purchase matrix
    item_matrix = profiles.pivot_table(
        index='customer_id',
        columns='item_name',
        values='quantity',
        fill_value=0
    )

    # Create combined item strings for each customer
    profiles['purchase_history'] = profiles.groupby('customer_id')['item_name'].transform(
        lambda x: ' '.join(x)
    )
    profiles = profiles[['customer_id', 'purchase_history']].drop_duplicates()

    # Calculate customer similarity
    vectorizer = TfidfVectorizer()
    purchase_vectors = vectorizer.fit_transform(profiles['purchase_history'])
    similarity_scores = cosine_similarity(purchase_vectors, purchase_vectors)
    similarity_df = pd.DataFrame(
        similarity_scores,
        index=profiles['customer_id'],
        columns=profiles['customer_id']
    )

    return {
        'item_matrix': item_matrix,
        'similarity_df': similarity_df,
        'full_data': data
    }


# Initialize data and recommendation engine
dataset = initialize_data()
engine = build_recommendation_engine(dataset)


class RecommendationQuery(BaseModel):
    """Request model for recommendation endpoint"""
    customer_id: int
    top_n: Optional[int] = 5
    min_inventory: Optional[int] = 0


@app.get("/", tags=["Root"])
def api_root():
    """Health check endpoint"""
    return {"status": "active", "message": "Recommendation service is running"}


@app.post("/recommendations/", tags=["Recommendations"])
def generate_recommendations(query: RecommendationQuery):
    """Generate product recommendations for a customer"""
    # Validate customer exists
    if query.customer_id not in engine['similarity_df'].index:
        raise HTTPException(
            status_code=404,
            detail=f"No customer found with ID {query.customer_id}"
        )

    # Find similar customers
    similar_customers = (
        engine['similarity_df'][query.customer_id]
        .sort_values(ascending=False)
        .index[1:query.top_n + 1]
    )

    # Calculate recommendation scores
    recommendation_scores = (
        engine['item_matrix']
        .loc[similar_customers]
        .mean(axis=0)
        .sort_values(ascending=False)
    )

    # Filter by inventory availability
    available_products = dataset[
        dataset['inventory_value'] >= query.min_inventory
        ]['item_name'].unique()

    recommendations = [
                          product for product in recommendation_scores.index
                          if product in available_products
                      ][:query.top_n]

    return {
        "customer_id": query.customer_id,
        "recommended_products": recommendations,
        "recommendation_count": len(recommendations)
    }
