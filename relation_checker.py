import streamlit as st
import pandas as pd
from openai import OpenAI
import json

# ---------- CONFIG ----------
st.set_page_config(page_title="Batch Category Classifier", layout="wide")
st.title("Column Relationship Checker")

# Initialize OpenAI client (will use OPENAI_API_KEY from environment)
client = OpenAI()
BATCH_SIZE = 5

# ---------- AI BATCH FUNCTION ----------
def classify_batch(pairs):
    formatted_pairs = "\n".join(
        [f"{i+1}. Category: {c} | Search Category: {s}"
         for i, (c, s) in enumerate(pairs)]
    )
    prompt = f"""
Classify if each category falls within its search category.
For each item return:
- match: "Yes" or "No"
- score: integer from 0 to 100 indicating relevance strength

Return ONLY valid JSON array with exactly {len(pairs)} objects in this format:
[
  {{"match": "Yes", "score": 85}},
  {{"match": "No", "score": 25}}
]

Items:
{formatted_pairs}
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a strict classifier. Return only valid JSON array with no markdown formatting."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        content = response.choices[0].message.content.strip()
        
        # Remove markdown code blocks if present
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()
        
        return json.loads(content)
    except json.JSONDecodeError as e:
        st.error(f"JSON parsing error: {e}")
        st.error(f"Response content: {response.choices[0].message.content}")
        # Return default values for the batch
        return [{"match": "Error", "score": 0} for _ in pairs]
    except Exception as e:
        st.error(f"API error: {e}")
        return [{"match": "Error", "score": 0} for _ in pairs]

# ---------- SIDEBAR (INPUT) ----------
st.sidebar.header("Input Dataset")
uploaded_file = st.sidebar.file_uploader(
    "Upload CSV file",
    type=["csv"]
)

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    # Check if dataframe is empty
    if df.empty:
        st.error("The uploaded CSV file is empty.")
    elif len(df.columns) < 2:
        st.error("The CSV file must have at least 2 columns.")
    else:
        columns = df.columns.tolist()
        
        category_col = st.sidebar.selectbox(
            "Select Column 1",
            columns
        )
        
        search_col = st.sidebar.selectbox(
            "Select Column 2",
            columns,
            index=1 if len(columns) > 1 else 0
        )
        
        # Validate that different columns are selected
        if category_col == search_col:
            st.sidebar.warning("Please select different columns for Category and Search Category.")
        
        run = st.sidebar.button("Run AI Classification")
        
        # ---------- MAIN AREA (OUTPUT) ----------
        st.subheader("Dataset Preview")
        st.dataframe(df.head())
        
        if run and category_col != search_col:
            # Remove rows with missing values in selected columns
            df_clean = df[[category_col, search_col]].dropna()
            
            if len(df_clean) == 0:
                st.error("No valid rows found. Both selected columns contain only missing values.")
            else:
                cache = {}
                results_match = []
                results_score = []
                
                rows = list(df_clean.itertuples(index=False, name=None))
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                with st.spinner("Running batch AI classification..."):
                    i = 0
                    while i < len(rows):
                        batch = rows[i:i + BATCH_SIZE]
                        uncached = []
                        
                        for pair in batch:
                            if pair not in cache:
                                uncached.append(pair)
                        
                        if uncached:
                            status_text.text(f"Processing batch {i//BATCH_SIZE + 1} of {(len(rows)-1)//BATCH_SIZE + 1}...")
                            ai_results = classify_batch(uncached)
                            
                            # Validate response length
                            if len(ai_results) != len(uncached):
                                st.warning(f"Expected {len(uncached)} results but got {len(ai_results)}. Using available results.")
                            
                            for idx, pair in enumerate(uncached):
                                if idx < len(ai_results):
                                    cache[pair] = ai_results[idx]
                                else:
                                    cache[pair] = {"match": "Error", "score": 0}
                        
                        for pair in batch:
                            res = cache[pair]
                            results_match.append(res["match"])
                            results_score.append(res["score"])
                        
                        i += BATCH_SIZE
                        progress_bar.progress(min(i / len(rows), 1.0))
                
                progress_bar.empty()
                status_text.empty()
                
                # Create results dataframe matching original index
                df_results = df.copy()
                df_results["ai_match"] = None
                df_results["ai_relevance_score"] = None
                
                # Map results back to original dataframe
                valid_indices = df_clean.index
                df_results.loc[valid_indices, "ai_match"] = results_match
                df_results.loc[valid_indices, "ai_relevance_score"] = results_score
                
                st.success(f"Classification completed! Processed {len(results_match)} rows with batching + caching.")
                st.dataframe(df_results)
                
                # Statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    match_count = sum(1 for m in results_match if m == "Yes")
                    st.metric("Matches", match_count)
                with col2:
                    no_match_count = sum(1 for m in results_match if m == "No")
                    st.metric("Non-Matches", no_match_count)
                with col3:
                    avg_score = sum(results_score) / len(results_score) if results_score else 0
                    st.metric("Avg Score", f"{avg_score:.1f}")
                
                csv = df_results.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download Result CSV",
                    data=csv,
                    file_name="classified_output.csv",
                    mime="text/csv"
                )
else:
    st.info("Please upload a CSV file to get started.")
