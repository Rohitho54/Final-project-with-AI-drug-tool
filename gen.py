"""
Enhanced Drug Safety Toolkit with Modern Streamlit UI
Streamlit prototype implementing:
1) DDI detection (RxNorm -> DrugBank optional -> local fallback)
2) Age-specific dosage suggestions (mg/kg rules, example table)
3) Alternative medication suggestions (class/ingredient mapping)
4) NLP extraction from free text (spaCy / medspacy example)
5) Enhanced AI Assistant with conversation memory and specialized prompts
6) Modern Streamlit UI with enhanced styling

NOTES:
- This is a demo/prototype. Replace local CSVs and mappings with production datasets/APIs.
- Provide DRUGBANK_API_KEY as an environment variable if you want to use DrugBank (recommended for production).
"""

import os
import itertools
import re
import requests
import streamlit as st
import pandas as pd
from typing import List, Dict, Optional, Any
from rapidfuzz import fuzz, process
import torch
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    AutoModelForCausalLM = None
    AutoTokenizer = None
    set_seed = None
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import google.generativeai as genai
# Removed Google API client as it's no longer used

# ----------------------------
# Page Configuration & Styling
# ----------------------------
st.set_page_config(
    page_title="Drug Safety Toolkit Pro", 
    page_icon="💊", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling (keeping your existing CSS)
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-color: #1f77b4;
        --secondary-color: #ff7f0e;
        --success-color: #2ca02c;
        --warning-color: #d62728;
        --info-color: #17a2b8;
        --light-bg: #f8f9fa;
        --dark-text: #343a40;
    }

    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Custom header styling */
    .main-header {
        background: linear-gradient(90deg, #1f77b4, #17a2b8);
        padding: 2rem 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }

    .main-header p {
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
        opacity: 0.9;
    }

    /* Card styling */
    .card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        border-left: 4px solid var(--primary-color);
        margin-bottom: 1rem;
    }

    .card-danger {
        border-left-color: var(--warning-color);
    }

    .card-success {
        border-left-color: var(--success-color);
    }

    .card-warning {
        border-left-color: var(--secondary-color);
    }

    .card-info {
        border-left-color: var(--info-color);
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem 0;
    }

    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        margin: 0;
    }

    .metric-label {
        font-size: 0.9rem;
        opacity: 0.8;
        margin: 0;
    }

    /* Button styling */
    .stButton > button {
        background: linear-gradient(90deg, var(--primary-color), var(--info-color));
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }

    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }

    /* Alert boxes */
    .alert {
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid;
    }

    .alert-danger {
        background-color: #f8d7da;
        border-left-color: var(--warning-color);
        color: #721c24;
    }

    .alert-success {
        background-color: #d4edda;
        border-left-color: var(--success-color);
        color: #155724;
    }

    .alert-warning {
        background-color: #fff3cd;
        border-left-color: var(--secondary-color);
        color: #856404;
    }

    .alert-info {
        background-color: #d1ecf1;
        border-left-color: var(--info-color);
        color: #0c5460;
    }

    /* Chat message styling */
    .chat-message {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 10px;
        max-width: 80%;
    }

    .chat-user {
        background: linear-gradient(135deg, #e3f2fd, #bbdefb);
        margin-left: auto;
        text-align: right;
    }

    .chat-assistant {
        background: linear-gradient(135deg, #f3e5f5, #e1bee7);
        margin-right: auto;
    }

    /* Data display styling */
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }

    /* Progress bars */
    .progress-bar {
        background: linear-gradient(90deg, var(--success-color), var(--primary-color));
        height: 10px;
        border-radius: 5px;
        margin: 1rem 0;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }

    .stTabs [data-baseweb="tab"] {
        background: white;
        border-radius: 10px 10px 0 0;
        padding: 1rem 2rem;
        font-weight: 600;
    }
</style>
    """, unsafe_allow_html=True)

def create_metric_card(title, value, icon=""):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{icon} {value}</div>
        <div class="metric-label">{title}</div>
    </div>
    """, unsafe_allow_html=True)

def create_alert(message, alert_type="info"):
    st.markdown(f"""
    <div class="alert alert-{alert_type}">
        {message}
    </div>
    """, unsafe_allow_html=True)

def create_severity_chart(interactions_df):
    if interactions_df.empty:
        return None
    
    severity_counts = interactions_df['severity'].value_counts()
    
    fig = px.pie(
        values=severity_counts.values,
        names=severity_counts.index,
        title="Drug Interaction Severity Distribution",
        color_discrete_map={
            'High': '#d62728',
            'Medium': '#ff7f0e', 
            'Low': '#2ca02c'
        }
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    return fig

def display_conversation_history():
    """Display conversation history in a chat-like interface"""
    if st.session_state.conversation_history:
        st.markdown("### 💬 Conversation History")
        
        for i, conv in enumerate(st.session_state.conversation_history[-5:]):  # Show last 5 conversations
            # User message
            st.markdown(f"""
            <div class="chat-message chat-user">
                <strong>You ({conv['timestamp']}):</strong><br>
                {conv['user']}
            </div>
            """, unsafe_allow_html=True)
            
            # Assistant message
            st.markdown(f"""
            <div class="chat-message chat-assistant">
                <strong>🤖 AI Assistant:</strong><br>
                {conv['assistant']}
            </div>
            """, unsafe_allow_html=True)

# ----------------------------
# Main Application
# ----------------------------
def main():
    # Initialize conversation history
    initialize_conversation_history()
    
    create_header()
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🔧 Quick Actions")
        st.markdown("---")
        
        # System Status
        st.markdown("### 📊 System Status")
        
        # Check API status
        drugbank_status = "✅ Connected" if DRUGBANK_API_KEY else "❌ Not Connected"
        st.markdown(f"**DrugBank API:** {drugbank_status}")
        st.markdown(f"**AI Model:** ✅ Loaded ({device.upper()})")
        st.markdown(f"**RxNorm API:** ✅ Available")
        
        st.markdown("---")
        
        # Recent activity
        st.markdown("### 📈 Quick Stats")
        create_metric_card("Available Drugs", len(medications), "💊")
        create_metric_card("API Endpoints", "3", "🔗")
        create_metric_card("Conversations", len(st.session_state.conversation_history), "💬")
        create_metric_card("Session Time", f"{datetime.now().strftime('%H:%M')}", "⏰")
        
        # AI Context Management
        st.markdown("---")
        st.markdown("### 🤖 AI Context")
        
        # Clear conversation history
        if st.button("🗑️ Clear Chat History"):
            st.session_state.conversation_history = []
            st.session_state.ai_context = {'current_medications': [], 'patient_info': {}, 'recent_interactions': []}
            st.success("Chat history cleared!")
        
        # Display current context
        if st.session_state.ai_context.get('current_medications'):
            st.markdown("**Current Meds:**")
            for med in st.session_state.ai_context['current_medications'][:3]:
                st.markdown(f"• {med}")
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.info("This is a prototype tool for educational purposes. Always consult healthcare professionals for medical decisions.")

    # Main content area with tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔍 Drug Interactions", 
        "💊 Dosage Calculator", 
        "🔄 Alternatives", 
        "📝 NLP Extraction", 
        "🤖 AI Assistant"
    ])

    # Tab 1: Drug Interactions (keeping existing implementation)
    with tab1:
        st.markdown("### 🔍 Drug Interaction Checker")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            <div class="card">
                <h4>Enter Medications</h4>
                <p>Enter medications separated by commas to check for potential interactions.</p>
            </div>
            """, unsafe_allow_html=True)
            
            drug_input = st.text_area(
                "List medications (comma separated)", 
                value="warfarin, aspirin, atorvastatin",
                height=100,
                help="Enter drug names separated by commas. Example: aspirin, warfarin, metformin"
            )
            
            check_interactions = st.button("🔍 Check Interactions", type="primary")
            
        with col2:
            st.markdown("""
            <div class="card card-info">
                <h4>💡 Tips</h4>
                <ul>
                    <li>Use generic drug names</li>
                    <li>Check spelling carefully</li>
                    <li>Include all current medications</li>
                    <li>Consider supplements too</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        if check_interactions:
            drug_list = [d.strip() for d in drug_input.split(",") if d.strip()]
            
            # Update AI context
            st.session_state.ai_context['current_medications'] = drug_list
            
            if not drug_list:
                create_alert("Please enter at least one medication.", "warning")
            else:
                with st.spinner("🔄 Analyzing medications and checking interactions..."):
                    # Progress bar
                    progress = st.progress(0)
                    progress.progress(25)
                    
                    # Normalize drugs
                    norm_map = {}
                    for i, drug in enumerate(drug_list):
                        norm_map[drug] = normalize_to_rxcui(drug)
                        progress.progress(25 + (i + 1) * 25 // len(drug_list))
                    
                    progress.progress(50)
                    
                    # Display normalization results
                    st.markdown("### 📋 Drug Normalization Results")
                    
                    norm_col1, norm_col2 = st.columns(2)
                    
                    with norm_col1:
                        st.markdown("#### ✅ Successfully Identified")
                        found_drugs = []
                        for drug, data in norm_map.items():
                            if data and "rxcui" in data:
                                found_drugs.append({
                                    "Drug": drug.title(),
                                    "RxCUI": data["rxcui"],
                                    "Status": "✅ Found"
                                })
                        
                        if found_drugs:
                            st.dataframe(pd.DataFrame(found_drugs), use_container_width=True)
                        else:
                            st.info("No drugs were successfully normalized.")
                    
                    with norm_col2:
                        st.markdown("#### ❌ Not Found")
                        not_found = [drug.title() for drug, data in norm_map.items() if not data]
                        if not_found:
                            for drug in not_found:
                                st.markdown(f"• {drug}")
                        else:
                            st.success("All drugs were successfully identified!")
                    
                    progress.progress(75)
                    
                    # Check interactions
                    rxcuis = [v["rxcui"] for v in norm_map.values() if v and "rxcui" in v]
                    interactions = check_interactions_drugbank(rxcuis) if rxcuis else []
                    
                    progress.progress(90)
                    
                    if interactions:
                        st.markdown("### ⚠️ Drug Interactions Found")
                        interactions_df = pd.DataFrame(interactions)
                        
                        # Update AI context
                        st.session_state.ai_context['recent_interactions'] = interactions
                        
                        # Create severity chart
                        severity_chart = create_severity_chart(interactions_df)
                        if severity_chart:
                            st.plotly_chart(severity_chart, use_container_width=True)
                        
                        # Display interactions table
                        st.dataframe(
                            interactions_df.style.applymap(
                                lambda x: 'background-color: #ffebee' if x == 'High' else
                                'background-color: #fff3e0' if x == 'Medium' else
                                'background-color: #e8f5e8' if x == 'Low' else '',
                                subset=['severity']
                            ),
                            use_container_width=True
                        )
                    else:
                        st.success("✅ No interactions found via DrugBank API!")
                        
                        # Try local DDI database
                        create_alert("Checking local database for additional interactions...", "info")
                        local_records = load_local_ddi("local_ddi.csv")
                        
                        if local_records:
                            found = find_ddi_local(drug_list, local_records)
                            if found:
                                st.markdown("### 📋 Local Database Interactions")
                                local_df = pd.DataFrame(found)
                                st.dataframe(local_df, use_container_width=True)
                            else:
                                create_alert("No additional interactions found in local database.", "success")
                        else:
                            create_alert("Local DDI database not available. Create 'local_ddi.csv' for extended checking.", "info")
                    
                    progress.progress(100)
                    st.success("✅ Analysis complete!")

    # Tab 2: Dosage Calculator (keeping existing implementation)
    with tab2:
        st.markdown("### 💊 Age-Specific Dosage Calculator")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("""
            <div class="card">
                <h4>Patient Information</h4>
                <p>Enter patient details for personalized dosage recommendations.</p>
            </div>
            """, unsafe_allow_html=True)
            
            age = st.number_input(
                "Patient age (years)", 
                min_value=0.0, 
                value=30.0, 
                step=0.1,
                help="Enter the patient's age in years"
            )
            
            weight = st.number_input(
                "Patient weight (kg)", 
                min_value=0.0, 
                value=0.0, 
                step=0.1,
                help="Enter weight in kg, or leave as 0 for automatic estimation"
            )
            
            selected_drug = st.selectbox(
                "Select medication", 
                options=list(medications.keys()),
                help="Choose from available medications with dosing rules"
            )
            
            calculate_dose_btn = st.button("📊 Calculate Dosage", type="primary")
            
        with col2:
            st.markdown("""
            <div class="card card-warning">
                <h4>⚠️ Important Notes</h4>
                <ul>
                    <li>These are reference dosages only</li>
                    <li>Always consult healthcare professionals</li>
                    <li>Consider patient-specific factors</li>
                    <li>Check for contraindications</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            # Age group indicator
            if age < 1:
                age_group = "👶 Infant"
                group_color = "#e3f2fd"
            elif age < 12:
                age_group = "🧒 Child"
                group_color = "#f3e5f5"
            elif age < 18:
                age_group = "👦 Adolescent"
                group_color = "#e8f5e8"
            elif age < 65:
                age_group = "👨 Adult"
                group_color = "#fff3e0"
            else:
                age_group = "👴 Elderly"
                group_color = "#ffebee"
            
            st.markdown(f"""
            <div class="card" style="background-color: {group_color};">
                <h4>Age Group Classification</h4>
                <p style="font-size: 1.2em; font-weight: bold;">{age_group}</p>
            </div>
            """, unsafe_allow_html=True)

        if calculate_dose_btn:
            w = weight if weight > 0 else None
            
            # Update AI context
            st.session_state.ai_context['patient_info'] = {'age': age, 'weight': weight, 'drug': selected_drug}
            
            with st.spinner("🔄 Calculating personalized dosage..."):
                rec = recommend_dose(selected_drug, age, weight_kg=w)
                
                if "error" in rec:
                    create_alert(rec["error"], "danger")
                else:
                    st.markdown("### 📊 Dosage Recommendation")
                    
                    # Create result cards
                    result_col1, result_col2, result_col3 = st.columns(3)
                    
                    with result_col1:
                        create_metric_card("Recommended Dose", rec.get("dosage", "N/A"), "💊")
                    
                    with result_col2:
                        create_metric_card("Maximum Daily", rec.get("max_per_day", "N/A"), "⚠️")
                    
                    with result_col3:
                        estimated_weight = rec.get("weight_kg", "N/A")
                        weight_label = f"{estimated_weight} kg" if estimated_weight != "N/A" else "N/A"
                        if w is None and estimated_weight != "N/A":
                            weight_label += " (est.)"
                        create_metric_card("Patient Weight", weight_label, "⚖️")
                    
                    # Detailed information
                    st.markdown("### 📋 Detailed Information")
                    
                    detail_data = {
                        "Parameter": ["Drug Name", "Patient Age", "Weight Used", "Recommended Dose", "Maximum Daily Dose"],
                        "Value": [
                            rec.get("drug", "N/A").title(),
                            f"{rec.get('age_years', 'N/A')} years",
                            f"{rec.get('weight_kg', 'N/A')} kg" + (" (estimated)" if w is None else ""),
                            rec.get("dosage", "N/A"),
                            rec.get("max_per_day", "N/A")
                        ]
                    }
                    
                    st.dataframe(pd.DataFrame(detail_data), use_container_width=True, hide_index=True)
                    
                    create_alert("⚠️ This dosage is for reference only. Always verify with current prescribing guidelines and consider patient-specific factors.", "warning")

    # Tab 3: Alternative Medications (keeping existing implementation)
    with tab3:
        st.markdown("### 🔄 Alternative Medication Finder")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            <div class="card">
                <h4>Find Therapeutic Alternatives</h4>
                <p>Search for alternative medications in the same therapeutic class or with similar effects.</p>
            </div>
            """, unsafe_allow_html=True)
            
            drug_for_alt = st.text_input(
                "Drug to find alternatives for", 
                value="warfarin",
                help="Enter the medication name to find alternatives"
            )
            
            find_alternatives_btn = st.button("🔍 Find Alternatives", type="primary")
        
        with col2:
            st.markdown("""
            <div class="card card-info">
                <h4>💡 Why Find Alternatives?</h4>
                <ul>
                    <li>Drug shortages</li>
                    <li>Cost considerations</li>
                    <li>Side effect profile</li>
                    <li>Drug interactions</li>
                    <li>Patient preferences</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        if find_alternatives_btn:
            alts = suggest_alternatives(drug_for_alt)
            
            if alts:
                st.markdown("### ✅ Alternative Medications Found")
                
                # Create cards for each alternative
                alt_cols = st.columns(min(len(alts), 3))
                
                for i, alt in enumerate(alts):
                    with alt_cols[i % 3]:
                        st.markdown(f"""
                        <div class="card card-success">
                            <h4>💊 {alt.title()}</h4>
                            <p><strong>Alternative to:</strong> {drug_for_alt.title()}</p>
                            <p><small>Consult healthcare provider before switching</small></p>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Create comparison table
                comparison_data = {
                    "Original Drug": [drug_for_alt.title()],
                    "Alternatives": [", ".join([alt.title() for alt in alts])],
                    "Total Options": [len(alts)]
                }
                
                st.markdown("### 📊 Summary")
                st.dataframe(pd.DataFrame(comparison_data), use_container_width=True, hide_index=True)
                
            else:
                create_alert(f"No alternatives found for '{drug_for_alt}' in the current database. Consider expanding the ALTERNATIVE_MAP or connecting to a comprehensive formulary API.", "warning")

    # Tab 4: NLP Extraction (keeping existing implementation)
    with tab4:
        st.markdown("### 📝 Natural Language Processing - Medication Extraction")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            <div class="card">
                <h4>Extract Medications from Text</h4>
                <p>Paste clinical notes, prescriptions, or medication lists to automatically extract drug information.</p>
            </div>
            """, unsafe_allow_html=True)
            
            sample_texts = {
                "Prescription Example": "Start paracetamol 500 mg PO every 6 hours and aspirin 75 mg daily for cardioprotection. Consider omeprazole 20 mg once daily if GI upset occurs.",
                "Discharge Summary": "Continue home medications: metformin 1000 mg twice daily, lisinopril 10 mg once daily, and atorvastatin 40 mg at bedtime.",
                "Clinical Note": "Patient reports taking ibuprofen 400 mg as needed for joint pain and levothyroxine 100 mcg every morning on empty stomach."
            }
            
            selected_example = st.selectbox("Choose example or enter custom text:", ["Custom"] + list(sample_texts.keys()))
            
            if selected_example == "Custom":
                default_text = "Enter your clinical text here..."
            else:
                default_text = sample_texts[selected_example]
            
            clinical_text = st.text_area(
                "Clinical/prescription text", 
                value=default_text,
                height=150,
                help="Enter or paste clinical text containing medication information"
            )
            
            extract_meds_btn = st.button("🔍 Extract Medications", type="primary")
        
        with col2:
            st.markdown("""
            <div class="card card-info">
                <h4>🤖 NLP Capabilities</h4>
                <ul>
                    <li>Drug name recognition</li>
                    <li>Dosage extraction</li>
                    <li>Route identification</li>
                    <li>Frequency detection</li>
                    <li>Instructions parsing</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="card card-warning">
                <h4>⚠️ Current Limitations</h4>
                <ul>
                    <li>Demo-level accuracy</li>
                    <li>Basic pattern matching</li>
                    <li>No clinical context</li>
                    <li>Limited drug database</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        if extract_meds_btn:
            with st.spinner("🔄 Analyzing text and extracting medication information..."):
                extraction_result = extract_med_info(clinical_text)
                
                if "error" in extraction_result:
                    create_alert(extraction_result["error"], "danger")
                else:
                    st.markdown("### 📊 Extraction Results")
                    
                    extracted_meds = extraction_result.get("extracted_medications", [])
                    
                    if extracted_meds:
                        # Update AI context
                        med_names = [med["name"] for med in extracted_meds]
                        st.session_state.ai_context['current_medications'].extend(med_names)
                        
                        # Summary metrics
                        metric_col1, metric_col2, metric_col3 = st.columns(3)
                        
                        with metric_col1:
                            create_metric_card("Medications Found", str(len(extracted_meds)), "💊")
                        
                        with metric_col2:
                            doses_found = sum(1 for med in extracted_meds if med.get("dose"))
                            create_metric_card("Doses Identified", str(doses_found), "📏")
                        
                        with metric_col3:
                            unique_names = len(set(med["name"].lower() for med in extracted_meds))
                            create_metric_card("Unique Drugs", str(unique_names), "🔢")
                        
                        # Detailed results table
                        st.markdown("### 📋 Detailed Results")
                        
                        results_df = pd.DataFrame(extracted_meds)
                        results_df.index = range(1, len(results_df) + 1)
                        results_df.columns = [col.title() for col in results_df.columns]
                        
                        st.dataframe(results_df, use_container_width=True)
                        
                        # Visualization
                        if len(extracted_meds) > 1:
                            st.markdown("### 📊 Medication Distribution")
                            
                            med_names = [med["name"] for med in extracted_meds]
                            name_counts = pd.Series(med_names).value_counts()
                            
                            fig = px.bar(
                                x=name_counts.index,
                                y=name_counts.values,
                                title="Medication Frequency in Text",
                                labels={"x": "Medication", "y": "Mentions"}
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                    else:
                        create_alert("No medications were extracted from the provided text. Try using more specific medication names with dosages.", "warning")
                    
                    if extraction_result.get("note"):
                        create_alert(extraction_result["note"], "info")

    # Tab 5: Enhanced AI Assistant
    with tab5:
        st.markdown("### 🤖 AI-Powered Prescription Assistant")
        
        st.markdown("""
        <div class="card">
            <h4>Interactive AI Assistant with Memory</h4>
            <p>Ask questions about medications, dosages, interactions, or get prescription guidance. The AI remembers your previous questions and medication context.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Create columns for layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Custom query input
            ai_query = st.text_area(
                "Your question:",
                value=st.session_state.get('ai_query', ''),
                height=100,
                help="Enter your medication-related question here. The AI will remember previous context."
            )
            
            # Input modes
            input_col1, input_col2 = st.columns(2)
            
            with input_col1:
                generate_response_btn = st.button("🚀 Get AI Response", type="primary", use_container_width=True)
            
            with input_col2:
                use_context = st.checkbox("Use conversation context", value=True, help="Include previous conversation in AI response")
        
        with col2:
            st.markdown("""
            <div class="card card-info">
                <h4>🎯 AI Features</h4>
                <ul>
                    <li>Context-aware responses</li>
                    <li>Conversation memory</li>
                    <li>Medical knowledge base</li>
                    <li>Safety prioritization</li>
                    <li>Follow-up suggestions</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

            # Display current context
            if st.session_state.ai_context.get('current_medications'):
                st.markdown("""
                <div class="card card-success">
                    <h4>📋 Current Context</h4>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("**Medications in context:**")
                for med in st.session_state.ai_context['current_medications'][-5:]:
                    st.markdown(f"• {med}")

                if st.session_state.ai_context.get('patient_info'):
                    patient = st.session_state.ai_context['patient_info']
                    st.markdown("**Patient info:**")
                    st.markdown(f"• Age: {patient.get('age', 'N/A')} years")
                    if patient.get('weight', 0) > 0:
                        st.markdown(f"• Weight: {patient.get('weight')} kg")

            st.markdown("""
            <div class="card card-warning">
                <h4>⚠️ AI Disclaimer</h4>
                <p>AI responses are for educational purposes only. Always consult healthcare professionals for medical decisions.</p>
            </div>
            """, unsafe_allow_html=True)

        # Handle AI query
        if generate_response_btn and ai_query.strip():
            with st.spinner("🤖 AI is thinking and generating response..."):
                # Extract medical entities from query
                entities = extract_medical_entities(ai_query)
                if entities['medications']:
                    st.session_state.ai_context['current_medications'].extend(entities['medications'])
                    # Remove duplicates while preserving order
                    st.session_state.ai_context['current_medications'] = list(dict.fromkeys(st.session_state.ai_context['current_medications']))
                
                # Generate response with or without context
                context = st.session_state.ai_context if use_context else None
                ai_response = generate_enhanced_ai_response(ai_query, context)
                
                # Add to conversation history
                add_to_conversation(ai_query, ai_response)
                
                # Display the response
                st.markdown("### 🤖 AI Response")
                
                st.markdown(f"""
                <div class="chat-message chat-assistant">
                    <strong>🤖 AI Assistant:</strong><br>
                    {ai_response}
                </div>
                """, unsafe_allow_html=True)
                
                # Generate follow-up questions
                follow_ups = suggest_follow_up_questions(ai_query, ai_response)
                
                if follow_ups:
                    st.markdown("### 🔄 Suggested Follow-up Questions")
                    
                    follow_col1, follow_col2, follow_col3 = st.columns(3)
                    cols = [follow_col1, follow_col2, follow_col3]
                    
                    for i, follow_up in enumerate(follow_ups):
                        with cols[i % 3]:
                            if st.button(f"❓ {follow_up}", key=f"followup_{i}", help=f"Ask: {follow_up}"):
                                st.session_state.ai_query = follow_up
                                st.experimental_rerun()
                
                # Safety disclaimer for each response
                create_alert("⚠️ This AI-generated response is for informational purposes only. Always consult with healthcare professionals before making medical decisions.", "warning")
                
                # Clear the query for next question
                if 'ai_query' in st.session_state:
                    del st.session_state.ai_query
        
        elif generate_response_btn and not ai_query.strip():
            create_alert("Please enter a question.", "warning")

# ----------------------------
# Config / constants & sources
# ----------------------------
RXNORM_BASE = "https://rxnav.nlm.nih.gov/REST"
DRUGBANK_BASE = "https://api.drugbank.com/v1"   # requires account/key
DRUGBANK_API_KEY = os.getenv("DRUGBANK_API_KEY")  # set in environment if available
GEMINI_API_KEY = None  # Gemini API key removed


# if GEMINI_API_KEY:
#     genai.configure(api_key=GEMINI_API_KEY)

# Enhanced AI Assistant Configuration
AI_SYSTEM_PROMPT = """You are a specialized medical AI assistant for the Drug Safety Toolkit Pro. Your role is to provide accurate, evidence-based information about medications while prioritizing patient safety.

Key capabilities:
- Medication dosing and safety information
- Drug interaction analysis
- Clinical pharmacology explanations
- Age-specific dosing recommendations
- Contraindications and warnings

Important guidelines:
1. Always emphasize that your responses are for educational purposes only
2. Recommend consulting healthcare professionals for medical decisions
3. Provide specific, actionable information when available
4. Flag potential safety concerns prominently
5. Be clear about limitations and uncertainties
6. Use evidence-based information from reputable sources

Format your responses clearly with appropriate sections and bullet points when helpful."""

@st.cache_resource
def load_model():
    """Load the AI model with enhanced configuration"""
    if not TRANSFORMERS_AVAILABLE:
        return None, None, "cpu"
    try:
        model_path = "ibm-granite/granite-3.3-2b-instruct"
        device = "cpu"  # Force CPU usage to avoid CUDA issues
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,  # Use float32 for CPU
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        return model, tokenizer, device
    except Exception as e:
        st.error(f"Failed to load AI model: {str(e)}")
        return None, None, "cpu"

# Load model with spinner
with st.spinner("Loading AI model... Please wait."):
    model, tokenizer, device = load_model()

# Make model variables globally accessible
MODEL = model
TOKENIZER = tokenizer
DEVICE = device

# ----------------------------
# Enhanced AI Assistant Functions
# ----------------------------
def initialize_conversation_history():
    """Initialize conversation history in session state"""
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'ai_context' not in st.session_state:
        st.session_state.ai_context = {
            'current_medications': [],
            'patient_info': {},
            'recent_interactions': []
        }

def add_to_conversation(user_input: str, ai_response: str):
    """Add conversation to history"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.conversation_history.append({
        'timestamp': timestamp,
        'user': user_input,
        'assistant': ai_response
    })
    
    # Keep only last 10 conversations to manage memory
    if len(st.session_state.conversation_history) > 10:
        st.session_state.conversation_history = st.session_state.conversation_history[-10:]

def generate_enhanced_ai_response(query: str, context: Dict = None) -> str:
    """Generate AI response with enhanced prompting and context awareness"""

    # Build context-aware prompt
    context_info = ""
    if context and st.session_state.ai_context:
        if st.session_state.ai_context.get('current_medications'):
            context_info += f"Current medications in context: {', '.join(st.session_state.ai_context['current_medications'])}\n"
        if st.session_state.ai_context.get('patient_info'):
            context_info += f"Patient information: {st.session_state.ai_context['patient_info']}\n"

    # Include recent conversation context
    conversation_context = ""
    if len(st.session_state.conversation_history) > 0:
        recent_conv = st.session_state.conversation_history[-2:]  # Last 2 exchanges
        for conv in recent_conv:
            conversation_context += f"Previous Q: {conv['user'][:100]}...\nPrevious A: {conv['assistant'][:200]}...\n"

    # Get online search results for medical queries
    online_search_results = ""
    query_lower = query.lower()
    # Check if query is medical-related and worth searching online
    medical_keywords = ['dosage', 'dose', 'interaction', 'side effect', 'contraindication', 'medication', 'drug', 'treatment', 'symptom']
    if any(keyword in query_lower for keyword in medical_keywords):
        try:
            online_search_results = search_online(query)
            if online_search_results and "Online search not available" not in online_search_results:
                online_search_results = f"\nAdditional Online Information:\n{online_search_results}\n"
        except Exception as e:
            online_search_results = f"\nNote: Online search temporarily unavailable: {str(e)}\n"

    # Enhanced prompt with medical focus
    enhanced_prompt = f"""{AI_SYSTEM_PROMPT}

{context_info}

{conversation_context}

{online_search_results}

Current Question: {query}

Please provide a comprehensive response that includes:
1. Direct answer to the question
2. Safety considerations if applicable
3. Clinical context and relevance
4. Recommendations for follow-up or consultation
5. Reference any online information provided above when relevant

Response:"""

    # Try Gemini first if available
    if GEMINI_API_KEY:
        try:
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content(enhanced_prompt)
            if response and response.text:
                return response.text.strip()
        except Exception as e:
            st.warning(f"Gemini API error: {str(e)}. Falling back to local model.")

    # Fall back to local model if available
    if MODEL and TOKENIZER and TRANSFORMERS_AVAILABLE:
        try:
            # Prepare conversation for the model
            conv = [{"role": "user", "content": enhanced_prompt}]

            # Generate response
            inputs = TOKENIZER.apply_chat_template(conv, return_tensors="pt", add_generation_prompt=True).to(DEVICE)
            set_seed(42)

            with torch.no_grad():
                output = MODEL.generate(
                    input_ids=inputs,
                    max_new_tokens=800,  # Increased for more comprehensive responses
                    temperature=0.6,     # Slightly lower for more focused responses
                    do_sample=True,
                    pad_token_id=TOKENIZER.eos_token_id,
                    repetition_penalty=1.1,
                    top_p=0.9
                )

            input_length = inputs.shape[1]
            generated_tokens = output[0][input_length:]
            response = TOKENIZER.decode(generated_tokens, skip_special_tokens=True)

            # Post-process response
            response = response.strip()
            if len(response) == 0:
                response = "I apologize, but I couldn't generate a response for that query. Please try rephrasing your question or provide more specific details."

            return response

        except Exception as e:
            return f"Error generating response with local model: {str(e)}. Please try again with a different question."

    # If no AI models are available
    return "AI models are not available. Please ensure you have either Gemini API key set or transformers library installed with a compatible model."

def extract_medical_entities(text: str) -> Dict:
    """Extract medical entities from user input to build context"""
    entities = {
        'medications': [],
        'dosages': [],
        'conditions': [],
        'ages': []
    }
    
    # Simple regex patterns for entity extraction
    med_pattern = r'\b(?:paracetamol|ibuprofen|aspirin|warfarin|metformin|omeprazole|atorvastatin|lisinopril|amoxicillin|azithromycin|prednisone|ciprofloxacin|levothyroxine)\b'
    dosage_pattern = r'\b\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml|units?)\b'
    age_pattern = r'\b\d+(?:\.\d+)?\s*(?:year|yr|month|mo)s?(?:\s+old)?\b'
    
    entities['medications'] = re.findall(med_pattern, text, re.IGNORECASE)
    entities['dosages'] = re.findall(dosage_pattern, text, re.IGNORECASE)
    entities['ages'] = re.findall(age_pattern, text, re.IGNORECASE)
    
    return entities

def suggest_follow_up_questions(query: str, response: str) -> List[str]:
    """Generate relevant follow-up questions based on the conversation"""
    follow_ups = []

    query_lower = query.lower()

    if 'dosage' in query_lower or 'dose' in query_lower:
        follow_ups.extend([
            "What are the side effects of this medication?",
            "Are there any drug interactions I should be aware of?",
            "How should this medication be stored?"
        ])

    if 'interaction' in query_lower:
        follow_ups.extend([
            "What are safer alternatives to these medications?",
            "How can I monitor for interaction symptoms?",
            "Should the timing of doses be adjusted?"
        ])

    if 'child' in query_lower or 'pediatric' in query_lower:
        follow_ups.extend([
            "What are the weight-based dosing calculations?",
            "Are there any special considerations for children?",
            "What formulations are available for children?"
        ])

    return follow_ups[:3]  # Return max 3 suggestions

def search_online(query: str) -> str:
    """Search online using Gemini API for medical information"""
    if not GEMINI_API_KEY:
        return "Gemini API key not configured. Please set GEMINI_API_KEY environment variable."

    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        search_prompt = f"Please provide accurate, up-to-date medical information for the following query: {query}. Include relevant drug information, dosages, interactions, or medical facts from reliable sources."
        response = model.generate_content(search_prompt)
        if response and response.text:
            return f"Gemini AI Search Results:\n{response.text.strip()}"
        else:
            return "No information found from Gemini API."
    except Exception as e:
        return f"Error searching with Gemini API: {str(e)}"



# ----------------------------
# All existing utility functions (keeping them unchanged)
# ----------------------------
def fetch_drug_properties(rxcui: str) -> Optional[Dict]:
    """ Fetch properties for a given RxCUI from RxNorm. """
    try:
        resp = requests.get(f"{RXNORM_BASE}/rxcui/{rxcui}/properties.json", timeout=6)
        resp.raise_for_status()
        j = resp.json()
        return j.get("properties", {})
    except Exception:
        return None

def fetch_related_concepts(rxcui: str) -> Optional[Dict]:
    """ Fetch related concepts for a given RxCUI from RxNorm. """
    try:
        resp = requests.get(f"{RXNORM_BASE}/rxcui/{rxcui}/related.json", timeout=6)
        resp.raise_for_status()
        j = resp.json()
        return j.get("relatedGroup", {})
    except Exception:
        return None

def normalize_to_rxcui(name: str) -> Optional[Dict]:
    """ Map free-text drug name to RxCUI and fetch full data using RxNorm REST API. """
    try:
        resp = requests.get(f"{RXNORM_BASE}/rxcui.json", params={"name": name, "search": 1}, timeout=6)
        resp.raise_for_status()
        j = resp.json()
        ids = j.get("idGroup", {}).get("rxnormId")
        if ids:
            rxcui = str(ids[0])
            properties = fetch_drug_properties(rxcui)
            related = fetch_related_concepts(rxcui)
            return {"rxcui": rxcui, "properties": properties, "related": related}
    except Exception as e:
        pass
    return None

def check_interactions_drugbank(rxcui_list: List[str]) -> List[Dict]:
    """Example flow to call DrugBank Clinical API's interaction checker."""
    if not DRUGBANK_API_KEY:
        return []
    headers = {"Authorization": DRUGBANK_API_KEY, "Accept": "application/json"}
    try:
        url = f"{DRUGBANK_BASE}/drug_interactions"
        payload = {"product_concept_ids": rxcui_list}
        r = requests.post(url, headers=headers, json=payload, timeout=12)
        r.raise_for_status()
        j = r.json()
        interactions = []
        for it in j.get("interactions", []):
            interactions.append({
                "drug_a": it.get("drug_a"),
                "drug_b": it.get("drug_b"),
                "severity": it.get("severity"),
                "description": it.get("description"),
                "source": "DrugBank"
            })
        return interactions
    except Exception as e:
        return []

def load_local_ddi(path="local_ddi.csv"):
    """Expects CSV with columns: drug_a, drug_b, severity, description, source"""
    if not os.path.exists(path):
        return []
    df = pd.read_csv(path)
    return df.to_dict(orient="records")

def find_ddi_local(drug_names: List[str], local_records: List[Dict], fuzz_threshold: int = 85):
    db_drugs = set()
    for r in local_records:
        db_drugs.add(str(r.get("drug_a","")).lower())
        db_drugs.add(str(r.get("drug_b","")).lower())
    db_list = [d for d in db_drugs if d]

    mapping = {}
    for d in drug_names:
        res = process.extractOne(d.lower(), db_list, scorer=fuzz.token_sort_ratio)
        if res and res[1] >= fuzz_threshold:
            mapping[d] = res[0]
        else:
            mapping[d] = None

    found = []
    for a,b in itertools.combinations(drug_names, 2):
        ma, mb = mapping.get(a), mapping.get(b)
        if not ma or not mb:
            continue
        for rec in local_records:
            ra = str(rec.get("drug_a","")).lower()
            rb = str(rec.get("drug_b","")).lower()
            if (ra == ma and rb == mb) or (ra == mb and rb == ma):
                found.append({
                    "drug_a": a, "drug_b": b,
                    "matched_a": ma, "matched_b": mb,
                    "severity": rec.get("severity"),
                    "description": rec.get("description"),
                    "source": rec.get("source","local")
                })
    return found

# Medications data (keeping existing)
medications = {
    "paracetamol": [
        {"age_min": 0,  "age_max": 1,  "dose": "10 mg/kg every 6h", "max": "40 mg/kg/day"},
        {"age_min": 1,  "age_max": 12, "dose": "15 mg/kg every 6h", "max": "60 mg/kg/day"},
        {"age_min": 12, "age_max": 200,"dose": "500–1000 mg every 6h", "max": "4 g/day"},
    ],
    "ibuprofen": [
        {"age_min": 0.5,"age_max": 1,  "dose": "5 mg/kg every 8h", "max": "40 mg/kg/day"},
        {"age_min": 1,  "age_max": 12, "dose": "10 mg/kg every 6–8h", "max": "40 mg/kg/day"},
        {"age_min": 12, "age_max": 200,"dose": "200–400 mg every 6h", "max": "1200 mg/day (OTC)"},
    ],
    "amoxicillin": [
        {"age_min": 0,  "age_max": 1,  "dose": "20 mg/kg/day divided 3 doses", "max": "30 mg/kg/day"},
        {"age_min": 1,  "age_max": 12, "dose": "25–50 mg/kg/day in 2–3 doses", "max": "90 mg/kg/day"},
        {"age_min": 12, "age_max": 200,"dose": "500 mg every 8h", "max": "3 g/day"},
    ],
    "azithromycin": [
        {"age_min": 1,  "age_max": 12, "dose": "10 mg/kg once daily", "max": "500 mg"},
        {"age_min": 12, "age_max": 200,"dose": "500 mg day 1, then 250 mg daily", "max": "1.5 g/course"},
    ],
    "aspirin": [
        {"age_min": 12, "age_max": 200,"dose": "75–325 mg once daily", "max": "4 g/day"},
    ],
    "metformin": [
        {"age_min": 10, "age_max": 18, "dose": "500 mg twice daily", "max": "2 g/day"},
        {"age_min": 18, "age_max": 200,"dose": "500–1000 mg twice daily", "max": "2.5 g/day"},
    ],
    "omeprazole": [
        {"age_min": 1,  "age_max": 12, "dose": "0.7–3.5 mg/kg/day", "max": "20 mg/day"},
        {"age_min": 12, "age_max": 200,"dose": "20–40 mg daily", "max": "40 mg/day"},
    ],
    "prednisone": [
        {"age_min": 1,  "age_max": 12, "dose": "0.5–2 mg/kg/day", "max": "60 mg/day"},
        {"age_min": 12, "age_max": 200,"dose": "5–60 mg daily", "max": "80 mg/day"},
    ],
    "ciprofloxacin": [
        {"age_min": 1,  "age_max": 12, "dose": "10 mg/kg every 12h", "max": "500 mg/dose"},
        {"age_min": 12, "age_max": 200,"dose": "500–750 mg every 12h", "max": "1500 mg/day"},
    ],
    "levothyroxine": [
        {"age_min": 0,  "age_max": 1,  "dose": "10–15 mcg/kg/day", "max": "50 mcg/day"},
        {"age_min": 1,  "age_max": 12, "dose": "4–6 mcg/kg/day", "max": "100 mcg/day"},
        {"age_min": 12, "age_max": 200,"dose": "50–200 mcg/day", "max": "200 mcg/day"},
    ],
    "atorvastatin": [
        {"age_min": 10, "age_max": 18, "dose": "10–20 mg daily", "max": "20 mg/day"},
        {"age_min": 18, "age_max": 200,"dose": "10–80 mg daily", "max": "80 mg/day"},
    ],
    "lisinopril": [
        {"age_min": 6,  "age_max": 16, "dose": "0.07 mg/kg once daily", "max": "5 mg/day"},
        {"age_min": 16, "age_max": 200,"dose": "5–40 mg once daily", "max": "40 mg/day"},
    ]
}

ALTERNATIVE_MAP = {
    "warfarin": ["dabigatran", "apixaban", "rivaroxaban"],
    "ibuprofen": ["paracetamol"],
}

# [Keep all existing utility functions - adjust_max_for_age_weight, calculate_dose, etc.]
def adjust_max_for_age_weight(max_str, age, weight):
    """Reduce maximum doses conservatively for elderly and low weight patients."""
    match = re.search(r"(\d+(\.\d+)?)\s*(mg|mcg|g)", max_str)
    if not match:
        return max_str

    value = float(match.group(1))
    original_unit = match.group(3)

    if original_unit == "g":
        value *= 1000
        unit = "mg"
    else:
        unit = original_unit

    if age >= 80:
        value *= 0.6
    elif age >= 65:
        value *= 0.75

    if weight < 50:
        value *= 0.75

    if original_unit == "g" or (unit == "mg" and value >= 1000):
        return f"{value/1000:.1f} g/day (adjusted)"
    else:
        return f"{value:.0f} {unit}/day (adjusted)"

def calculate_dose(dose_str, weight, age):
    """Detect weight-based dose and calculate safely; otherwise return conservative fixed dose."""
    match = re.search(r"(\d+(\.\d+)?)(–(\d+(\.\d+)?))?\s*(mg|mcg|g)/kg", dose_str)
    if match:
        low = float(match.group(1))
        high = float(match.group(4)) if match.group(4) else low
        unit = match.group(6)
        dose_low = low * weight
        dose_high = high * weight
        if dose_low == dose_high:
            calc_dose = f"{dose_low:.1f} {unit}"
        else:
            calc_dose = f"{dose_low:.1f}–{dose_high:.1f} {unit}"
        suffix = dose_str[dose_str.find(unit + "/kg") + len(unit) + 3:]
        return calc_dose + suffix
    else:
        range_match = re.search(r"(\d+)(–(\d+))?\s*(mg|mcg|g)", dose_str)
        if range_match:
            low = int(range_match.group(1))
            high = int(range_match.group(3)) if range_match.group(3) else low
            unit = range_match.group(4)
            if age >= 65:
                chosen = low
            else:
                chosen = (low + high) // 2
            suffix = dose_str[dose_str.find(unit) + len(unit):]
            return f"{chosen} {unit}{suffix}"
        return dose_str

def estimate_weight_by_age(age_years: float) -> float:
    if age_years < 0.1:
        return 4.0
    elif age_years < 1:
        return 8.0
    elif age_years < 5:
        return 15.0
    elif age_years < 12:
        return 35.0
    else:
        return 70.0

def recommend_dose(drug_name: str, age_years: float, weight_kg: Optional[float] = None) -> Dict[str, Any]:
    key = drug_name.lower()
    rules = medications.get(key)
    if not rules:
        return {"error": f"No dosage data available for '{drug_name}' in medications."}

    rule = None
    for r in rules:
        if r["age_min"] <= age_years < r["age_max"]:
            rule = r
            break
    if not rule:
        return {"error": f"No dosing rule found for {drug_name} at age {age_years} years."}

    if weight_kg is None:
        weight_kg = estimate_weight_by_age(age_years)

    dosage = calculate_dose(rule["dose"], weight_kg, age_years)
    max_dose_adjusted = adjust_max_for_age_weight(rule.get("max", ""), age_years, weight_kg)

    return {
        "drug": drug_name,
        "age_years": age_years,
        "weight_kg": weight_kg,
        "dosage": dosage,
        "max_per_day": max_dose_adjusted
    }

def suggest_alternatives(drug_name: str):
    return ALTERNATIVE_MAP.get(drug_name.lower(), [])

def extract_med_info(text: str):
    """Simple spaCy-based extraction template."""
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)
        meds = []
        pattern = r"([A-Za-z0-9\-\_ ]+?)\s+(\d+(?:\.\d+)?\s*(?:mg|mcg|g|units|ml))"
        matches = re.findall(pattern, text, flags=re.I)
        for m in matches:
            meds.append({"name": m[0].strip(), "dose": m[1].strip()})
        return {"extracted_medications": meds, "note": "This is a demo extractor; use medSpaCy for production."}
    except Exception as e:
        return {"error": "spaCy not installed or model missing; see app requirements."}

# ----------------------------
# Enhanced UI Components (keeping existing ones)
# ----------------------------
def create_header():
    st.markdown("""
    <div class="main-header">
        <h1>💊 Drug Safety Toolkit Pro</h1>
        <p>Advanced medication analysis with AI-powered insights and safety checks</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
