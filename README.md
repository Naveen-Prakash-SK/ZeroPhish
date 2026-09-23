# 🛡️ ZeroPhish

### AI-Based Phishing Detection System

ZeroPhish is an AI-powered cybersecurity platform designed to detect **phishing URLs and suspicious messages** using machine learning, security indicators, and explainable risk analysis.

---

## 🚨 Problem

Phishing attacks use deceptive websites, URLs, and messages to trick users into revealing sensitive information.

ZeroPhish provides a simple interface where users can analyze suspicious content **before interacting with it**.

---

## 💡 Solution

ZeroPhish combines:

- 🤖 Machine Learning
- 🔗 URL Analysis
- 💬 Message Analysis
- 🔍 Explainable Threat Indicators
- 📊 Risk Scoring
- 📜 Scan History
- 📈 Security Dashboard

into a single web-based cybersecurity platform.

---

## ✨ Features

### 🔗 URL Scanner

Analyze URLs using multiple security and lexical characteristics such as:

- HTTPS usage
- URL length
- Domain structure
- Subdomains
- Suspicious TLDs
- IP-based URLs
- Suspicious keywords
- Special characters
- URL shortening
- Brand impersonation indicators

### 💬 Message Scanner

Detect suspicious messages using indicators such as:

- Urgency
- Account verification requests
- Credential requests
- Financial-related requests
- Suspicious links
- Phishing-related keywords
- Social-engineering patterns

### 📊 Risk Analysis

Every scan produces:

**Risk Score → 0–100**

| Score | Result |
|---|---|
| 🟢 0–30 | SAFE |
| 🟡 31–60 | SUSPICIOUS |
| 🔴 61–100 | PHISHING |

The system also provides explainable indicators to help users understand why content was flagged.

---

## 🧠 Machine Learning

The URL detection model uses a **Random Forest Classifier** trained using the:

**UCI PhiUSIIL Phishing URL Dataset**

### Dataset

- 235,795 original records
- 235,370 records after duplicate removal
- 100,520 phishing URLs
- 134,850 legitimate URLs

### Held-Out Test Performance

| Metric | Result |
|---|---:|
| Accuracy | **99.60%** |
| Precision | **99.89%** |
| Recall | **99.16%** |
| F1 Score | **99.53%** |

> Performance reported above is from the held-out UCI dataset test split and does not represent guaranteed real-world accuracy.

---

## 🏗️ Architecture

```text
                    ┌──────────────────┐
                    │      USER        │
                    │ URL / MESSAGE    │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ React Frontend   │
                    │   ZeroPhish UI   │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │   FastAPI API    │
                    │     Backend      │
                    └────────┬─────────┘
                             │
                 ┌───────────┴───────────┐
                 ▼                       ▼
        ┌────────────────┐      ┌────────────────┐
        │ URL Detection  │      │Message Analysis│
        │ Random Forest  │      │ NLP + Rules    │
        └───────┬────────┘      └───────┬────────┘
                │                       │
                └───────────┬───────────┘
                            ▼
                   ┌──────────────────┐
                   │  Risk Assessment │
                   │ Score + Reasons  │
                   └────────┬─────────┘
                            │
                            ▼
                   ┌──────────────────┐
                   │ SQLite Database  │
                   │ History & Stats  │
                   └──────────────────┘
