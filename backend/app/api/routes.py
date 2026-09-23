"""
PhishGuard AI — API Routes
All API endpoints for phishing detection, history, and statistics.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.app.detectors.url_detector import analyze_url, extract_url_features
from backend.app.detectors.message_detector import analyze_message
from backend.app.detectors.behavioral_detector import analyze_behavior
from backend.app.services.risk_aggregator import aggregate_risk
from backend.app.ml.predictor import predictor
from backend.app.database.db import save_scan, get_history, get_statistics

router = APIRouter()


# --- Request/Response Models ---

class URLRequest(BaseModel):
    url: str = Field(..., min_length=3, description="URL to analyze")


class MessageRequest(BaseModel):
    message: str = Field(..., min_length=5, description="Message/email text to analyze")


class AnalysisResponse(BaseModel):
    classification: str
    risk_score: int
    ml_probability: float
    indicators: list[str]
    recommendations: list[str]
    explanation: str
    score_breakdown: dict
    input_type: str
    scan_id: int


# --- Endpoints ---

@router.get("/health")
def health_check():
    """Health check endpoint."""
    model_status = predictor.get_status()
    return {
        "status": "healthy",
        "service": "PhishGuard AI",
        "version": "1.0.0",
        "ml_models": model_status,
    }


@router.post("/analyze/url", response_model=AnalysisResponse)
def analyze_url_endpoint(request: URLRequest):
    """
    Analyze a URL for phishing indicators.
    Pipeline: URL → feature extraction → URL analysis → ML prediction → risk aggregation → result
    """
    try:
        url = request.url.strip()

        # Step 1: URL feature extraction and heuristic analysis
        url_analysis = analyze_url(url)

        # Step 2: ML prediction using extracted features
        url_features = url_analysis["features"]
        ml_probability = predictor.predict_url(url_features)

        # Step 3: Behavioral analysis (on the URL itself as text)
        behavioral = analyze_behavior(url)

        # Step 4: Risk aggregation
        all_indicators = url_analysis["indicators"] + behavioral["indicators"]
        # Deduplicate indicators
        all_indicators = list(dict.fromkeys(all_indicators))

        result = aggregate_risk(
            ml_probability=ml_probability,
            url_indicator_score=url_analysis["indicator_score"],
            content_indicator_score=0,  # No content for pure URL analysis
            behavioral_score=behavioral["behavioral_score"],
            all_indicators=all_indicators,
            analysis_mode="url",
        )

        # Step 5: Save to database
        scan_id = save_scan(
            input_type="url",
            input_text=url,
            classification=result["classification"],
            risk_score=result["risk_score"],
            ml_probability=result["ml_probability"],
            indicators=result["indicators"],
            recommendations=result["recommendations"],
            explanation=result["explanation"],
        )

        return AnalysisResponse(
            classification=result["classification"],
            risk_score=result["risk_score"],
            ml_probability=result["ml_probability"],
            indicators=result["indicators"],
            recommendations=result["recommendations"],
            explanation=result["explanation"],
            score_breakdown=result["score_breakdown"],
            input_type="url",
            scan_id=scan_id,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")


@router.post("/analyze/message", response_model=AnalysisResponse)
def analyze_message_endpoint(request: MessageRequest):
    """
    Analyze a message/email for phishing indicators.
    Pipeline: Message → content analysis → behavioral analysis → ML prediction → risk aggregation → result
    """
    try:
        message = request.message.strip()

        # Step 1: Content/keyword analysis (also extracts and analyzes embedded URLs)
        content_analysis = analyze_message(message)

        # Step 2: Behavioral analysis
        behavioral = analyze_behavior(message)

        # Step 3: ML prediction on message text
        ml_probability = predictor.predict_message(message)

        # Step 4: Get URL indicator score from any embedded URLs
        url_indicator_score = 0
        if content_analysis["url_analysis"]:
            # Use the max score from analyzed URLs
            url_scores = [ua["indicator_score"] for ua in content_analysis["url_analysis"]]
            url_indicator_score = max(url_scores) if url_scores else 0

        # Step 5: Risk aggregation
        all_indicators = content_analysis["indicators"] + behavioral["indicators"]
        # Deduplicate indicators
        all_indicators = list(dict.fromkeys(all_indicators))

        result = aggregate_risk(
            ml_probability=ml_probability,
            url_indicator_score=url_indicator_score,
            content_indicator_score=content_analysis["indicator_score"],
            behavioral_score=behavioral["behavioral_score"],
            all_indicators=all_indicators,
            analysis_mode="message",
        )

        # Step 6: Save to database
        scan_id = save_scan(
            input_type="message",
            input_text=message,
            classification=result["classification"],
            risk_score=result["risk_score"],
            ml_probability=result["ml_probability"],
            indicators=result["indicators"],
            recommendations=result["recommendations"],
            explanation=result["explanation"],
        )

        return AnalysisResponse(
            classification=result["classification"],
            risk_score=result["risk_score"],
            ml_probability=result["ml_probability"],
            indicators=result["indicators"],
            recommendations=result["recommendations"],
            explanation=result["explanation"],
            score_breakdown=result["score_breakdown"],
            input_type="message",
            scan_id=scan_id,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")


@router.get("/history")
def get_history_endpoint(limit: int = 50):
    """Get scan history from the database."""
    try:
        history = get_history(limit=limit)
        return {"history": history, "count": len(history)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@router.get("/statistics")
def get_statistics_endpoint():
    """Get scan statistics from the database."""
    try:
        stats = get_statistics()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
