"""Feedback API endpoints for user corrections."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/feedback", tags=["feedback"])

# Storage path for feedback (in production, use a database)
FEEDBACK_FILE = Path("feedback_data.json")


class FeedbackRequest(BaseModel):
    """User feedback on format detection."""
    detection_id: str = Field(..., description="ID from detection response")
    correct_format: str = Field(..., description="The correct format name")
    user_comments: Optional[str] = Field(None, description="Additional comments")
    detected_format: str = Field(..., description="What was originally detected")
    confidence_score: float = Field(..., description="Original confidence score")


class CorrectionRequest(BaseModel):
    """Request to correct format mappings."""
    format_name: str = Field(..., description="Format to update")
    column_mappings: dict = Field(..., description="New column mappings")
    reason: Optional[str] = Field(None, description="Reason for correction")


def load_feedback() -> List[dict]:
    """Load existing feedback."""
    if FEEDBACK_FILE.exists():
        with open(FEEDBACK_FILE, 'r') as f:
            return json.load(f)
    return []


def save_feedback(feedback_list: List[dict]):
    """Save feedback to file."""
    with open(FEEDBACK_FILE, 'w') as f:
        json.dump(feedback_list, f, indent=2, default=str)


@router.post("/submit")
async def submit_feedback(feedback: FeedbackRequest):
    """Submit feedback on a format detection result."""
    try:
        # Load existing feedback
        all_feedback = load_feedback()
        
        # Add new feedback
        feedback_entry = {
            "id": f"fb_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "timestamp": datetime.now().isoformat(),
            "detection_id": feedback.detection_id,
            "detected_format": feedback.detected_format,
            "correct_format": feedback.correct_format,
            "confidence_score": feedback.confidence_score,
            "user_comments": feedback.user_comments
        }
        
        all_feedback.append(feedback_entry)
        save_feedback(all_feedback)
        
        logger.info(f"Feedback received: {feedback.detected_format} → {feedback.correct_format}")
        
        return {
            "success": True,
            "message": "Feedback recorded successfully",
            "feedback_id": feedback_entry["id"]
        }
        
    except Exception as e:
        logger.error(f"Error saving feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/correction")
async def submit_correction(correction: CorrectionRequest):
    """Submit a correction for format mappings."""
    try:
        # In production, this would update the database
        correction_entry = {
            "id": f"corr_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "timestamp": datetime.now().isoformat(),
            "format_name": correction.format_name,
            "column_mappings": correction.column_mappings,
            "reason": correction.reason
        }
        
        # For now, just log it
        logger.info(f"Correction received for format: {correction.format_name}")
        
        return {
            "success": True,
            "message": "Correction recorded for review",
            "correction_id": correction_entry["id"]
        }
        
    except Exception as e:
        logger.error(f"Error saving correction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def feedback_stats():
    """Get feedback statistics."""
    try:
        all_feedback = load_feedback()
        
        if not all_feedback:
            return {"total_feedback": 0, "corrections_needed": []}
        
        # Calculate stats
        format_corrections = {}
        for fb in all_feedback:
            key = f"{fb['detected_format']} → {fb['correct_format']}"
            format_corrections[key] = format_corrections.get(key, 0) + 1
        
        return {
            "total_feedback": len(all_feedback),
            "unique_corrections": len(format_corrections),
            "top_corrections": sorted(
                [{"correction": k, "count": v} for k, v in format_corrections.items()],
                key=lambda x: x["count"],
                reverse=True
            )[:10]
        }
        
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

