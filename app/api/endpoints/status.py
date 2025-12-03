"""
TTS processing status endpoints
"""

from typing import Optional, Dict, Any, List
from fastapi import APIRouter, Query

from app.models import TTSStatusResponse, TTSStatisticsResponse, APIInfoResponse
from app.core import (
    add_route_aliases,
    get_tts_status,
    get_tts_history,
    get_tts_statistics,
    clear_tts_history,
    get_version,
    get_version_info
)

# Create router with aliasing support
base_router = APIRouter()
router = add_route_aliases(base_router)


@router.get(
    "/status",
    response_model=TTSStatusResponse,
    summary="Get TTS processing status",
    description="Get current TTS processing status, progress, and information"
)
async def get_processing_status(
    include_history: bool = Query(False, description="Include recent request history"),
    include_stats: bool = Query(False, description="Include processing statistics"),
    history_limit: int = Query(5, description="Number of history records to return", ge=1, le=20)
) -> Dict[str, Any]:
    """Get comprehensive TTS processing status information"""
    
    # Get base status
    status = get_tts_status()
    
    
    # Add request history if requested
    if include_history:
        try:
            status["request_history"] = get_tts_history(history_limit)
        except Exception as e:
            status["request_history"] = {"error": f"Failed to get history: {str(e)}"}
    
    # Add processing statistics if requested
    if include_stats:
        try:
            status["statistics"] = get_tts_statistics()
        except Exception as e:
            status["statistics"] = {"error": f"Failed to get statistics: {str(e)}"}
    
    return status


@router.get(
    "/info",
    response_model=APIInfoResponse,
    summary="Get API info and status",
    description="Get comprehensive API information including TTS status and statistics"
)
async def get_api_info() -> Dict[str, Any]:
    """Get comprehensive API information"""
    try:
        # Get version information
        version_info = get_version_info()
        
        # Get all information
        tts_status = get_tts_status()
        tts_stats = get_tts_statistics()
        recent_history = get_tts_history(3)  # Last 3 requests
        
        return {
            "api_name": version_info.get("name", "Chatterbox TTS API"),
            "version": version_info["version"],
            "api_version": version_info["api_version"],
            "description": version_info.get("description", ""),
            "status": "operational",
            "version_info": version_info,
            "tts_status": tts_status,
            "statistics": tts_stats,
            "recent_requests": recent_history,
            "uptime_info": {
                "total_requests": tts_stats.get("total_requests", 0),
                "success_rate": tts_stats.get("success_rate", 0),
                "is_processing": tts_status.get("is_processing", False)
            }
        }
    except Exception as e:
        version = get_version()
        return {
            "api_name": "Chatterbox TTS API",
            "version": version,
            "api_version": version,
            "status": "error",
            "error": f"Failed to get API info: {str(e)}"
        }


# Export the base router for the main app to use
__all__ = ["base_router"] 