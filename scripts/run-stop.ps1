# =========================================================================
# Project:
# - CCRM
#
# Script:
# - run-stop.ps1
#
# Purpose:
# - 전체 스택 중지
# - mysql + pipeline + mlflow + ml_cpu + streamlit
#
# Author: @nobrain711
# Created: 2026-03-12
#
# Updated:
# - 2026-03-12: initial version (@nobrain711)
# =========================================================================

Write-Host "Stopping CCRM containers..." -ForegroundColor Yellow

try {
    docker compose down
}
catch {
    Write-Host "docker compose failed, trying docker-compose..." -ForegroundColor Red
    docker-compose down
}

Write-Host "Docker environment stopped." -ForegroundColor Green