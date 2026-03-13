# =========================================================================
# Project:
# - CCRM
#
# Script:
# - run-full.ps1
#
# Purpose:
# - 전체 스택 실행
# - mysql + pipeline + mlflow + ml_cpu + streamlit
#
# Author: @nobrain711
# Created: 2026-03-12
#
# Updated:
# - 2026-03-12: initial version (@nobrain711)
# =========================================================================

# Environment configutation
$envFile = "config/.env.full"
$profile = "full"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host " Starting CCRM Full Environment" -ForegroundColor Green
Write-Host " Profile : $profile"
Write-Host " EnvFile : $envFile"
Write-Host "========================================" -ForegroundColor Cyan

# Docker Compose up
try {
    docker compose `
        --env-file $envFile `
        --profile $profile `
        up -d
}
catch {
    Write-Host "docker compose failed, trying docker-compose..." -ForegroundColor Yellow

    docker-compose `
        --env-file $envFile `
        --profile $profile `
        up -d

}

# Docker Compose status
Write-Host ""
Write-Host "Running containers:" -ForegroundColor Yellow

try {
    docker compose ps
}
catch {
    Write-Host "docker-compose failed, trying docker compose..." -ForegroundColor Yellow
    docekr-compose ps
}

Write-Host ""
Write-Host "CCRM full environment started." -ForegroundColor Green