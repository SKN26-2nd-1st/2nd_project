# =========================================================================
# Project:
# - CCRM 
#
# Module:
# - scripts
#
# File: run-dev.ps1
#
# Purpose:
# - CCRM 개발 환경 실행 스크립트
# - Docker Compose develop profile 실행
#
# Description:
# - ml_cpu + streamlit 컨테이너 실행
# - config/.env.dev 환경 변수 파일 사용
# - 개발용 UI 및 notebook 환경 구성
#
# Usage:
# - PowerShell 실행
# - ./scripts/run-dev.ps1
#
# Author: @nobrain711
# Created: 2026-03-12
#
# Updated:
# - 2026-03-12: initial version (@nobrain711)
# =========================================================================

$envFile = "config/.env.dev"
$profile = "develop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host " Starting CCRM Development Environment" -ForegroundColor Green
Write-Host " Profile : $profile"
Write-Host " EnvFile : $envFile"
Write-Host "========================================" -ForegroundColor Cyan


try {
    docker compose `
        --env-file $envFile `
        --profile $profile `
        up -d
}
catch {
    Write-Host "docker compose failed, trying docker-compose..." -ForegroundColor Red
    
    docker-compose `
        --env-file $envFile `
        --profile $profile `
        up -d

}

Write-Host ""
Write-Host "Running containers:" -ForegroundColor Yellow

try {
    docker compose ps
}
catch {
    Write-Host "docker compose failed, trying docker-compose..." -ForegroundColor Red
    docker-compose ps
}


Write-Host ""
Write-Host "CCRM develop environment started." -ForegroundColor Green