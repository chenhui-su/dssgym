@echo off
setlocal

REM 已改为项目内相对调用，可直接透传 ppo_agent.py 参数
REM 示例:
REM   test.bat --model_path results\xxx\model\ppo_model.zip --env_name 13Bus_cbat --test_only true
REM   test.bat --env_name 34Bus --ev_demand_path ev_demand\ev_demand-public_parking-general-250-A95.csv --test_only true

cd /d "%~dp0"
python ppo_agent.py %*

if errorlevel 1 (
  echo.
  echo 命令执行失败，请检查参数和当前 Python 环境依赖是否完整。
)

pause
