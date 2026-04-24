@echo off
cd /d c:\vscode\predict

echo Building exe with PyInstaller...
pyinstaller --clean discriminant_analysis_gui.spec

if exist "dist\discriminant_analysis_gui.exe" (
    echo Compressing with UPX...
    c:\upx\upx.exe -9 "dist\discriminant_analysis_gui.exe"
    echo.
    echo Build completed successfully!
    echo Final exe location: dist\discriminant_analysis_gui.exe
    echo Size:
    dir "dist\discriminant_analysis_gui.exe" | findstr "discriminant_analysis_gui.exe"
) else (
    echo Build failed. Exe not found.
)

pause