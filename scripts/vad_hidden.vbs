' VAD Hidden Launcher
' Runs vad_windows.py in the background without a visible window
' Use this alongside ptt_hidden.vbs for both input modes

Set WshShell = CreateObject("WScript.Shell")
Set FSO = CreateObject("Scripting.FileSystemObject")

' Get script directory and project root
scriptPath = FSO.GetParentFolderName(WScript.ScriptFullName)
projectPath = FSO.GetParentFolderName(scriptPath)  ' Go up one level to project root

' Change to project directory
WshShell.CurrentDirectory = projectPath

' Launch VAD in hidden window
' Use pythonw.exe for no console window
WshShell.Run "pythonw.exe audio\vad_windows.py", 0, False
