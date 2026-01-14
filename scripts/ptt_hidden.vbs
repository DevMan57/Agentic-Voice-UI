' PTT Launcher - Runs the microphone listener completely hidden
' This script launches ptt_windows.py with no visible window
' v3 - Uses py launcher for maximum compatibility, then falls back to PATH

Set WshShell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

' Get the directory where this script is located
scriptPath = fso.GetParentFolderName(WScript.ScriptFullName)
projectPath = fso.GetParentFolderName(scriptPath)  ' Go up one level to project root
pythonScript = projectPath & "\audio\ptt_windows.py"
logFile = projectPath & "\recordings\ptt_launch.log"

Sub WriteLog(msg)
    On Error Resume Next
    Set logStream = fso.OpenTextFile(logFile, 8, True)
    logStream.WriteLine Now() & " - " & msg
    logStream.Close
End Sub

Function TryLaunch(command, hidden)
    TryLaunch = False
    On Error Resume Next
    If hidden Then
        WshShell.Run command, 0, False
    Else
        WshShell.Run command, 1, False
    End If
    If Err.Number = 0 Then
        WriteLog "SUCCESS: " & command
        TryLaunch = True
    Else
        WriteLog "FAILED: " & command & " - Error " & Err.Number & ": " & Err.Description
        Err.Clear
    End If
End Function

' Clear old log
On Error Resume Next
If fso.FileExists(logFile) Then fso.DeleteFile(logFile)

WriteLog "Starting PTT launcher..."
WriteLog "Script: " & pythonScript

' Try py launcher first (works with any Python installation from python.org)
' py -0 lists installed versions, py -3.X runs specific version, py runs default
If TryLaunch("py -B """ & pythonScript & """ --quiet", True) Then WScript.Quit 0

' Try pythonw from PATH
If TryLaunch("pythonw """ & pythonScript & """ --quiet", True) Then WScript.Quit 0

' Try python from PATH
If TryLaunch("python """ & pythonScript & """ --quiet", True) Then WScript.Quit 0

' Try explicit Python paths as last resort (for miniconda/anaconda users)
userProfile = WshShell.ExpandEnvironmentStrings("%USERPROFILE%")
altPaths = Array( _
    userProfile & "\miniconda3\pythonw.exe", _
    userProfile & "\miniconda3\python.exe", _
    userProfile & "\anaconda3\pythonw.exe", _
    userProfile & "\anaconda3\python.exe", _
    userProfile & "\AppData\Local\Programs\Python\Python313\pythonw.exe", _
    userProfile & "\AppData\Local\Programs\Python\Python313\python.exe", _
    userProfile & "\AppData\Local\Programs\Python\Python312\pythonw.exe", _
    userProfile & "\AppData\Local\Programs\Python\Python312\python.exe", _
    userProfile & "\AppData\Local\Programs\Python\Python311\pythonw.exe", _
    userProfile & "\AppData\Local\Programs\Python\Python311\python.exe" _
)

For Each path In altPaths
    If fso.FileExists(path) Then
        If TryLaunch("""" & path & """ """ & pythonScript & """ --quiet", True) Then
            WScript.Quit 0
        End If
    End If
Next

WriteLog "FAILED: All launch attempts failed!"
WriteLog "Please install Python from python.org or ensure Python is in your PATH"
WScript.Quit 1
