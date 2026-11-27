* --- GDXXRW needs Windows and Excel. On macOS/Linux, %gdxxrw% calls the
*     openpyxl based replacement mac/gdxxrw.py with the embedded GAMS Python.
  $$ifi "%system.filesys%"=="UNIX"     $$setglobal gdxxrw %gams.sysdir%GMSPython/bin/python3 mac/gdxxrw.py
  $$ifi not "%system.filesys%"=="UNIX" $$setglobal gdxxrw gdxxrw.exe
